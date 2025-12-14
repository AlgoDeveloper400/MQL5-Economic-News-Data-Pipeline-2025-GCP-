import os
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.cuda.amp import autocast
from contextlib import nullcontext
from sklearn.metrics import r2_score, mean_squared_error
import mlflow

# Database connector (root access)
from db_connector import load_events_data, save_test_forecasts, save_live_forecasts, is_first_live_forecasts_run

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Helper Functions --------------------
def normalize_feature(arr):
    arr = np.asarray(arr, dtype=float)
    if np.all(np.isnan(arr)):
        return np.zeros_like(arr), 0.0, 1.0
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    rng = mx - mn if mx != mn else 1.0
    return (arr - mn) / rng, mn, rng

# -------------------- RNN Model --------------------
class RNNModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1])
        return self.fc(out).squeeze(-1)

# -------------------- VECTORIZED TEST FUNCTION --------------------
def run_test_vectorized(test_df, event_model_types, rnn_model, xgb_model, seq_length, device):
    """Vectorized test function for significant performance improvement"""
    
    amp_ctx = autocast if device.type=="cuda" else nullcontext
    forecast_rows = []
    live_forecast_rows = []

    # Process each currency-event group
    for key, group in test_df.groupby(['Currency','Event']):
        group = group.sort_values('DateTime')
        n = len(group)
        if n < 2:
            continue

        model_type = event_model_types.get(key, "xgb")

        actuals = group['Actual_numeric'].values
        previous = group['Previous_numeric'].values
        impacts = group['Impact'].astype(float).values
        hicount = group['HighImpactCount'].astype(float).values
        actual_lag1 = group['Actual_lag1'].values
        prev_lag1 = group['Previous_lag1'].values

        norm_actuals, mn_act, rng_act = normalize_feature(actuals)
        norm_previous, _, _ = normalize_feature(previous)
        norm_impacts, _, _ = normalize_feature(impacts)
        norm_hic, _, _ = normalize_feature(hicount)
        norm_al1, _, _ = normalize_feature(actual_lag1)
        norm_pl1, _, _ = normalize_feature(prev_lag1)

        features = np.stack([norm_actuals, norm_previous, norm_impacts, norm_hic, norm_al1, norm_pl1], axis=1)

        preds_scaled, trues_scaled = [], []

        # Generate predictions in batches based on model type
        if model_type == "rnn" and rnn_model:
            rnn_model.eval()
            with torch.no_grad():
                # Create all RNN sequences at once
                sequences = []
                targets = []
                for i in range(seq_length, n):
                    sequences.append(features[i-seq_length:i])
                    targets.append(norm_actuals[i])
                
                if sequences:
                    # Convert to batch tensor
                    sequences_tensor = torch.tensor(np.array(sequences), 
                                                  dtype=torch.float32).to(device)
                    with amp_ctx():
                        batch_preds = rnn_model(sequences_tensor).cpu().numpy()
                    
                    preds_scaled.extend(batch_preds)
                    trues_scaled.extend(targets)
                    
                    # Create live forecast entries for RNN - only latest forecast
                    if len(batch_preds) > 0:
                        latest_pred = batch_preds[-1]
                        forecast_value = latest_pred * rng_act + mn_act
                        live_forecast_rows.append({
                            'Event': key[1],
                            'Currency': key[0],
                            'ForecastValue': float(forecast_value)
                        })
                    
        elif model_type == "xgb" and xgb_model:
            # Process all XGBoost samples at once
            if n > 0:
                batch_preds = xgb_model.predict(features)
                preds_scaled.extend(batch_preds)
                trues_scaled.extend(norm_actuals)
                
                # Create live forecast entries for XGBoost - only latest forecast
                if len(batch_preds) > 0:
                    latest_pred = batch_preds[-1]
                    forecast_value = latest_pred * rng_act + mn_act
                    live_forecast_rows.append({
                        'Event': key[1],
                        'Currency': key[0],
                        'ForecastValue': float(forecast_value)
                    })
        else:
            continue

        if len(trues_scaled) > 1:
            # Denormalize all predictions at once
            preds = [p * rng_act + mn_act for p in preds_scaled]
            trues = [t * rng_act + mn_act for t in trues_scaled]
            
            # Vectorized metric calculation
            r2 = r2_score(trues, preds)
            mse = mean_squared_error(trues, preds)
            samples = len(trues)
            forecast_rows.append((key[0], key[1], r2, mse, samples))

    return forecast_rows, live_forecast_rows

# -------------------- Main Test Function --------------------
def run_test(output_folder):
    # Load parameters from saved training data (NOT from MLflow)
    print("Loading parameters from training data splits...")
    data_splits_file = os.path.join(output_folder, "data_splits.joblib")
    if not os.path.exists(data_splits_file):
        raise FileNotFoundError("No data splits found. Please run training first.")

    data_splits = joblib.load(data_splits_file)
    params = data_splits['training_params']
    seq_length = params["SEQ_LENGTH"]

    with mlflow.start_run():
        mlflow.log_params(params)

        # Load test data and model assignments
        print("Loading test data and model assignments...")
        test_df = data_splits['test_df']
        event_model_types = data_splits['event_model_types']

        print(f"✓ Loaded {len(test_df)} test samples")
        print(f"  Test date range: {test_df['DateTime'].min()} to {test_df['DateTime'].max()}")

        # Load models and normalization params
        print("Loading models and normalization parameters...")
        rnn_file = os.path.join(output_folder, "rnn_model.pt")
        xgb_file = os.path.join(output_folder, "xgb_model.joblib")
        event_norm_file = os.path.join(output_folder, "event_norm_params.joblib")

        event_norm_params = joblib.load(event_norm_file) if os.path.exists(event_norm_file) else {}

        rnn_model = None
        if os.path.exists(rnn_file):
            rnn_model = RNNModel()
            rnn_model.load_state_dict(torch.load(rnn_file, map_location=device))
            rnn_model.to(device)
            rnn_model.eval()

        xgb_model = joblib.load(xgb_file) if os.path.exists(xgb_file) else None

        # Prepare test features
        test_df = test_df.sort_values(['Currency','Event','DateTime'])
        test_df['EventDate'] = test_df['DateTime'].dt.date

        # Compute high impact counts
        high_impact_counts = (
            test_df[test_df['Impact']==3]
            .groupby(['Currency','EventDate'])
            .size()
            .rename('HighImpactCount')
            .reset_index()
        )
        test_df = pd.merge(test_df, high_impact_counts, how='left', on=['Currency','EventDate'])
        test_df['HighImpactCount'] = test_df['HighImpactCount'].fillna(0)

        test_df['Actual_lag1'] = test_df.groupby(['Currency','Event'])['Actual_numeric'].shift(1)
        test_df['Previous_lag1'] = test_df.groupby(['Currency','Event'])['Previous_numeric'].shift(1)
        test_df['Actual_lag1'] = test_df.groupby(['Currency','Event'])['Actual_lag1'].bfill().ffill()
        test_df['Previous_lag1'] = test_df.groupby(['Currency','Event'])['Previous_lag1'].bfill().ffill()

        # -------------------- Generate Test Forecasts (VECTORIZED) --------------------
        print("Running VECTORIZED test forecasting...")
        forecast_rows, live_forecast_rows = run_test_vectorized(
            test_df, event_model_types, rnn_model, xgb_model, seq_length, device
        )

        # -------------------- Save Test Forecasts --------------------
        if forecast_rows:
            forecast_df = pd.DataFrame(forecast_rows, columns=["Currency", "Event", "R2", "MSE", "Samples"])
            save_test_forecasts(forecast_df)
            print(f"✓ Saved {len(forecast_rows)} test forecast metrics to database (REPLACED existing data)")

            mlflow.log_metric("test_r2_mean", forecast_df["R2"].mean())
            mlflow.log_metric("test_mse_mean", forecast_df["MSE"].mean())
            mlflow.log_metric("test_samples_total", forecast_df["Samples"].sum())
            
            # Print some sample metrics
            print(f"  Sample test metrics:")
            for i, row in enumerate(forecast_rows[:3]):
                print(f"    {row[0]}, {row[1]}: R2: {row[2]:.4f}, MSE: {row[3]:.4f}, Samples: {row[4]}")

        # -------------------- Save Live Forecasts --------------------
        if live_forecast_rows:
            live_forecast_df = pd.DataFrame(live_forecast_rows)
            
            # Ensure we only have Event, Currency, ForecastValue columns
            required_columns = ['Event', 'Currency', 'ForecastValue']
            live_forecast_df = live_forecast_df[required_columns]
            
            # Check if this is the first run for live forecasts
            is_first_run = is_first_live_forecasts_run()
            print(f"  Live forecasts: First run = {is_first_run}")
            
            save_live_forecasts(live_forecast_df, is_first_run=is_first_run)
            print(f"✓ Saved {len(live_forecast_rows)} live forecast records to database")
            
            # Print some sample live forecasts
            print(f"  Sample live forecasts:")
            for i, row in enumerate(live_forecast_rows[:3]):
                print(f"    {row['Currency']}, {row['Event']}: ForecastValue: {row['ForecastValue']:.4f}")

        return {
            "message": "Testing complete", 
            "forecast_rows": len(forecast_rows),
            "live_forecast_rows": len(live_forecast_rows)
        }