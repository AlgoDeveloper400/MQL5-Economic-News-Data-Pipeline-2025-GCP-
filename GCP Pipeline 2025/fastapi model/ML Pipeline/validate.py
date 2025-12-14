import os
import re
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import torch
from torch.cuda.amp import autocast
from contextlib import nullcontext
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import mlflow
from torch import nn

# Import database connector
from db_connector import load_events_data, save_validation_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# -------------------- Helper functions --------------------
def parse_numeric_val(val):
    if pd.isna(val):
        return None
    if isinstance(val,(int,float)):
        return float(val)
    val = str(val).strip()
    if val == "":
        return None
    if val.endswith("%"):
        try:
            return float(val.rstrip("%"))
        except:
            return None
    m = re.match(r"^([-+]?[0-9]*\.?[0-9]+)\s*([KkMmBbTt])$", val)
    if m:
        num = float(m.group(1))
        unit = m.group(2).upper()
        mult = {'K':1e3,'M':1e6,'B':1e9,'T':1e12}.get(unit,1)
        return num * mult
    try:
        return float(val)
    except:
        return None

def normalize_feature(arr):
    arr = np.asarray(arr, dtype=float)
    if np.all(np.isnan(arr)):
        return np.zeros_like(arr), 0.0, 1.0
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    rng = mx - mn if mx != mn else 1.0
    return (arr - mn) / rng, mn, rng

# -------------------- VECTORIZED VALIDATION FUNCTION --------------------
def run_validation_vectorized(all_val_samples, rnn_model, xgb_model, event_norm_params, device):
    """Vectorized validation for significant performance improvement"""
    
    # Separate samples by model type
    rnn_samples = [s for s in all_val_samples if s[3] == "rnn"]
    xgb_samples = [s for s in all_val_samples if s[3] == "xgb"]
    
    # Group by currency-event for efficient processing
    event_data = {}
    
    # Process RNN samples in batches
    if rnn_samples and rnn_model is not None:
        rnn_model.eval()
        with torch.no_grad():
            # Group RNN samples by key
            rnn_by_key = {}
            for features, target, key, _ in rnn_samples:
                if key not in rnn_by_key:
                    rnn_by_key[key] = {'features': [], 'targets': []}
                rnn_by_key[key]['features'].append(features)
                rnn_by_key[key]['targets'].append(target)
            
            # Process each key's RNN samples in batch
            for key, data in rnn_by_key.items():
                if len(data['features']) == 0:
                    continue
                    
                # Convert to batch tensor
                features_tensor = torch.tensor(np.array(data['features']), 
                                             dtype=torch.float32).to(device)
                predictions = rnn_model(features_tensor).cpu().numpy()
                targets = np.array(data['targets'])
                
                if key not in event_data:
                    event_data[key] = {'actuals': [], 'predictions': []}
                
                event_data[key]['actuals'].extend(targets)
                event_data[key]['predictions'].extend(predictions)
    
    # Process XGBoost samples in batches
    if xgb_samples and xgb_model is not None:
        # Group XGBoost samples by key
        xgb_by_key = {}
        for features, target, key, _ in xgb_samples:
            if key not in xgb_by_key:
                xgb_by_key[key] = {'features': [], 'targets': []}
            xgb_by_key[key]['features'].append(features)
            xgb_by_key[key]['targets'].append(target)
        
        # Process each key's XGBoost samples
        for key, data in xgb_by_key.items():
            if len(data['features']) == 0:
                continue
                
            features_array = np.array(data['features'])
            predictions = xgb_model.predict(features_array)
            targets = np.array(data['targets'])
            
            if key not in event_data:
                event_data[key] = {'actuals': [], 'predictions': []}
            
            event_data[key]['actuals'].extend(targets)
            event_data[key]['predictions'].extend(predictions)
    
    # Calculate metrics for each event
    metrics_records = []
    for key, data in event_data.items():
        currency, event = key
        actuals = np.array(data['actuals'])
        predictions = np.array(data['predictions'])
        
        if len(actuals) < 2:
            continue
            
        # Denormalize if needed
        if key in event_norm_params:
            mn, rng = event_norm_params[key]['actual']
            if rng > 0:
                actuals = actuals * rng + mn
                predictions = predictions * rng + mn
        
        # Vectorized metric calculation
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        
        if ss_tot == 0:
            r2 = -1.0
        else:
            r2 = 1 - (ss_res / ss_tot)
        
        mse = np.mean((actuals - predictions) ** 2)
        
        metrics_records.append({
            'Currency': currency,
            'Event': event,
            'R2': float(r2),
            'MSE': float(mse),
            'Samples': len(actuals)
        })
    
    return metrics_records

# -------------------- Main Validation --------------------
def run_validation(output_folder):
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
        
        # Load pre-split validation data and model assignments
        print("Loading validation data and model assignments from training splits...")
        val_df = data_splits['val_df']
        event_model_types = data_splits['event_model_types']
        
        print(f"✓ Loaded {len(val_df)} validation samples")
        print(f"  Validation date range: {val_df['DateTime'].min()} to {val_df['DateTime'].max()}")
        print(f"  Model assignments: {sum(1 for mt in event_model_types.values() if mt == 'rnn')} RNN, "
              f"{sum(1 for mt in event_model_types.values() if mt == 'xgb')} XGBoost")

        # Load models and normalization parameters
        print("Loading models and normalization parameters...")
        rnn_file = os.path.join(output_folder, "rnn_model.pt")
        xgb_file = os.path.join(output_folder, "xgb_model.joblib")
        event_norm_file = os.path.join(output_folder, "event_norm_params.joblib")

        # Check if models exist
        if not os.path.exists(rnn_file) and not os.path.exists(xgb_file):
            raise FileNotFoundError("No trained models found. Please run training first.")

        event_norm_params = joblib.load(event_norm_file) if os.path.exists(event_norm_file) else {}

        rnn_model = None
        if os.path.exists(rnn_file):
            rnn_model = RNNModel()
            rnn_model.load_state_dict(torch.load(rnn_file, map_location=device))
            rnn_model.to(device)
            rnn_model.eval()

        xgb_model = joblib.load(xgb_file) if os.path.exists(xgb_file) else None

        # Prepare validation data features
        print("Preparing validation features...")
        val_df = val_df.sort_values(['Currency', 'Event', 'DateTime'])
        
        # Create features for validation data
        val_df['EventMonth'] = val_df['DateTime'].dt.to_period('M')
        val_df['EventDate'] = val_df['DateTime'].dt.date
        
        # Calculate high impact counts (using only validation data to avoid leakage)
        high_impact_counts = val_df[val_df['Impact']==3].groupby(['Currency','EventDate']).size().rename('HighImpactCount').reset_index()
        val_df = pd.merge(val_df, high_impact_counts, how='left', on=['Currency','EventDate'])
        val_df['HighImpactCount'] = val_df['HighImpactCount'].fillna(0)
        
        val_df['Actual_lag1'] = val_df.groupby(['Currency','Event'])['Actual_numeric'].shift(1)
        val_df['Previous_lag1'] = val_df.groupby(['Currency','Event'])['Previous_numeric'].shift(1)
        
        # Fill lag features
        val_df['Actual_lag1'] = val_df.groupby(['Currency','Event'])['Actual_lag1'].bfill().ffill()
        val_df['Previous_lag1'] = val_df.groupby(['Currency','Event'])['Previous_lag1'].bfill().ffill()

        # Construct validation samples
        print("Constructing validation samples...")
        all_val_samples = []
        
        for key, group in val_df.groupby(['Currency','Event']):
            group = group.sort_values('DateTime')
            n = len(group)
            if n < 2:
                continue

            # Use pre-determined model type
            model_type = event_model_types.get(key, "xgb")

            actuals = group['Actual_numeric'].values
            previous = group['Previous_numeric'].values
            impacts = group['Impact'].astype(float).values
            hicount = group['HighImpactCount'].astype(float).values
            actual_lag1 = group['Actual_lag1'].values
            prev_lag1 = group['Previous_lag1'].values

            # Use normalization parameters from training
            norm_info = event_norm_params.get(key, {})
            
            # Normalize using training parameters
            if 'actual' in norm_info:
                mn_act, rng_act = norm_info['actual']
                norm_actuals = (actuals - mn_act) / rng_act if rng_act > 0 else np.zeros_like(actuals)
            else:
                norm_actuals, _, _ = normalize_feature(actuals)
                
            if 'previous' in norm_info:
                mn_prev, rng_prev = norm_info.get('previous', (0, 1))
                norm_previous = (previous - mn_prev) / rng_prev if rng_prev > 0 else np.zeros_like(previous)
            else:
                norm_previous, _, _ = normalize_feature(previous)
                
            norm_impacts, _, _ = normalize_feature(impacts)
            norm_hic, _, _ = normalize_feature(hicount)
            
            if 'actual_lag1' in norm_info:
                mn_al1, rng_al1 = norm_info.get('actual_lag1', (0, 1))
                norm_al1 = (actual_lag1 - mn_al1) / rng_al1 if rng_al1 > 0 else np.zeros_like(actual_lag1)
            else:
                norm_al1, _, _ = normalize_feature(actual_lag1)
                
            if 'previous_lag1' in norm_info:
                mn_pl1, rng_pl1 = norm_info.get('previous_lag1', (0, 1))
                norm_pl1 = (prev_lag1 - mn_pl1) / rng_pl1 if rng_pl1 > 0 else np.zeros_like(prev_lag1)
            else:
                norm_pl1, _, _ = normalize_feature(prev_lag1)

            features = np.stack([norm_actuals, norm_previous, norm_impacts, norm_hic, norm_al1, norm_pl1], axis=1)
            targets = norm_actuals

            # Create validation samples based on model type
            if model_type == "xgb":
                # XGBoost validation samples
                for i in range(n):
                    all_val_samples.append((features[i], targets[i], key, "xgb"))
            else:
                # RNN validation sequences
                for i in range(max(0, n - seq_length)):
                    all_val_samples.append((features[i:i+seq_length], targets[i+seq_length], key, "rnn"))

        print(f"Created {len(all_val_samples)} validation samples")

        # Run VECTORIZED validation
        print("Running VECTORIZED validation...")
        metrics_records = run_validation_vectorized(
            all_val_samples, rnn_model, xgb_model, event_norm_params, device
        )

        # Save to database - ALWAYS REPLACE
        if metrics_records:
            val_df = pd.DataFrame(metrics_records)
            save_validation_metrics(val_df)
            print(f"✓ Saved {len(val_df)} validation metrics to database (REPLACED existing data)")
            
            # Log overall metrics
            mlflow.log_metric("val_r2_mean", val_df["R2"].mean())
            mlflow.log_metric("val_mse_mean", val_df["MSE"].mean())
            mlflow.log_metric("val_samples_total", val_df["Samples"].sum())
            
            print(f"  Overall R2: {val_df['R2'].mean():.4f}")
            print(f"  Overall MSE: {val_df['MSE'].mean():.4f}")
            
            # Print some sample metrics
            print(f"  Sample validation metrics:")
            for i, record in enumerate(metrics_records[:3]):
                print(f"    {record['Currency']}, {record['Event']}: R2: {record['R2']:.4f}, MSE: {record['MSE']:.4f}, Samples: {record['Samples']}")
        else:
            print("⚠ No validation metrics to save")

        return {"message": "validation is complete"}