import os
import re
from datetime import datetime, date
import pandas as pd
import numpy as np
import joblib
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from contextlib import nullcontext
import xgboost as xgb
import mlflow
import mlflow.pytorch
import mlflow.xgboost

# Import the database connector
from db_connector import load_events_data, save_train_metrics

# ================== CONFIG ==================
DEFAULT_PARAMS = {
    "SEQ_LENGTH": 5,
    "HIDDEN_SIZE": 64,
    "NUM_LAYERS": 2,
    "DROPOUT": 0.2,
    "BATCH_SIZE": 64,
    "EPOCHS": 1,
    "LR": 0.001,
    "WEIGHT_DECAY": 1e-4,
    "XGB_N_ESTIMATORS": 100,
    "XGB_MAX_DEPTH": 3,
    "XGB_LEARNING_RATE": 0.1
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== HELPERS ==================
def parse_numeric_val(val):
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    val = str(val).strip()
    if val == "":
        return None
    if val.endswith("%"):
        try:
            return float(val.rstrip("%"))
        except:
            return None
    k_match = re.match(r"^([-+]?[0-9]*\.?[0-9]+)\s*([KkMmBbTt])$", val)
    if k_match:
        num = float(k_match.group(1))
        unit = k_match.group(2).upper()
        mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}.get(unit, 1)
        return num * mult
    try:
        return float(val)
    except:
        return None

def parse_datetime_from_db(date_val, time_val):
    """Parse datetime from database format - handles both string and datetime objects."""
    try:
        # If date_val is already a datetime object
        if isinstance(date_val, datetime):
            base_date = date_val
        elif isinstance(date_val, date):
            base_date = datetime.combine(date_val, datetime.min.time())
        else:
            # Try to parse as string
            date_str = str(date_val).strip()
            for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y"):
                try:
                    base_date = datetime.strptime(date_str, fmt)
                    break
                except:
                    continue
            else:
                return pd.NaT
        
        # Handle time component
        if pd.isna(time_val) or time_val is None:
            return base_date
        
        if isinstance(time_val, (datetime, pd.Timestamp)):
            time_component = time_val.time()
        elif isinstance(time_val, str):
            time_str = str(time_val).strip()
            # Handle timedelta strings like "0 days 17:00:00"
            if 'days' in time_str:
                try:
                    # Extract time part from timedelta string
                    time_parts = time_str.split(' ')[-1].split(':')
                    if len(time_parts) >= 2:
                        hours = int(time_parts[0])
                        minutes = int(time_parts[1])
                        seconds = int(time_parts[2]) if len(time_parts) > 2 else 0
                        time_component = datetime.strptime(f"{hours:02d}:{minutes:02d}:{seconds:02d}", "%H:%M:%S").time()
                    else:
                        time_component = datetime.min.time()
                except:
                    time_component = datetime.min.time()
            else:
                # Try to parse as time string
                for fmt in ("%H:%M:%S", "%H:%M"):
                    try:
                        time_component = datetime.strptime(time_str, fmt).time()
                        break
                    except:
                        continue
                else:
                    time_component = datetime.min.time()
        else:
            time_component = datetime.min.time()
        
        return datetime.combine(base_date.date(), time_component)
        
    except Exception as e:
        print(f"  ⚠ Datetime parsing error: {e} for date={date_val}, time={time_val}")
        return pd.NaT

def normalize_feature(arr):
    arr = np.asarray(arr, dtype=float)
    if np.all(np.isnan(arr)):
        return np.zeros_like(arr), 0.0, 1.0
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    rng = mx - mn if mx != mn else 1.0
    return (arr - mn) / rng, mn, rng

def split_data_by_time(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split data chronologically by time to ensure no data leakage
    Returns: train_df, val_df, test_df
    """
    # Sort by datetime to maintain temporal order
    df = df.sort_values('DateTime')
    
    # Calculate split indices
    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split chronologically
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()
    
    print(f"  Data split: Train={len(train_df)} ({len(train_df)/n_total*100:.1f}%), "
          f"Val={len(val_df)} ({len(val_df)/n_total*100:.1f}%), "
          f"Test={len(test_df)} ({len(test_df)/n_total*100:.1f}%)")
    
    return train_df, val_df, test_df

# ================== MODEL ==================
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

# ================== VECTORIZED METRICS CALCULATION ==================
def calculate_metrics_vectorized(all_train_samples, rnn_model, xgb_model, event_norm_params, device):
    """Vectorized metrics calculation for significant performance improvement"""
    
    # Separate samples by model type
    rnn_samples = [s for s in all_train_samples if s[3] == "rnn"]
    xgb_samples = [s for s in all_train_samples if s[3] == "xgb"]
    
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

# ================== TRAINING ==================
def run_training(output_folder, params=None):
    print("\n" + "="*80)
    print("STARTING TRAINING PIPELINE")
    print("="*80 + "\n")
    
    # Use provided parameters or defaults (no MLflow fetching)
    if params is None:
        params = DEFAULT_PARAMS.copy()
    else:
        final_params = DEFAULT_PARAMS.copy()
        final_params.update(params)
        params = final_params

    seq_length = params["SEQ_LENGTH"]
    hidden_size = params["HIDDEN_SIZE"]
    num_layers = params["NUM_LAYERS"]
    dropout = params["DROPOUT"]
    batch_size = params["BATCH_SIZE"]
    epochs = params["EPOCHS"]
    lr = params["LR"]
    weight_decay = params["WEIGHT_DECAY"]

    xgb_params = {
        "n_estimators": params["XGB_N_ESTIMATORS"],
        "max_depth": params["XGB_MAX_DEPTH"],
        "learning_rate": params["XGB_LEARNING_RATE"]
    }

    # Start MLflow run and log parameters
    with mlflow.start_run():
        mlflow.log_params(params)

        # Load data from database
        print("\n[STEP 1] Loading data from database...")
        try:
            df = load_events_data()
            print(f"✓ Successfully loaded {len(df)} rows from database")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
            
        except Exception as e:
            print(f"✗ ERROR loading data from database: {e}")
            raise
        
        # Parse datetime from database format
        print("\n[STEP 2] Parsing datetime...")
        df['DateTime'] = df.apply(lambda r: parse_datetime_from_db(r['Date'], r['Time']), axis=1)
        
        valid_datetime_count = df['DateTime'].notna().sum()
        print(f"  DateTime parsed successfully")
        print(f"  Valid datetime rows: {valid_datetime_count}/{len(df)}")
        
        df = df.dropna(subset=['DateTime'])
        print(f"  Rows after dropping invalid datetime: {len(df)}")
        
        if len(df) == 0:
            print("✗ ERROR: No valid data remaining after datetime parsing!")
            return {"error": "No valid datetime data"}
        
        # Data preprocessing
        print("\n[STEP 3] Preprocessing data...")
        impact_map = {'low': 1, 'medium': 2, 'high': 3}
        df['Impact'] = df['Impact'].map(impact_map).fillna(0).astype(int)
        df['Currency'] = df['Currency'].fillna('Unknown').astype(str).str.strip()
        df['Event'] = df['Event'].fillna('Unknown').astype(str).str.strip()
        df['Actual_numeric'] = df['Actual'].apply(parse_numeric_val)
        df['Previous_numeric'] = df['Previous'].apply(parse_numeric_val)
        df['Forecast_numeric'] = df['Forecast'].apply(parse_numeric_val)

        print(f"  Unique currencies: {df['Currency'].nunique()}")
        print(f"  Unique events: {df['Event'].nunique()}")
        print(f"  Impact distribution:\n{df['Impact'].value_counts().to_dict()}")

        # Fill missing values
        print("\n[STEP 4] Filling missing values...")
        event_avgs = df.groupby(['Currency', 'Event'])[['Actual_numeric', 'Previous_numeric']].mean().to_dict('index')

        def fill_missing(row):
            key = (row['Currency'], row['Event'])
            if pd.isna(row['Actual_numeric']):
                row['Actual_numeric'] = event_avgs.get(key, {}).get('Actual_numeric', 0.0)
            if pd.isna(row['Previous_numeric']):
                row['Previous_numeric'] = event_avgs.get(key, {}).get('Previous_numeric', 0.0)
            return row

        df = df.apply(fill_missing, axis=1)

        # Drop events with no data
        print("\n[STEP 5] Filtering valid events...")
        grouped = df.groupby(['Currency', 'Event'])
        valid_keys = []
        for key, sub in grouped:
            if sub['Actual_numeric'].isna().all() and sub['Previous_numeric'].isna().all() and sub['Forecast_numeric'].isna().all():
                continue
            valid_keys.append(key)
        
        print(f"  Valid events: {len(valid_keys)} out of {len(grouped)}")
        df = df[df.set_index(['Currency', 'Event']).index.isin(valid_keys)]
        print(f"  Rows after filtering: {len(df)}")

        # ================== KEY CHANGE: Determine model types based on TOTAL sample size ==================
        print("\n[STEP 6] Determining model types based on total sample size per event...")
        event_model_types = {}
        event_total_samples = {}
        
        for key, group in df.groupby(['Currency', 'Event']):
            total_samples = len(group)
            event_total_samples[key] = total_samples
            # Use same threshold: RNN for >=50 samples, XGBoost for <50
            model_type = "rnn" if total_samples >= 50 else "xgb"
            event_model_types[key] = model_type
            
        rnn_events_count = sum(1 for mt in event_model_types.values() if mt == "rnn")
        xgb_events_count = sum(1 for mt in event_model_types.values() if mt == "xgb")
        
        print(f"  Model assignment based on total samples:")
        print(f"  - RNN events (≥50 samples): {rnn_events_count}")
        print(f"  - XGBoost events (<50 samples): {xgb_events_count}")
        
        # Show some examples
        print(f"  Sample model assignments:")
        for i, (key, model_type) in enumerate(list(event_model_types.items())[:5]):
            print(f"    {key[0]}, {key[1]}: {event_total_samples[key]} samples -> {model_type.upper()}")

        # Split data into train/val/test (70/15/15)
        print("\n[STEP 7] Splitting data into train/val/test (70/15/15)...")
        train_df, val_df, test_df = split_data_by_time(df)
        
        # Save the splits AND model type assignments for validation and test scripts
        os.makedirs(output_folder, exist_ok=True)
        joblib.dump({
            'train_df': train_df,
            'val_df': val_df, 
            'test_df': test_df,
            'event_model_types': event_model_types,  # Save model type assignments
            'event_total_samples': event_total_samples,  # Save total samples for reference
            'training_params': params  # Save training parameters for validation/test
        }, os.path.join(output_folder, "data_splits.joblib"))
        
        print("  ✓ Data splits and model assignments saved")

        # Features for training data only
        print("\n[STEP 8] Engineering features for training data...")
        def create_features(dataframe):
            df_temp = dataframe.copy()
            df_temp['EventMonth'] = df_temp['DateTime'].dt.to_period('M')
            df_temp['EventDate'] = df_temp['DateTime'].dt.date
            high_impact_counts = (df_temp[df_temp['Impact'] == 3].groupby(['Currency', 'EventDate']).size()
                                  .rename('HighImpactCount').reset_index())
            df_temp = pd.merge(df_temp, high_impact_counts, how='left', on=['Currency', 'EventDate'])
            df_temp['HighImpactCount'] = df_temp['HighImpactCount'].fillna(0)
            df_temp = df_temp.sort_values(['Currency', 'Event', 'DateTime'])
            df_temp['Actual_lag1'] = df_temp.groupby(['Currency', 'Event'])['Actual_numeric'].shift(1)
            df_temp['Previous_lag1'] = df_temp.groupby(['Currency', 'Event'])['Previous_numeric'].shift(1)
            
            # Forward fill and backward fill for lag features
            df_temp['Actual_lag1'] = df_temp.groupby(['Currency', 'Event'])['Actual_lag1'].ffill().bfill()
            df_temp['Previous_lag1'] = df_temp.groupby(['Currency', 'Event'])['Previous_lag1'].ffill().bfill()
            
            return df_temp

        train_df = create_features(train_df)
        print("  ✓ Training features engineered successfully")

        # Prepare training data only
        print("\n[STEP 9] Preparing training data...")
        all_train_samples = []
        event_norm_params = {}

        rnn_events = 0
        xgb_events = 0
        skipped_events = 0

        for key, group in train_df.groupby(['Currency', 'Event']):
            group = group.sort_values('DateTime')
            n = len(group)
            if n < 2:
                skipped_events += 1
                continue
            
            # ================== KEY CHANGE: Use pre-determined model type ==================
            model_type = event_model_types.get(key, "xgb")  # Default to xgb if not found
            
            if model_type == "rnn":
                rnn_events += 1
            else:
                xgb_events += 1

            actuals = group['Actual_numeric'].values
            previous = group['Previous_numeric'].values
            impacts = group['Impact'].astype(float).values
            hicount = group['HighImpactCount'].astype(float).values
            actual_lag1 = group['Actual_lag1'].values
            prev_lag1 = group['Previous_lag1'].values

            norm_actuals, mn_act, rng_act = normalize_feature(actuals)
            norm_previous, mn_prev, rng_prev = normalize_feature(previous)
            norm_impacts, mn_imp, rng_imp = normalize_feature(impacts)
            norm_hic, mn_hic, rng_hic = normalize_feature(hicount)
            norm_al1, mn_al1, rng_al1 = normalize_feature(actual_lag1)
            norm_pl1, mn_pl1, rng_pl1 = normalize_feature(prev_lag1)

            event_norm_params[key] = {
                'actual': (mn_act, rng_act),
                'model_type': model_type  # Store model type for consistency
            }

            features = np.stack([norm_actuals, norm_previous, norm_impacts,
                                 norm_hic, norm_al1, norm_pl1], axis=1)
            targets = norm_actuals

            # Create training samples based on model type
            if model_type == "rnn":
                # Create sequences for RNN training
                for i in range(max(0, n - seq_length)):
                    if i + seq_length < n:
                        all_train_samples.append((features[i:i + seq_length], targets[i + seq_length], key, model_type))
            else:
                # Use all samples for XGBoost training
                for i in range(n):
                    all_train_samples.append((features[i], targets[i], key, model_type))

        print(f"  RNN events: {rnn_events}")
        print(f"  XGBoost events: {xgb_events}")
        print(f"  Skipped events (< 2 samples): {skipped_events}")
        print(f"  Total training samples: {len(all_train_samples)}")

        # Train RNN
        train_rnn = [s for s in all_train_samples if s[3] == "rnn"]
        
        print(f"\n[STEP 10] Training RNN model...")
        print(f"  RNN training samples: {len(train_rnn)}")
        
        rnn_model = None
        if train_rnn:
            rnn_model = RNNModel(input_size=6, hidden_size=hidden_size,
                                 num_layers=num_layers, dropout=dropout).to(device)
            optimizer = torch.optim.Adam(rnn_model.parameters(), lr=lr, weight_decay=weight_decay)
            amp_ctx = autocast if device.type == "cuda" else nullcontext
            scaler = GradScaler(enabled=(device.type == "cuda"))
            loss_fn = nn.MSELoss()

            rnn_model.train()
            for epoch in range(epochs):
                total_loss = 0.0
                for i in range(0, len(train_rnn), batch_size):
                    batch = train_rnn[i:i + batch_size]
                    xs = np.array([b[0] for b in batch], dtype=float)
                    ys = np.array([b[1] for b in batch], dtype=float)
                    xb = torch.tensor(xs, dtype=torch.float32).to(device)
                    yb = torch.tensor(ys, dtype=torch.float32).to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with amp_ctx():
                        pred = rnn_model(xb)
                        loss = loss_fn(pred, yb)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    total_loss += loss.item()
                epoch_loss = total_loss / max(1, len(train_rnn) // batch_size)
                mlflow.log_metric("rnn_epoch_loss", epoch_loss, step=epoch)
                print(f"  Epoch {epoch+1}/{epochs} loss: {epoch_loss:.6f}")
        else:
            print("  No RNN samples to train.")

        # Train XGBoost
        xgb_train = [s for s in all_train_samples if s[3] == "xgb"]
        
        print(f"\n[STEP 11] Training XGBoost model...")
        print(f"  XGBoost training samples: {len(xgb_train)}")
        
        xgb_model = None
        if xgb_train:
            X_train = np.array([s[0] for s in xgb_train])
            y_train = np.array([s[1] for s in xgb_train])
            xgb_model = xgb.XGBRegressor(**xgb_params)
            xgb_model.fit(X_train, y_train)
            mlflow.xgboost.log_model(xgb_model, artifact_path="xgb_model",
                                     registered_model_name="MyXGBModel")
            print("  XGBoost training completed")
        else:
            print("  No XGBoost samples to train.")

        # Save models
        print(f"\n[STEP 12] Saving models to {output_folder}...")
        if rnn_model:
            torch.save(rnn_model.state_dict(), os.path.join(output_folder, "rnn_model.pt"))
            mlflow.pytorch.log_model(rnn_model, artifact_path="rnn_model", registered_model_name="MyRNNModel")
            print("  ✓ RNN model saved")
        if xgb_model:
            joblib.dump(xgb_model, os.path.join(output_folder, "xgb_model.joblib"))
            print("  ✓ XGBoost model saved")
        
        joblib.dump(event_norm_params, os.path.join(output_folder, "event_norm_params.joblib"))
        mlflow.log_artifact(os.path.join(output_folder, "event_norm_params.joblib"))
        print("  ✓ Normalization parameters saved")

        # Calculate training metrics using VECTORIZED approach
        print(f"\n[STEP 13] Calculating and saving training metrics (VECTORIZED)...")
        
        metrics_records = calculate_metrics_vectorized(
            all_train_samples, rnn_model, xgb_model, event_norm_params, device
        )

        # Save metrics to database - ALWAYS REPLACE
        if metrics_records:
            metrics_df = pd.DataFrame(metrics_records)
            try:
                save_train_metrics(metrics_df)
                print(f"  ✓ Saved {len(metrics_records)} training metrics to database (REPLACED existing data)")
                
                # Print some sample metrics
                print(f"  Sample metrics:")
                for i, record in enumerate(metrics_records[:3]):
                    print(f"    {record['Currency']}, {record['Event']}: R2: {record['R2']:.4f}, MSE: {record['MSE']:.4f}, Samples: {record['Samples']}")
                    
            except Exception as e:
                print(f"  ✗ ERROR saving training metrics to database: {e}")
        else:
            print("  ⚠ No training metrics to save")

        print("\n" + "="*80)
        print("TRAINING PIPELINE COMPLETED")
        print("="*80 + "\n")
        
        return {"message": "Training completed successfully"}