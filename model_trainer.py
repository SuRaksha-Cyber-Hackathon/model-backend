import os
import json
from datetime import datetime
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

# Paths
data_file = "data_store/capture_data.jsonl"
model_dir = "models"
model_path = os.path.join(model_dir, "isolation_forest.joblib")

# Feature extraction helpers
def flatten_keypress(events):
    """
    Compute inter-key latencies and summary statistics.
    """
    timestamps = []
    for e in events:
        try:
            t = datetime.fromisoformat(e.get('timestamp')).timestamp()
            timestamps.append(t)
        except:
            continue
    if len(timestamps) < 2:
        # Not enough data
        return {
            'key_count': len(timestamps),
            'mean_latency': 0.0,
            'std_latency': 0.0
        }
    # compute latencies
    latencies = [t2 - t1 for t1, t2 in zip(timestamps, timestamps[1:])]
    return {
        'key_count': len(timestamps),
        'mean_latency': pd.Series(latencies).mean(),
        'std_latency': pd.Series(latencies).std()
    }

def flatten_swipe(events):
    """
    Compute swipe count, duration, and average speed.
    """
    if not events:
        return {
            'swipe_count': 0,
            'duration': 0.0,
            'avg_interval': 0.0
        }
    times = []
    for e in events:
        try:
            times.append(datetime.fromisoformat(e.get('timestamp')).timestamp())
        except:
            continue
    if len(times) < 2:
        return {
            'swipe_count': len(times),
            'duration': 0.0,
            'avg_interval': 0.0
        }
    duration = max(times) - min(times)
    intervals = [t2 - t1 for t1, t2 in zip(times, times[1:])]
    return {
        'swipe_count': len(times),
        'duration': duration,
        'avg_interval': pd.Series(intervals).mean()
    }

# Load and prepare DataFrame
def load_data():
    records = []
    if not os.path.exists(data_file):
        return pd.DataFrame()
    with open(data_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            kp_feats = flatten_keypress(entry.get('keypress_events', []))
            sw_feats = flatten_swipe(entry.get('swipe_events', []))
            record = {**kp_feats, **sw_feats}
            records.append(record)
    return pd.DataFrame(records)

# Train Isolation Forest model
def train_isolation_forest():
    os.makedirs(model_dir, exist_ok=True)
    df = load_data()
    if df.empty:
        print("No data available for training.")
        return

    # Fill any NaNs
    df.fillna(0, inplace=True)

    # Features for model
    feature_cols = df.columns.tolist()
    X = df[feature_cols]

    # Initialize & fit
    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )
    model.fit(X)

    # Save model and feature list
    joblib.dump({'model': model, 'features': feature_cols}, model_path)
    print(f"Model trained on {len(df)} samples; saved to {model_path}")


