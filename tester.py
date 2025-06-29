import os
import json
import joblib
import pandas as pd
from datetime import datetime

# --- 1. Load model + feature names ---
model_bundle = joblib.load("models/isolation_forest.joblib")
model = model_bundle['model']
features = model_bundle['features']

# --- 2. File path ---
data_file = "data_store/capture_data.jsonl"
if not os.path.exists(data_file):
    raise FileNotFoundError(f"{data_file} not found.")

# --- 3. Feature extraction helpers ---
def flatten_keypress(events):
    ts = []
    for e in events:
        try:
            ts.append(datetime.fromisoformat(e['timestamp']).timestamp())
        except:
            continue
    if len(ts) < 2:
        return {'key_count': len(ts), 'mean_latency': 0.0, 'std_latency': 0.0}
    lats = [t2 - t1 for t1, t2 in zip(ts, ts[1:])]
    return {
        'key_count': len(ts),
        'mean_latency': pd.Series(lats).mean(),
        'std_latency': pd.Series(lats).std()
    }

def flatten_swipe(events):
    ts = []
    for e in events:
        try:
            ts.append(datetime.fromisoformat(e['timestamp']).timestamp())
        except:
            continue
    if len(ts) < 2:
        return {'swipe_count': len(ts), 'duration': 0.0, 'avg_interval': 0.0}
    duration = max(ts) - min(ts)
    intervals = [t2 - t1 for t1, t2 in zip(ts, ts[1:])]
    return {
        'swipe_count': len(ts),
        'duration': duration,
        'avg_interval': pd.Series(intervals).mean()
    }

# --- 4. Read JSONL and build records ---
records = []
with open(data_file, 'r') as f:
    for line in f:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        kp_feats = flatten_keypress(obj.get('keypress_events', []))
        sw_feats = flatten_swipe(obj.get('swipe_events', []))
        records.append({**kp_feats, **sw_feats})

if not records:
    print("No valid records found in JSONL.")
    exit()

# --- 5. Create DataFrame and align features ---
df = pd.DataFrame(records).fillna(0)
X_test = df.reindex(columns=features, fill_value=0)

# --- 6. Predict & score ---
df['anomaly_score'] = model.decision_function(X_test)
df['is_anomaly']   = model.predict(X_test) == -1

# --- 7. Output results ---
print(df)
