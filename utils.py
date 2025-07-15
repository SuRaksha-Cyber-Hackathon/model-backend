from datetime import datetime
import numpy as np

def extract_sensor_feature_windows(user_dict, win_size=10, step_size=5):
    sensor = user_dict["events"]["sensor_events"]
    taps = user_dict["events"]["tap_events"]
    swipes = user_dict["events"]["swipe_events"]
    keys = user_dict["events"]["keypress_events"]

    acc = [e for e in sensor if e["type"] == "accelerometer"]
    gyro = [e for e in sensor if e["type"] == "gyroscope"]

    feats = []
    for a, g in zip(acc, gyro):
        try:
            ts_a = datetime.fromisoformat(a["timestamp"])
            ts_g = datetime.fromisoformat(g["timestamp"])
            if abs((ts_a - ts_g).total_seconds()) > 0.1:
                continue
            feats.append((ts_a, [a["x"], a["y"], a["z"], g["x"], g["y"], g["z"]]))
        except:
            continue

    if not feats:
        return []

    feats.sort()
    windows = []
    times = [ts for ts, _ in feats]
    features = [f for _, f in feats]

    tap_times = [datetime.fromisoformat(e["timestamp"]) for e in taps]
    swipe_times = [datetime.fromisoformat(e["timestamp"]) for e in swipes]
    key_times = [datetime.fromisoformat(e["timestamp"]) for e in keys]

    for i in range(0, len(features) - win_size, step_size):
        x = features[i:i+win_size]
        ts_win = times[i:i+win_size]
        start, end = ts_win[0], ts_win[-1]
        tap_count = sum(start <= t <= end for t in tap_times)
        swipe_count = sum(start <= t <= end for t in swipe_times)
        key_count = sum(start <= t <= end for t in key_times)
        vec = np.array(x).flatten()
        full = np.concatenate([vec, [tap_count, swipe_count, key_count]])
        windows.append(full)

    return windows