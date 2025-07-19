import os
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Union, Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)

KEYPRESS_EMBEDDING_DIR = "KEY_EMBEDDINGS"
os.makedirs(KEYPRESS_EMBEDDING_DIR, exist_ok=True)

KEYPRESS_REQUIRED_SAMPLES = 10
KEYPRESS_MAX_EMBEDDINGS_STORED = 20

# Pydantic Models
class KeyEvent(BaseModel):
    id: Optional[str] = None
    key_code: int
    key_label: Union[str, int]
    event_type: str
    duration_ms: int
    timestamp: str
    digram_key1: Optional[str] = None
    digram_key2: Optional[str] = None
    context_screen: Optional[str] = None
    field_name: Optional[str] = None

class Events(BaseModel):
    keypress_events: List[KeyEvent]
    swipe_events: List[Dict[str, Any]] = []
    tap_events: List[Dict[str, Any]] = []
    sensor_events: List[Dict[str, Any]] = []
    scroll_events: List[Dict[str, Any]] = []

class BBAData(BaseModel):
    id: str
    events: Events

# Persistence Functions
def save_embedding_to_disk_keypress(user_id: str, embedding: np.ndarray):
    files = [f for f in os.listdir(KEYPRESS_EMBEDDING_DIR) if f.startswith(user_id)]
    idx = len(files)
    path = os.path.join(KEYPRESS_EMBEDDING_DIR, f"{user_id}_{idx}.npy")
    np.save(path, embedding)

def load_user_embeddings_keypress(user_id: str) -> List[np.ndarray]:
    files = sorted(
        [f for f in os.listdir(KEYPRESS_EMBEDDING_DIR) if f.startswith(user_id)],
        key=lambda f: int(f.split("_")[-1].split('.')[0]),
    )
    return [np.load(os.path.join(KEYPRESS_EMBEDDING_DIR, f)) for f in files]

def prune_old_embeddings_keypress(user_id: str, max_embeddings=KEYPRESS_MAX_EMBEDDINGS_STORED):
    files = sorted(
        [f for f in os.listdir(KEYPRESS_EMBEDDING_DIR) if f.startswith(user_id)],
        key=lambda f: int(f.split("_")[-1].split('.')[0]),
    )
    for f in files[:-max_embeddings]:
        os.remove(os.path.join(KEYPRESS_EMBEDDING_DIR, f))

def clear_user_embeddings_keypress(user_id: str):
    for f in os.listdir(KEYPRESS_EMBEDDING_DIR):
        if f.startswith(user_id):
            os.remove(os.path.join(KEYPRESS_EMBEDDING_DIR, f))

# Feature Processing
def preprocess_keypress_events(events: List[KeyEvent]) -> np.ndarray:
    if len(events) < 2:
        raise ValueError("Need at least 2 key events")
    evs = sorted(
        events,
        key=lambda x: datetime.fromisoformat(x.timestamp.replace('Z', '+00:00'))
    )
    mapping = {k: i for i, k in enumerate({e.key_label for e in evs})}
    feats = []
    for i, e in enumerate(evs):
        code = mapping[e.key_label]
        dur = e.duration_ms
        if i == 0:
            ikt = 0.0
        else:
            prev = datetime.fromisoformat(evs[i - 1].timestamp.replace('Z', '+00:00'))
            curr = datetime.fromisoformat(e.timestamp.replace('Z', '+00:00'))
            ikt = (curr - prev).total_seconds() * 1000
        feats.append([code, dur, ikt])
    return np.array(feats, dtype=np.float32)

def normalize_keypress_features(feats: np.ndarray) -> np.ndarray:
    out = feats.copy()
    for col in range(out.shape[1]):
        mn, mx = out[:, col].min(), out[:, col].max()
        if mx > mn:
            out[:, col] = (out[:, col] - mn) / (mx - mn)
    return out

# ------------- ---------------------- ---------------------------

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
            if abs((ts_a - ts_g).total_seconds()) > 0.5:
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

