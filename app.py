from fastapi import FastAPI, Request
import os
import numpy as np
from fastapi.responses import JSONResponse

from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class SensorNetwork(nn.Module):
    def __init__(self, input_size, embedding_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward_once(self, x):
        return self.fc(x)

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SENSOR_MODEL_INPUT_SIZE = 63  # if window=10: 10*6 + 3
sensor_model = SensorNetwork(input_size=SENSOR_MODEL_INPUT_SIZE)
sensor_model.load_state_dict(torch.load("sensor_model.pt", map_location=device))
sensor_model.eval()

SENSOR_EMBEDDING_DIR = "sensor_embeddings"
os.makedirs(SENSOR_EMBEDDING_DIR, exist_ok=True)

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

def embed_sensor(features):
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32).to(device)
        emb = sensor_model.forward_once(x).cpu().numpy()
        return emb

@app.get("/check_user/{user_id}")
def check_sensor_user(user_id: str):
    """
    Check if the user has already enrolled (i.e., embedding file exists).
    """
    path = os.path.join(SENSOR_EMBEDDING_DIR, f"{user_id}.npy")
    exists = os.path.exists(path)
    return JSONResponse(content={"exists": exists})

@app.post("/receive")
async def receive_sensor_data(req: Request):
    data = await req.json()
    user_id = data["id"]
    windows = extract_sensor_feature_windows(data)

    if not windows:
        return {"status": "error", "msg": "No usable windows"}

    embs = np.stack([embed_sensor(w) for w in windows])
    np.save(f"{SENSOR_EMBEDDING_DIR}/{user_id}.npy", embs)
    return {"status": "stored", "user_id": user_id, "windows": len(embs)}

@app.post("/authenticate")
async def authenticate_sensor_data(req: Request):
    data = await req.json()
    user_id = data["id"]
    user_path = os.path.join(SENSOR_EMBEDDING_DIR, f"{user_id}.npy")

    if not os.path.exists(user_path):
        return {"auth": False, "msg": "No reference for user"}

    ref = np.load(user_path)  # shape: [n_ref, 128]
    windows = extract_sensor_feature_windows(data)
    if not windows:
        return {"auth": False, "msg": "No usable window"}

    current_embs = np.stack([embed_sensor(w) for w in windows])  # shape: [n_current, 128]

    # Compute distance matrix
    dists = np.linalg.norm(ref[:, None, :] - current_embs[None, :, :], axis=2)  # shape: [n_ref, n_current]
    score = float(np.min(dists))  # best match distance

    threshold = 1.0  # tune as needed
    is_auth = score < threshold

    print(f"[AUTH] User: {user_id} | Score: {score:.4f} | Auth: {is_auth}")

    if is_auth:
        updated_embs = np.vstack([ref, current_embs])
        
        # Trim to max 100 recent embeddings
        MAX_HISTORY = 100
        if updated_embs.shape[0] > MAX_HISTORY:
            updated_embs = updated_embs[-MAX_HISTORY:]

        np.save(user_path, updated_embs)

    return {
        "status": "ok" if is_auth else "anomaly",
        "auth": is_auth,
        "score": score,
        "threshold": threshold
    }

