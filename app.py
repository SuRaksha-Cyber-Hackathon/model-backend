from fastapi import FastAPI, Request
import os
import numpy as np
import torch
from model import SiameseNetwork
from utils import extract_feature_windows
from fastapi.responses import JSONResponse

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_SIZE = 63  # if window=10: 10*6 + 3
model = SiameseNetwork(input_size=INPUT_SIZE)
model.load_state_dict(torch.load("sensor_model.pt", map_location=device))
model.eval()

EMBEDDING_DIR = "embeddings"
os.makedirs(EMBEDDING_DIR, exist_ok=True)

def embed(features):
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32).to(device)
        emb = model.forward_once(x).cpu().numpy()
        return emb

@app.get("/check_user/{user_id}")
def check_user(user_id: str):
    """
    Check if the user has already enrolled (i.e., embedding file exists).
    """
    path = os.path.join(EMBEDDING_DIR, f"{user_id}.npy")
    exists = os.path.exists(path)
    return JSONResponse(content={"exists": exists})

@app.post("/receive")
async def receive_data(req: Request):
    data = await req.json()
    user_id = data["id"]
    windows = extract_feature_windows(data)

    if not windows:
        return {"status": "error", "msg": "No usable windows"}

    # Store new user template
    embs = np.stack([embed(w) for w in windows])
    np.save(f"{EMBEDDING_DIR}/{user_id}.npy", embs)
    return {"status": "stored", "user_id": user_id, "windows": len(embs)}

@app.post("/authenticate")
async def authenticate(req: Request):
    data = await req.json()
    user_id = data["id"]
    user_path = os.path.join(EMBEDDING_DIR, f"{user_id}.npy")

    if not os.path.exists(user_path):
        return {"auth": False, "msg": "No reference for user"}

    ref = np.load(user_path)  # shape: [n_ref, 128]
    windows = extract_feature_windows(data)
    if not windows:
        return {"auth": False, "msg": "No usable window"}

    current_embs = np.stack([embed(w) for w in windows])  # shape: [n_current, 128]

    # Compute distance matrix
    dists = np.linalg.norm(ref[:, None, :] - current_embs[None, :, :], axis=2)  # shape: [n_ref, n_current]
    score = float(np.min(dists))  # best match distance

    threshold = 1.0  # tune as needed
    is_auth = score < threshold

    print(f"[AUTH] User: {user_id} | Score: {score:.4f} | Auth: {is_auth}")

    if is_auth:
        # ✅ Auth passed: update embedding
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

