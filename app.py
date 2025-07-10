from fastapi import FastAPI, Request
import os
import numpy as np
import torch
from model import SiameseNetwork
from utils import extract_feature_windows
from fastapi.responses import JSONResponse
from parser import parse_request_data

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_SIZE = 63  # if window=10: 10*6 (sensor) + 3 (tap, scroll, keypress)
model = SiameseNetwork(input_size=INPUT_SIZE)
model.load_state_dict(torch.load("bba_siamese_model.pt", map_location=device))
model.eval()

EMBEDDING_DIR = "embeddings"
os.makedirs(EMBEDDING_DIR, exist_ok=True)

def embed(features):
    """Encodes a single feature window into an embedding using the Siamese model."""
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32).to(device)
        emb = model.forward_once(x).cpu().numpy()
        return emb

@app.get("/check_user/{user_id}")
def check_user(user_id: str):
    """
    Endpoint: /check_user/{user_id}
    Purpose: Check if a user is already enrolled.
    Returns:
        - {"exists": True} if the embedding file for the user exists
        - {"exists": False} otherwise
    Used during app startup to determine whether to enroll or authenticate the user.
    """
    path = os.path.join(EMBEDDING_DIR, f"{user_id}.npy")
    exists = os.path.exists(path)
    return JSONResponse(content={"exists": exists})

@app.post("/receive")
async def receive_data(req: Request):
    """
    Endpoint: /check_user/{user_id}
    Purpose: Check if a user is already enrolled.
    Returns:
        - {"exists": True} if the embedding file for the user exists
        - {"exists": False} otherwise
    Used during app startup to determine whether to enroll or authenticate the user.
    """
    raw_data = await req.json()
    user_id = raw_data["id"]

    parsed_data = parse_request_data(raw_data) 
    windows = extract_feature_windows(parsed_data)

    if not windows:
        return {"status": "error", "msg": "No usable windows"}

    embs = np.stack([embed(w) for w in windows])
    np.save(f"{EMBEDDING_DIR}/{user_id}.npy", embs)

    return {
        "status": "stored",
        "user_id": user_id,
        "windows": len(embs)
    }

@app.post("/authenticate")
async def authenticate(req: Request):
    """
    Endpoint: /authenticate
    Purpose: Authenticate an enrolled user using new behavior data.
    Workflow:
        - Extracts new embeddings from the request
        - Compares with stored user embeddings via Euclidean distance
        - If match score < threshold:
            - Authenticated 
            - User embedding history is updated to adapt to behavior drift
        - Else:
            - Marked as anomaly 
    Returns:
        - "auth" boolean
        - "score" and "threshold" for interpretability
        - "status": either "ok" or "anomaly"
    """

    raw_data = await req.json()
    user_id = raw_data["id"]
    user_path = os.path.join(EMBEDDING_DIR, f"{user_id}.npy")

    if not os.path.exists(user_path):
        return {"auth": False, "msg": "No reference for user"}

    parsed_data = parse_request_data(raw_data)  
    windows = extract_feature_windows(parsed_data)
    if not windows:
        return {"auth": False, "msg": "No usable window"}

    ref = np.load(user_path)
    current_embs = np.stack([embed(w) for w in windows])

    dists = np.linalg.norm(ref[:, None, :] - current_embs[None, :, :], axis=2)
    score = float(np.min(dists))
    threshold = 1.0

    #####################
    # To be implemented: 
    # - dynamic thresholding based on user history
    #   - instead of directly setting a static threshold, use 
    #     statistics from the user's embedding history to set a dynamic threshold
    #####################

    is_auth = score < threshold

    print(f"[AUTH] User: {user_id} | Score: {score:.4f} | Auth: {is_auth}")

    if is_auth:
        updated_embs = np.vstack([ref, current_embs])
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

