import logging
import torch
from fastapi import FastAPI, HTTPException
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
import os 
from fastapi import Request
import numpy as np

from keypress_utils import extract_sensor_feature_windows
import keypress_model
from keypress_model import load_model, SensorNetwork
from keypress_utils import (
    BBAData,
    KEYPRESS_REQUIRED_SAMPLES,
    preprocess_keypress_events,
    normalize_keypress_features,
    save_embedding_to_disk_keypress,
    load_user_embeddings_keypress,
    prune_old_embeddings_keypress
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Global Server", version="1.0.0")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    
    allow_methods=["*"],
    allow_headers=["*"],
)

SENSOR_MODEL_INPUT_SIZE = 63  # if window=10: 10*6 + 3
sensor_model = SensorNetwork(input_size=SENSOR_MODEL_INPUT_SIZE)
sensor_model.load_state_dict(torch.load("sensor_model.pt", map_location=device))
sensor_model.eval()

KEYPRESS_EMBEDDING_DIR = "KEY_EMBEDDINGS"
os.makedirs(KEYPRESS_EMBEDDING_DIR, exist_ok=True)

SENSOR_EMBEDDING_DIR = "sensor_embeddings"
os.makedirs(SENSOR_EMBEDDING_DIR, exist_ok=True)

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

 
# ------------- - --------------------- ----------------

@app.on_event("startup")
async def on_startup():
    load_model()

@app.get("/")
async def root():
    return {"message": "Server running"}

@app.get("/health")
async def health_check_keypress():
    return {
        "status": "healthy",
        "model_loaded": keypress_model is not None,
        "device": str(device),
    }

@app.post("/enroll/{user_id}")
async def enroll_user_keypress(user_id: str, typing_data: BBAData):
    evs = typing_data.events.keypress_events
    if len(evs) < 2:
        raise HTTPException(
            status_code=400,
            detail={"status": "bad_request", "message": "Need at least 2 key events for enrollment"}
        )
    try:
        feats = preprocess_keypress_events(evs)
        norm = normalize_keypress_features(feats)

        with torch.no_grad():
            emb = (
                keypress_model.model.forward_once(torch.tensor(norm).unsqueeze(0).to(device))
                .cpu()
                .numpy()
                .flatten()
            )

        save_embedding_to_disk_keypress(user_id, emb)
        prune_old_embeddings_keypress(user_id)

        count = len(load_user_embeddings_keypress(user_id))
        complete = count >= KEYPRESS_REQUIRED_SAMPLES

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "phase": "enroll",
                "sample_count": count,
                "required_samples": KEYPRESS_REQUIRED_SAMPLES,
                "enrollment_complete": complete,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ENROLL] {e}")
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": str(e)}
        )

@app.post("/verify/{user_id}")
async def verify_user_keypress(user_id: str, typing_data: BBAData):
    stored = []
    try:
        stored = load_user_embeddings_keypress(user_id)
    except Exception:
        pass

    if not stored:
        raise HTTPException(
            status_code=404,
            detail={"status": "user_not_found", "message": "No embeddings found, please enroll first"}
        )

    if len(stored) < KEYPRESS_REQUIRED_SAMPLES:
        raise HTTPException(
            status_code=409,
            detail={
                "status": "enrollment_incomplete",
                "message": f"Enrollment incomplete ({len(stored)}/{KEYPRESS_REQUIRED_SAMPLES})",
                "current_samples": len(stored),
                "required_samples": KEYPRESS_REQUIRED_SAMPLES,
            }
        )

    evs = typing_data.events.keypress_events
    if len(evs) < 2:
        raise HTTPException(
            status_code=400,
            detail={"status": "bad_request", "message": "Need at least 2 key events for verification"}
        )

    try:
        feats = preprocess_keypress_events(evs)
        norm = normalize_keypress_features(feats)

        with torch.no_grad():
            curr = (
                keypress_model.model.forward_once(torch.tensor(norm).unsqueeze(0).to(device))
                .cpu()
                .numpy()
                .flatten()
            )

        distances = [float(np.linalg.norm(stored_emb - curr)) for stored_emb in stored]
        avg_dist = float(np.mean(distances))
        max_dist = float(np.max(distances))

        distance_threshold = 1.5
        verified = avg_dist < distance_threshold

        if verified:
            save_embedding_to_disk_keypress(user_id, curr)
            prune_old_embeddings_keypress(user_id)

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "phase": "verify",
                "verified": verified,
                "average_similarity": avg_dist,
                "max_similarity": max_dist,
                "threshold": distance_threshold,
                "stored_embeddings_count": len(stored),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[VERIFY] {e}")
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": str(e)}
        )