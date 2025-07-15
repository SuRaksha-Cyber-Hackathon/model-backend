import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.spatial.distance import cosine
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from torch import nn

# ─── Configuration ─────────────────────────────────────────────────────────────
EMBEDDING_DIR = "KEY_EMBEDDINGS"
os.makedirs(EMBEDDING_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REQUIRED_SAMPLES = 10       # number of samples needed for completion
MAX_EMBEDDINGS_STORED = 20  # max stored per user after pruning

# ─── Persistence Helpers ───────────────────────────────────────────────────────
def save_embedding_to_disk(user_id: str, embedding: np.ndarray):
    files = [f for f in os.listdir(EMBEDDING_DIR) if f.startswith(user_id)]
    idx = len(files)
    path = os.path.join(EMBEDDING_DIR, f"{user_id}_{idx}.npy")
    np.save(path, embedding)

def load_user_embeddings(user_id: str) -> List[np.ndarray]:
    files = sorted(
        [f for f in os.listdir(EMBEDDING_DIR) if f.startswith(user_id)],
        key=lambda f: int(f.split("_")[-1].split('.')[0]),
    )
    return [np.load(os.path.join(EMBEDDING_DIR, f)) for f in files]

def prune_old_embeddings(user_id: str, max_embeddings=MAX_EMBEDDINGS_STORED):
    files = sorted(
        [f for f in os.listdir(EMBEDDING_DIR) if f.startswith(user_id)],
        key=lambda f: int(f.split("_")[-1].split('.')[0]),
    )
    for f in files[:-max_embeddings]:
        os.remove(os.path.join(EMBEDDING_DIR, f))

def clear_user_embeddings(user_id: str):
    for f in os.listdir(EMBEDDING_DIR):
        if f.startswith(user_id):
            os.remove(os.path.join(EMBEDDING_DIR, f))

# ─── Model Definition ──────────────────────────────────────────────────────────
class SiameseGRU(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward_once(self, x):
        _, h = self.gru(x)
        return h.squeeze(0)

    def forward(self, x1, x2):
        o1, o2 = self.forward_once(x1), self.forward_once(x2)
        diff = torch.abs(o1 - o2)
        return self.fc(diff)

# ─── Pydantic Schemas ─────────────────────────────────────────────────────────
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

# ─── Utilities ─────────────────────────────────────────────────────────────────
def preprocess_keypress_events(events: List[KeyEvent]) -> np.ndarray:
    if len(events) < 2:
        raise ValueError("Need at least 2 key events")
    evs = sorted(
        events,
        key=lambda x: datetime.fromisoformat(x.timestamp.replace('Z', '+00:00'))
    )
    # map key_label to integer codes
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

def normalize_features(feats: np.ndarray) -> np.ndarray:
    out = feats.copy()
    for col in range(out.shape[1]):
        mn, mx = out[:, col].min(), out[:, col].max()
        if mx > mn:
            out[:, col] = (out[:, col] - mn) / (mx - mn)
    return out

# ─── Model Loading ────────────────────────────────────────────────────────────
model: SiameseGRU
device: torch.device

def load_model():
    global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseGRU(input_dim=3, hidden_dim=64).to(device)
    model.load_state_dict(torch.load("siamese_gru_model.pth", map_location=device))
    model.eval()
    logger.info(f"Model loaded on {device}")

# ─── FastAPI Setup ────────────────────────────────────────────────────────────
app = FastAPI(title="BBA Siamese GRU Server", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # update in production to your client’s origins
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def on_startup():
    load_model()

@app.get("/")
async def root():
    return {"message": "Server running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
    }

# ─── ENROLL ────────────────────────────────────────────────────────────────────
@app.post("/enroll/{user_id}")
async def enroll_user(user_id: str, typing_data: BBAData):
    evs = typing_data.events.keypress_events
    if len(evs) < 2:
        raise HTTPException(
            status_code=400,
            detail={"status": "bad_request", "message": "Need at least 2 key events for enrollment"}
        )
    try:
        feats = preprocess_keypress_events(evs)
        norm = normalize_features(feats)

        with torch.no_grad():
            emb = (
                model.forward_once(torch.tensor(norm).unsqueeze(0).to(device))
                .cpu()
                .numpy()
                .flatten()
            )

        save_embedding_to_disk(user_id, emb)
        prune_old_embeddings(user_id)

        count = len(load_user_embeddings(user_id))
        complete = count >= REQUIRED_SAMPLES

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "phase": "enroll",
                "sample_count": count,
                "required_samples": REQUIRED_SAMPLES,
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

# ─── VERIFY ────────────────────────────────────────────────────────────────────
@app.post("/verify/{user_id}")
async def verify_user(user_id: str, typing_data: BBAData):
    stored = []
    try:
        stored = load_user_embeddings(user_id)
    except Exception:
        pass

    if not stored:
        # no embeddings at all
        raise HTTPException(
            status_code=404,
            detail={"status": "user_not_found", "message": "No embeddings found, please enroll first"}
        )

    if len(stored) < REQUIRED_SAMPLES:
        # not enough samples yet
        raise HTTPException(
            status_code=409,
            detail={
                "status": "enrollment_incomplete",
                "message": f"Enrollment incomplete ({len(stored)}/{REQUIRED_SAMPLES})",
                "current_samples": len(stored),
                "required_samples": REQUIRED_SAMPLES,
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
        norm = normalize_features(feats)

        with torch.no_grad():
            curr = (
                model.forward_once(torch.tensor(norm).unsqueeze(0).to(device))
                .cpu()
                .numpy()
                .flatten()
            )

        distances = [float(np.linalg.norm(stored_emb - curr)) for stored_emb in stored]
        avg_dist = float(np.mean(distances))
        max_dist = float(np.max(distances))

        # smaller distance => more similar
        distance_threshold = 1.5  # tune this!
        verified = avg_dist < distance_threshold

        if verified:
            save_embedding_to_disk(user_id, curr)
            prune_old_embeddings(user_id)

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

# ─── USER LISTING & MANAGEMENT ────────────────────────────────────────────────
@app.get("/users")
async def list_users():
    infos: Dict[str, Dict[str, Any]] = {}
    for f in os.listdir(EMBEDDING_DIR):
        if not f.endswith(".npy"):
            continue
        uid = f.split("_")[0]
        if uid in infos:
            continue
        embs = load_user_embeddings(uid)
        infos[uid] = {
            "sample_count": len(embs),
            "enrollment_complete": len(embs) >= REQUIRED_SAMPLES,
        }
    return {"status": "success", "total_users": len(infos), "users": infos}

@app.get("/users/{user_id}")
async def check_user(user_id: str):
    try:
        embs = load_user_embeddings(user_id)
    except Exception:
        embs = []
    return {
        "status": "success",
        "user_id": user_id,
        "sample_count": len(embs),
        "enrollment_complete": len(embs) >= REQUIRED_SAMPLES,
    }

@app.delete("/users/{user_id}")
async def delete_user(user_id: str):
    try:
        clear_user_embeddings(user_id)
        return {"status": "success", "message": f"User {user_id} deleted"}
    except Exception as e:
        logger.error(f"[DELETE] {e}")
        raise HTTPException(
            status_code=500, detail={"status": "error", "message": str(e)}
        )

@app.post("/reset/{user_id}")
async def reset_user(user_id: str):
    try:
        clear_user_embeddings(user_id)
        return {"status": "success", "message": f"User {user_id} enrollment reset"}
    except Exception as e:
        logger.error(f"[RESET] {e}")
        raise HTTPException(
            status_code=500, detail={"status": "error", "message": str(e)}
        )
