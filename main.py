import os
import json
from datetime import datetime
from fastapi import FastAPI, Request
from apscheduler.schedulers.background import BackgroundScheduler
import uvicorn

# Import the trainer function
def train_isolation_forest():
    from model_trainer import train_isolation_forest as _train
    _train()

# Paths
DATA_FILE = "data_store/capture_data.jsonl"

# Ensure data directory exists
os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)

app = FastAPI(title="Behavior Capture API")

# Scheduler setup
scheduler = BackgroundScheduler()
scheduler.add_job(train_isolation_forest, 'interval', minutes=10)

@app.on_event("startup")
def startup_event():
    scheduler.start()
    # Optional: run immediately on startup
    # train_isolation_forest()

@app.on_event("shutdown")
def shutdown_event():
    scheduler.shutdown()

@app.post("/receive")
async def receive_behavior_data(request: Request):
    data = await request.json()

    print('Data recieved : ', data  )
    data['received_at'] = datetime.utcnow().isoformat()
    # Save to JSONL
    with open(DATA_FILE, 'a') as f:
        f.write(json.dumps(data) + "\n")
    return {"status": "received", "total_events": len(data.get('keypress_events', [])) + len(data.get('swipe_events', []))}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
