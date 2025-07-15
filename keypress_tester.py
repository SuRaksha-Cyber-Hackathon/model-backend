import pytest
import httpx
import uuid
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8000"  # Change this if hosted elsewhere

# Helper to generate fake keypress events
def generate_keypress_events(n=5):
    now = datetime.utcnow()
    events = []
    for i in range(n):
        events.append({
            "key_code": 65 + i,
            "key_label": chr(65 + i),
            "event_type": "keydown",
            "duration_ms": 120 + i * 10,
            "timestamp": (now + timedelta(milliseconds=100 * i)).isoformat() + "Z"
        })
    return events

@pytest.fixture(scope="module")
def test_client():
    return httpx.Client(base_url=BASE_URL)

def test_health_check(test_client):
    res = test_client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

def test_enrollment_and_verification_flow(test_client):
    user_id = str(uuid.uuid4())
    keypress = generate_keypress_events(6)

    # Enroll 5 times
    for i in range(5):
        response = test_client.post(f"/enroll/{user_id}", json={
            "id": user_id,
            "events": {"keypress_events": keypress}
        })
        assert response.status_code == 200
        data = response.json()
        assert "embedding_shape" in data

    # Verify
    response = test_client.post(f"/verify/{user_id}", json={
        "id": user_id,
        "events": {"keypress_events": keypress}
    })
    assert response.status_code == 200
    data = response.json()
    assert "verified" in data
    assert data["average_similarity"] > 0

    # List users
    response = test_client.get("/users")
    assert response.status_code == 200
    user_list = response.json()["enrolled_users"]
    assert user_id in user_list

    # Reset
    response = test_client.post(f"/reset/{user_id}")
    assert response.status_code == 200
    assert "reset successfully" in response.json()["message"]

    # Delete
    response = test_client.delete(f"/users/{user_id}")
    assert response.status_code == 200
    assert "deleted successfully" in response.json()["message"]

def test_enroll_with_insufficient_data(test_client):
    user_id = str(uuid.uuid4())
    bad_keypress = generate_keypress_events(1)  # Only 1 event

    response = test_client.post(f"/enroll/{user_id}", json={
        "id": user_id,
        "events": {"keypress_events": bad_keypress}
    })
    assert response.status_code == 400
    assert "at least 2 key events" in response.json()["detail"]
