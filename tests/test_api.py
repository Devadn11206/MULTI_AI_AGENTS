import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_providers_status():
    response = client.get("/health/providers")
    assert response.status_code == 200
    data = response.json()
    assert "gemini" in data
    assert "groq" in data
    assert data["primary"] == "gemini"

def test_invalid_workflow_creation_missing_dataset():
    response = client.post("/workflows", json={
        "objective": "Predict churn",
        "priority": "recall"
        # missing dataset_id
    })
    assert response.status_code == 422

def test_workflow_status_not_found():
    response = client.get("/workflows/non_existent_id")
    assert response.status_code == 404

def test_upload_invalid_file_type():
    with open("requirements.txt", "rb") as f:
        response = client.post("/datasets/upload", files={"file": ("requirements.txt", f, "text/plain")})
    assert response.status_code == 422
    assert "Only .csv and .xlsx" in response.text
