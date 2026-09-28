import os
import time
import pandas as pd
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def create_dummy_dataset():
    data = {
        "feature1": [1.0, 2.0, 1.5, 8.0, 9.0],
        "feature2": [1.0, 1.0, 1.0, 9.0, 9.0],
        "Target":   [0,   0,   0,   1,   1]
    }
    df = pd.DataFrame(data)
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/dummy_api_test.csv", index=False)
    return "data/raw/dummy_api_test.csv"

def main():
    print("=========================================")
    print("TESTING MLFORGE API/UI PIPELINE (PHASE 12)")
    print("=========================================")
    
    # 1. Health check
    res = client.get("/health")
    print(f"Health: {res.json()}")
    
    # 2. Upload dataset
    dummy_path = create_dummy_dataset()
    with open(dummy_path, "rb") as f:
        res = client.post("/datasets/upload", files={"file": ("dummy_api_test.csv", f, "text/csv")})
    
    assert res.status_code == 201
    ds_data = res.json()
    dataset_id = ds_data["dataset_id"]
    print(f"Uploaded Dataset ID: {dataset_id}")
    
    # 3. Create workflow
    res = client.post("/workflows", json={
        "dataset_id": dataset_id,
        "objective": "Predict Target",
        "priority": "accuracy",
        "target_column": "Target",
        "task_type": "classification"
    })
    
    assert res.status_code == 201
    wf_data = res.json()
    wf_id = wf_data["workflow_id"]
    print(f"Workflow ID: {wf_id}")
    
    # 4. Wait for it to finish (in memory job manager)
    # The BackgroundTasks in FastAPI testclient actually run synchronously when using TestClient
    # But let's poll just in case.
    for i in range(10):
        status_res = client.get(f"/workflows/{wf_id}")
        st = status_res.json()
        print(f"Status: {st['status']} ({st['progress']}%) - {st['message']}")
        if st["status"] in ["completed", "failed"]:
            break
        time.sleep(1)
        
    # 5. Fetch events
    ev_res = client.get(f"/workflows/{wf_id}/events")
    events = ev_res.json().get("events", [])
    print(f"\nCollected {len(events)} events.")
    for ev in events:
        print(f"[{ev['agent']}] {ev['event_type']}: {ev['message']}")
        
    # 6. Fetch results
    res_res = client.get(f"/workflows/{wf_id}/result")
    if res_res.status_code == 200:
        result = res_res.json()
        print(f"\nFinal Workflow Result: {result['status']}")
    else:
        print(f"\nResult fetch failed: {res_res.status_code} - {res_res.text}")

if __name__ == "__main__":
    main()
