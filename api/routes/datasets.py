import os
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from api.schemas.dataset import DatasetUploadResponse
from api.services.dataset_service import DatasetService
from api.dependencies.auth import get_current_user
from models.domain import User

router = APIRouter(prefix="/datasets", tags=["datasets"])

@router.post("/upload", response_model=DatasetUploadResponse, status_code=201)
def upload_dataset(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    return DatasetService.save_upload(file, current_user.id)

@router.get("/{dataset_id}/columns")
def get_dataset_columns(dataset_id: str, current_user: User = Depends(get_current_user)):
    """Return columns, dtypes, shape and 5-row preview for a dataset."""
    path = DatasetService.get_dataset_path(dataset_id)
    if not path:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found.")
    try:
        df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
        preview = df.head(5).fillna("").astype(str).to_dict(orient="records")
        dtypes = {col: str(df[col].dtype) for col in df.columns}
        missing = int(df.isnull().sum().sum())
        return {
            "dataset_id": dataset_id,
            "columns": list(df.columns),
            "dtypes": dtypes,
            "row_count": len(df),
            "col_count": len(df.columns),
            "missing_values": missing,
            "preview": preview
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read dataset: {e}")
