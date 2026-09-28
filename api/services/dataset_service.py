import os
import uuid
import shutil
import hashlib
from fastapi import UploadFile, HTTPException
from core.database import SessionLocal
from models.domain import Dataset

UPLOAD_DIR = "data/uploads"

class DatasetService:
    @staticmethod
    def save_upload(file: UploadFile, user_id: str) -> dict:
        if not file.filename.endswith(('.csv', '.xlsx')):
            raise HTTPException(status_code=422, detail="Only .csv and .xlsx files are supported.")
            
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        # Generate safe internal ID
        dataset_id = f"ds_{uuid.uuid4().hex[:8]}"
        ext = os.path.splitext(file.filename)[1]
        safe_filename = f"{dataset_id}{ext}"
        file_path = os.path.join(UPLOAD_DIR, safe_filename)
        
        sha256_hash = hashlib.sha256()
        with open(file_path, "wb") as buffer:
            while chunk := file.file.read(8192):
                sha256_hash.update(chunk)
                buffer.write(chunk)
                
        dataset_hash = sha256_hash.hexdigest()
            
        size_bytes = os.path.getsize(file_path)
        from config.settings import get_settings
        settings = get_settings()
        MAX_UPLOAD_SIZE = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if size_bytes > MAX_UPLOAD_SIZE:
            os.remove(file_path)
            raise HTTPException(status_code=413, detail="File too large.")
            
        db = SessionLocal()
        try:
            ds = Dataset(
                id=dataset_id,
                owner_id=user_id,
                filename=file.filename,
                dataset_hash=dataset_hash,
                size_bytes=size_bytes,
                storage_path=file_path
            )
            db.add(ds)
            db.commit()
        finally:
            db.close()
            
        return {
            "dataset_id": dataset_id,
            "filename": file.filename,
            "safe_path": file_path,
            "size_bytes": size_bytes,
            "status": "uploaded"
        }
        
    @staticmethod
    def get_dataset_path(dataset_id: str) -> str:
        # Prevent path traversal
        if ".." in dataset_id or "/" in dataset_id or "\\" in dataset_id:
            return None
            
        db = SessionLocal()
        try:
            ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if ds:
                return ds.storage_path
        finally:
            db.close()
            
        # Fallback to local search for backwards compatibility with existing datasets
        for f in os.listdir(UPLOAD_DIR):
            if f.startswith(dataset_id):
                return os.path.join(UPLOAD_DIR, f)
        return None
