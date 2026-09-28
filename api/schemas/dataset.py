from pydantic import BaseModel

class DatasetUploadResponse(BaseModel):
    dataset_id: str
    filename: str
    size_bytes: int
    status: str
