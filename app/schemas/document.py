from datetime import datetime
from pydantic import BaseModel, ConfigDict


class DocumentUploadResponse(BaseModel):
    message: str
    filename: str
    category: str


class DocumentResponse(BaseModel):
    id: int
    filename: str
    category: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentDetailResponse(DocumentResponse):
    total_chunks: int


class DocumentDeleteResponse(BaseModel):
    message: str
    document_id: int