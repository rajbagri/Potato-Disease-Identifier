from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

# Chat Models
class ChatCreate(BaseModel):
    name: str

class ChatRename(BaseModel):
    new_name: str

class ChatResponse(BaseModel):
    id: str
    name: str
    created_at: str
    message_count: int = 0
    
    class Config:
        from_attributes = True

# Message Models
class MessageCreate(BaseModel):
    content: str
    language: str = "English"

class MessageResponse(BaseModel):
    id: Optional[int]
    sender: str
    content: str
    timestamp: str
    metadata: Optional[dict] = None
    
    class Config:
        from_attributes = True

class ChatDetailResponse(BaseModel):
    id: str
    name: str
    created_at: str
    messages: List[MessageResponse] = []
    
    class Config:
        from_attributes = True

class StreamMessage(BaseModel):
    type: str  # "chunk", "complete", "error", "user_received"
    content: Optional[str] = None
    sources_count: Optional[int] = None
    error: Optional[str] = None

# Health Check
class HealthResponse(BaseModel):
    status: str
    message: str
    database: str
    api_version: str = "1.0.0"

# Image Analysis Models
class DiseaseCandidate(BaseModel):
    disease: str
    display_name: str
    score: float

class MatchedRefImage(BaseModel):
    image_path: str
    disease: str
    similarity_score: float

class ImageAnalysisResponse(BaseModel):
    prediction: str
    confidence: float
    top_candidates: List[DiseaseCandidate] = []
    matched_ref_images: List[MatchedRefImage] = []
    rag_query: str
    rag_response: Optional[str] = None
    source_documents: Optional[List[dict]] = []
    timings: Optional[dict] = None