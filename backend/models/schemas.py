"""
Pydantic schemas used across the API layer.

Keeping these separate from the SQLAlchemy ORM models (database/db.py)
follows the standard FastAPI pattern of decoupling the wire format from
the persistence format.
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ---------- Documents ----------

class DocumentOut(BaseModel):
    id: str
    filename: str
    file_type: str
    size_bytes: int
    num_chunks: int
    status: str  # "processing" | "indexed" | "failed"
    uploaded_at: datetime

    class Config:
        from_attributes = True


class DocumentUploadResponse(BaseModel):
    documents: list[DocumentOut]
    message: str


class ReindexResponse(BaseModel):
    document_id: str
    num_chunks: int
    message: str


# ---------- Query / Chat ----------

class SourceChunk(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    content: str
    relevance_score: float


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[str] = None
    document_ids: Optional[list[str]] = None  # metadata filter: restrict to specific docs
    retrieval_strategy: str = Field(default="similarity")  # "similarity" | "mmr" | "hybrid"
    top_k: Optional[int] = None
    stream: bool = False


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float


class QueryResponse(BaseModel):
    conversation_id: str
    answer: str
    sources: list[SourceChunk]
    confidence_score: float
    latency_ms: float
    token_usage: TokenUsage
    from_cache: bool = False


class ConversationTurn(BaseModel):
    role: str  # "user" | "assistant"
    content: str
    timestamp: datetime


class ConversationHistory(BaseModel):
    conversation_id: str
    turns: list[ConversationTurn]
