"""
Database layer (SQLite via SQLAlchemy).

Stores document metadata and conversation history. The actual vector
embeddings live in ChromaDB (see services/embedding_service.py) -- SQLite
here is only for structured, relational metadata that ChromaDB is not
well suited to query (status, timestamps, chat turns, etc).
"""
import os
import uuid
from datetime import datetime, timezone

from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, Float
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from backend.config.settings import get_settings

settings = get_settings()
os.makedirs(os.path.dirname(settings.sqlite_db_path) or ".", exist_ok=True)

engine = create_engine(
    f"sqlite:///{settings.sqlite_db_path}",
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class DocumentRecord(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    size_bytes = Column(Integer, nullable=False)
    num_chunks = Column(Integer, default=0)
    status = Column(String, default="processing")  # processing | indexed | failed
    error_message = Column(Text, nullable=True)
    uploaded_at = Column(DateTime, default=now_utc)


class ConversationTurnRecord(Base):
    __tablename__ = "conversation_turns"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String, index=True, nullable=False)
    role = Column(String, nullable=False)  # user | assistant
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=now_utc)


class QueryLogRecord(Base):
    """Audit log of every query for token-usage tracking / observability."""
    __tablename__ = "query_logs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String, index=True, nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    latency_ms = Column(Float, default=0.0)
    confidence_score = Column(Float, default=0.0)
    from_cache = Column(Integer, default=0)  # 0/1 boolean
    timestamp = Column(DateTime, default=now_utc)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    """FastAPI dependency that yields a DB session and guarantees closure."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
