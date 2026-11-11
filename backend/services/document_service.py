"""
Document management service.

Owns the lifecycle of an uploaded document: saving the raw file to disk,
creating/updating its metadata row in SQLite, and delegating the actual
chunk/embed work to EmbeddingService. Keeping this separate from
EmbeddingService follows single-responsibility: this class knows about
files and metadata, EmbeddingService knows about vectors.
"""
import os
import uuid
import shutil
from sqlalchemy.orm import Session

from backend.config.settings import get_settings
from backend.database.db import DocumentRecord
from backend.services.embedding_service import get_embedding_service
from backend.utils.exceptions import DocumentNotFoundError, UnsupportedFileTypeError
from backend.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


class DocumentService:
    def __init__(self):
        os.makedirs(settings.upload_dir, exist_ok=True)
        self._embedding_service = get_embedding_service()

    def save_and_index(self, db: Session, filename: str, file_bytes: bytes) -> DocumentRecord:
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise UnsupportedFileTypeError(filename)

        document_id = str(uuid.uuid4())
        stored_filename = f"{document_id}{ext}"
        filepath = os.path.join(settings.upload_dir, stored_filename)

        with open(filepath, "wb") as f:
            f.write(file_bytes)

        record = DocumentRecord(
            id=document_id,
            filename=filename,
            file_type=ext.lstrip("."),
            size_bytes=len(file_bytes),
            status="processing",
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        try:
            num_chunks = self._embedding_service.index_document(document_id, filepath, filename)
            record.num_chunks = num_chunks
            record.status = "indexed"
        except Exception as exc:
            record.status = "failed"
            record.error_message = str(exc)
            logger.exception("Indexing failed for %s", filename)
        finally:
            db.commit()
            db.refresh(record)

        return record

    def list_documents(self, db: Session) -> list[DocumentRecord]:
        return db.query(DocumentRecord).order_by(DocumentRecord.uploaded_at.desc()).all()

    def get_document(self, db: Session, document_id: str) -> DocumentRecord:
        record = db.query(DocumentRecord).filter(DocumentRecord.id == document_id).first()
        if record is None:
            raise DocumentNotFoundError(document_id)
        return record

    def delete_document(self, db: Session, document_id: str) -> None:
        record = self.get_document(db, document_id)
        self._embedding_service.delete_document(document_id)

        ext = f".{record.file_type}"
        filepath = os.path.join(settings.upload_dir, f"{document_id}{ext}")
        if os.path.exists(filepath):
            os.remove(filepath)

        db.delete(record)
        db.commit()
        logger.info("Deleted document %s (%s)", document_id, record.filename)

    def reindex_document(self, db: Session, document_id: str) -> int:
        record = self.get_document(db, document_id)
        ext = f".{record.file_type}"
        filepath = os.path.join(settings.upload_dir, f"{document_id}{ext}")

        self._embedding_service.delete_document(document_id)
        num_chunks = self._embedding_service.index_document(document_id, filepath, record.filename)

        record.num_chunks = num_chunks
        record.status = "indexed"
        record.error_message = None
        db.commit()
        db.refresh(record)
        return num_chunks


_service_instance: DocumentService | None = None


def get_document_service() -> DocumentService:
    global _service_instance
    if _service_instance is None:
        _service_instance = DocumentService()
    return _service_instance
