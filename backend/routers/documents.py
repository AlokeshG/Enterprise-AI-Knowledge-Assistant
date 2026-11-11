"""
Document management endpoints: upload, list, delete, reindex.
"""
from fastapi import APIRouter, UploadFile, File, Depends
from sqlalchemy.orm import Session

from backend.database.db import get_db
from backend.services.document_service import get_document_service
from backend.models.schemas import DocumentOut, DocumentUploadResponse, ReindexResponse

router = APIRouter(tags=["Documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_documents(
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    service = get_document_service()
    results = []
    for file in files:
        contents = await file.read()
        record = service.save_and_index(db, file.filename, contents)
        results.append(DocumentOut.model_validate(record))

    return DocumentUploadResponse(
        documents=results,
        message=f"Processed {len(results)} document(s).",
    )


@router.get("/documents", response_model=list[DocumentOut])
def list_documents(db: Session = Depends(get_db)):
    service = get_document_service()
    records = service.list_documents(db)
    return [DocumentOut.model_validate(r) for r in records]


@router.delete("/document/{document_id}")
def delete_document(document_id: str, db: Session = Depends(get_db)):
    service = get_document_service()
    service.delete_document(db, document_id)
    return {"message": f"Document {document_id} deleted."}


@router.post("/reindex/{document_id}", response_model=ReindexResponse)
def reindex_document(document_id: str, db: Session = Depends(get_db)):
    service = get_document_service()
    num_chunks = service.reindex_document(db, document_id)
    return ReindexResponse(
        document_id=document_id,
        num_chunks=num_chunks,
        message="Document re-indexed successfully.",
    )
