"""
Query / chat endpoints, including streaming responses and conversation history.
"""
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.database.db import get_db
from backend.services.rag_service import get_rag_service
from backend.models.schemas import QueryRequest, QueryResponse, ConversationHistory, ConversationTurn

router = APIRouter(tags=["Query"])


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, db: Session = Depends(get_db)):
    service = get_rag_service()

    if request.stream:
        # For non-streaming clients that set stream=True by mistake, we still
        # return a normal JSON response via the non-streaming path.
        pass

    return service.answer_query(
        db=db,
        question=request.question,
        conversation_id=request.conversation_id,
        document_ids=request.document_ids,
        strategy=request.retrieval_strategy,
        top_k=request.top_k,
    )


@router.post("/query/stream")
def query_stream(request: QueryRequest, db: Session = Depends(get_db)):
    service = get_rag_service()

    def token_generator():
        for token in service.stream_answer(
            db=db,
            question=request.question,
            conversation_id=request.conversation_id,
            document_ids=request.document_ids,
            strategy=request.retrieval_strategy,
            top_k=request.top_k,
        ):
            yield token

    return StreamingResponse(token_generator(), media_type="text/plain")


@router.get("/history/{conversation_id}", response_model=ConversationHistory)
def get_history(conversation_id: str, db: Session = Depends(get_db)):
    service = get_rag_service()
    turns = service.get_history(db, conversation_id)
    return ConversationHistory(
        conversation_id=conversation_id,
        turns=[
            ConversationTurn(role=t.role, content=t.content, timestamp=t.timestamp)
            for t in turns
        ],
    )
