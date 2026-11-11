"""
RAG (Retrieval-Augmented Generation) service.

Implements the full LangChain pipeline described in the spec:

    Query Rewriting -> Retriever -> Prompt Template -> LLM -> Output Parser

Design notes:
- The system prompt strictly forbids answering outside the retrieved
  context, to prevent hallucination.
- Confidence score is derived from the retriever's similarity scores
  (not the LLM's own self-report, which is unreliable) -- a simple but
  defensible heuristic for a portfolio project.
- Conversation memory is persisted to SQLite (not just in-process) so
  history survives server restarts, which matters for an "enterprise"
  positioning.
- Caching uses a hash of (question + document filter + strategy) so
  identical repeated questions skip the LLM call entirely.
"""
import time
import uuid
from typing import Optional, Generator

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sqlalchemy.orm import Session

from backend.config.settings import get_settings
from backend.database.db import ConversationTurnRecord, QueryLogRecord
from backend.services.embedding_service import get_embedding_service
from backend.utils.cache import get_cache
from backend.utils.logger import get_logger
from backend.utils.exceptions import LLMGenerationError, RetrievalError
from backend.models.schemas import QueryResponse, SourceChunk, TokenUsage

logger = get_logger(__name__)
settings = get_settings()

SYSTEM_PROMPT = """You are an enterprise knowledge assistant. Answer the user's \
question using ONLY the context provided below, which was retrieved from the \
company's uploaded documents.

Rules:
1. Only use information present in the context. Never use outside knowledge.
2. Never hallucinate or make up facts, numbers, or policy details.
3. If the context does not contain the answer, respond exactly with:
   "I couldn't find relevant information in the uploaded documents."
4. When you do answer, be concise and cite which part of the context supports it.
5. If the conversation history is relevant to interpreting the question, use it.

Context:
{context}

Conversation history:
{history}
"""

QUERY_REWRITE_PROMPT = """Rewrite the user's follow-up question into a fully \
self-contained question, using the conversation history for context. If the \
question is already self-contained, return it unchanged. Return ONLY the \
rewritten question, nothing else.

Conversation history:
{history}

Follow-up question: {question}

Rewritten question:"""

# Approximate Gemini 1.5 Flash pricing (USD per 1K tokens) for the token-usage
# estimate. Well within the free tier's daily quota for demo/testing use.
_PRICE_PER_1K_PROMPT = 0.000075
_PRICE_PER_1K_COMPLETION = 0.0003


class RAGService:
    def __init__(self):
        self._llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.google_api_key or None,
            temperature=0,
        )
        self._parser = StrOutputParser()
        self._embedding_service = get_embedding_service()
        self._cache = get_cache()

    # ---------- Conversation memory ----------

    def _load_history(self, db: Session, conversation_id: str, max_turns: int = 6) -> str:
        turns = (
            db.query(ConversationTurnRecord)
            .filter(ConversationTurnRecord.conversation_id == conversation_id)
            .order_by(ConversationTurnRecord.timestamp.desc())
            .limit(max_turns)
            .all()
        )
        turns.reverse()
        if not turns:
            return "(no prior conversation)"
        return "\n".join(f"{t.role.upper()}: {t.content}" for t in turns)

    def _save_turn(self, db: Session, conversation_id: str, role: str, content: str) -> None:
        db.add(ConversationTurnRecord(conversation_id=conversation_id, role=role, content=content))
        db.commit()

    # ---------- Query rewriting ----------

    def _rewrite_query(self, question: str, history: str) -> str:
        if history == "(no prior conversation)":
            return question
        try:
            prompt = ChatPromptTemplate.from_template(QUERY_REWRITE_PROMPT)
            chain = prompt | self._llm | self._parser
            rewritten = chain.invoke({"history": history, "question": question}).strip()
            return rewritten or question
        except Exception:
            logger.warning("Query rewriting failed, falling back to original question")
            return question

    # ---------- Retrieval ----------

    def _retrieve(self, question: str, strategy: str, top_k: Optional[int], document_ids: Optional[list[str]]):
        try:
            retriever = self._embedding_service.get_retriever(
                strategy=strategy, top_k=top_k, document_ids=document_ids
            )
            docs = retriever.invoke(question)
            return docs
        except Exception as exc:
            logger.exception("Retrieval failed")
            raise RetrievalError(str(exc)) from exc

    def _score_documents(self, question: str, k: int) -> list[tuple]:
        """Get (document, similarity_score) pairs for confidence estimation."""
        try:
            return self._embedding_service.vector_store.similarity_search_with_relevance_scores(
                question, k=k
            )
        except Exception:
            return []

    # ---------- Token / cost estimation ----------

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        # Rough heuristic: ~4 characters per token (avoids a tiktoken dependency).
        return max(1, len(text) // 4)

    def _build_token_usage(self, prompt_text: str, completion_text: str) -> TokenUsage:
        prompt_tokens = self._estimate_tokens(prompt_text)
        completion_tokens = self._estimate_tokens(completion_text)
        cost = (
            (prompt_tokens / 1000) * _PRICE_PER_1K_PROMPT
            + (completion_tokens / 1000) * _PRICE_PER_1K_COMPLETION
        )
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            estimated_cost_usd=round(cost, 6),
        )

    # ---------- Main query pipeline ----------

    def answer_query(
        self,
        db: Session,
        question: str,
        conversation_id: Optional[str],
        document_ids: Optional[list[str]],
        strategy: str,
        top_k: Optional[int],
    ) -> QueryResponse:
        start = time.perf_counter()
        conversation_id = conversation_id or str(uuid.uuid4())

        cache_key = get_cache().make_key(question, document_ids, strategy, top_k)
        if settings.enable_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.info("Cache hit for question: %s", question[:60])
                cached.conversation_id = conversation_id
                cached.from_cache = True
                return cached

        history = self._load_history(db, conversation_id)
        rewritten_question = self._rewrite_query(question, history)

        k = top_k or settings.retrieval_top_k
        docs = self._retrieve(rewritten_question, strategy, k, document_ids)
        scored_docs = self._score_documents(rewritten_question, k)

        if not docs:
            answer = "I couldn't find relevant information in the uploaded documents."
            sources: list[SourceChunk] = []
            confidence = 0.0
        else:
            context = "\n\n---\n\n".join(
                f"[Source: {d.metadata.get('filename')}, chunk {d.metadata.get('chunk_index')}]\n{d.page_content}"
                for d in docs
            )
            try:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", SYSTEM_PROMPT),
                    ("human", "{question}"),
                ])
                chain = prompt | self._llm | self._parser
                answer = chain.invoke({
                    "context": context,
                    "history": history,
                    "question": rewritten_question,
                })
            except Exception as exc:
                logger.exception("LLM generation failed")
                raise LLMGenerationError(str(exc)) from exc

            sources = [
                SourceChunk(
                    document_id=d.metadata.get("document_id", "unknown"),
                    filename=d.metadata.get("filename", "unknown"),
                    chunk_index=d.metadata.get("chunk_index", -1),
                    content=d.page_content[:500],
                    relevance_score=round(score, 4),
                )
                for d, score in scored_docs
            ] if scored_docs else [
                SourceChunk(
                    document_id=d.metadata.get("document_id", "unknown"),
                    filename=d.metadata.get("filename", "unknown"),
                    chunk_index=d.metadata.get("chunk_index", -1),
                    content=d.page_content[:500],
                    relevance_score=0.0,
                )
                for d in docs
            ]
            confidence = round(sum(s.relevance_score for s in sources) / len(sources), 4) if sources else 0.0

        latency_ms = (time.perf_counter() - start) * 1000
        token_usage = self._build_token_usage(context if docs else question, answer)

        response = QueryResponse(
            conversation_id=conversation_id,
            answer=answer,
            sources=sources,
            confidence_score=confidence,
            latency_ms=round(latency_ms, 2),
            token_usage=token_usage,
            from_cache=False,
        )

        self._save_turn(db, conversation_id, "user", question)
        self._save_turn(db, conversation_id, "assistant", answer)
        db.add(QueryLogRecord(
            conversation_id=conversation_id,
            question=question,
            answer=answer,
            prompt_tokens=token_usage.prompt_tokens,
            completion_tokens=token_usage.completion_tokens,
            latency_ms=latency_ms,
            confidence_score=confidence,
            from_cache=0,
        ))
        db.commit()

        if settings.enable_cache:
            self._cache.set(cache_key, response)

        return response

    def stream_answer(
        self,
        db: Session,
        question: str,
        conversation_id: Optional[str],
        document_ids: Optional[list[str]],
        strategy: str,
        top_k: Optional[int],
    ) -> Generator[str, None, None]:
        """Streaming variant: yields answer tokens as they're generated."""
        conversation_id = conversation_id or str(uuid.uuid4())
        history = self._load_history(db, conversation_id)
        rewritten_question = self._rewrite_query(question, history)

        k = top_k or settings.retrieval_top_k
        docs = self._retrieve(rewritten_question, strategy, k, document_ids)

        if not docs:
            fallback = "I couldn't find relevant information in the uploaded documents."
            yield fallback
            self._save_turn(db, conversation_id, "user", question)
            self._save_turn(db, conversation_id, "assistant", fallback)
            return

        context = "\n\n---\n\n".join(d.page_content for d in docs)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])
        chain = prompt | self._llm | self._parser

        full_answer = ""
        try:
            for token in chain.stream({"context": context, "history": history, "question": rewritten_question}):
                full_answer += token
                yield token
        except Exception as exc:
            logger.exception("Streaming generation failed")
            raise LLMGenerationError(str(exc)) from exc
        finally:
            if full_answer:
                self._save_turn(db, conversation_id, "user", question)
                self._save_turn(db, conversation_id, "assistant", full_answer)

    def get_history(self, db: Session, conversation_id: str) -> list[ConversationTurnRecord]:
        return (
            db.query(ConversationTurnRecord)
            .filter(ConversationTurnRecord.conversation_id == conversation_id)
            .order_by(ConversationTurnRecord.timestamp.asc())
            .all()
        )


_service_instance: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    global _service_instance
    if _service_instance is None:
        _service_instance = RAGService()
    return _service_instance
