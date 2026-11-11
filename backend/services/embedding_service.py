"""
Embedding / ingestion service.

Implements the LangChain ingestion pipeline:

    Document Loader -> Text Splitter -> Embedding Model -> ChromaDB

Each uploaded file is loaded with the appropriate LangChain loader,
split into overlapping chunks, embedded with Google Gemini embeddings, and
persisted into a ChromaDB collection. Metadata (document_id, filename,
chunk_index) is attached to every chunk so we can later filter retrieval
by document and show source citations.
"""
import os
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from backend.config.settings import get_settings
from backend.utils.logger import get_logger
from backend.utils.exceptions import (
    UnsupportedFileTypeError,
    DocumentProcessingError,
    EmbeddingGenerationError,
)

logger = get_logger(__name__)
settings = get_settings()

_LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
}


class EmbeddingService:
    """Owns the Chroma vector store and the ingestion pipeline."""

    def __init__(self):
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.google_api_key or None,
        )
        self._vector_store = Chroma(
            collection_name="enterprise_documents",
            embedding_function=self._embeddings,
            persist_directory=settings.chroma_persist_dir,
        )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def load_document(self, filepath: str, filename: str) -> list[Document]:
        """Load a single file into LangChain Document objects."""
        ext = os.path.splitext(filename)[1].lower()
        loader_cls = _LOADER_MAP.get(ext)
        if loader_cls is None:
            raise UnsupportedFileTypeError(filename)

        try:
            loader = loader_cls(filepath)
            return loader.load()
        except Exception as exc:
            logger.exception("Failed to load document %s", filename)
            raise DocumentProcessingError(filename, str(exc)) from exc

    def chunk_document(self, docs: list[Document]) -> list[Document]:
        return self._splitter.split_documents(docs)

    def index_document(self, document_id: str, filepath: str, filename: str) -> int:
        """
        Full ingestion pipeline for a single document.
        Returns the number of chunks indexed.
        """
        raw_docs = self.load_document(filepath, filename)
        chunks = self.chunk_document(raw_docs)

        if not chunks:
            raise DocumentProcessingError(filename, "No extractable text content found.")

        for idx, chunk in enumerate(chunks):
            chunk.metadata.update({
                "document_id": document_id,
                "filename": filename,
                "chunk_index": idx,
            })

        try:
            self._vector_store.add_documents(chunks)
        except Exception as exc:
            logger.exception("Embedding generation failed for %s", filename)
            raise EmbeddingGenerationError(str(exc)) from exc

        logger.info("Indexed %d chunks for document %s (%s)", len(chunks), filename, document_id)
        return len(chunks)

    def delete_document(self, document_id: str) -> None:
        try:
            self._vector_store.delete(where={"document_id": document_id})
        except ValueError:
            logger.info("No vector chunks found for document %s; nothing to delete", document_id)
        logger.info("Deleted all chunks for document %s", document_id)

    def get_retriever(
        self,
        strategy: str = "similarity",
        top_k: Optional[int] = None,
        document_ids: Optional[list[str]] = None,
    ):
        """
        Build a retriever per request. Supports:
          - similarity: plain top-k cosine similarity
          - mmr: Maximal Marginal Relevance (diversity-aware retrieval)
          - hybrid: similarity search widened then re-ranked (simple hybrid approximation)
        """
        k = top_k or settings.retrieval_top_k
        search_kwargs: dict = {"k": k}

        if document_ids:
            search_kwargs["filter"] = {"document_id": {"$in": document_ids}}

        if strategy == "mmr":
            search_kwargs["lambda_mult"] = settings.mmr_lambda
            search_kwargs["fetch_k"] = max(k * 4, 20)
            return self._vector_store.as_retriever(search_type="mmr", search_kwargs=search_kwargs)

        if strategy == "hybrid":
            # Simple hybrid approximation: widen the similarity pool, rely on
            # downstream re-ranking (query rewriting + prompt grounding) to
            # surface the most relevant chunks. A true hybrid (BM25 + vector)
            # is documented as a future improvement in README.md.
            search_kwargs["k"] = max(k * 2, 10)
            return self._vector_store.as_retriever(search_type="similarity", search_kwargs=search_kwargs)

        return self._vector_store.as_retriever(search_type="similarity", search_kwargs=search_kwargs)

    @property
    def vector_store(self) -> Chroma:
        return self._vector_store


_service_instance: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    global _service_instance
    if _service_instance is None:
        _service_instance = EmbeddingService()
    return _service_instance
