"""
Custom exception hierarchy.

Using typed exceptions (instead of raising generic Exception) lets the
FastAPI exception handlers return precise HTTP status codes and lets
calling code catch failures selectively.
"""


class AppException(Exception):
    """Base class for all application-specific exceptions."""

    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class DocumentNotFoundError(AppException):
    def __init__(self, document_id: str):
        super().__init__(f"Document '{document_id}' was not found.", status_code=404)


class UnsupportedFileTypeError(AppException):
    def __init__(self, filename: str):
        super().__init__(
            f"Unsupported file type for '{filename}'. Allowed: .pdf, .docx, .txt",
            status_code=400,
        )


class DocumentProcessingError(AppException):
    def __init__(self, filename: str, reason: str):
        super().__init__(f"Failed to process '{filename}': {reason}", status_code=422)


class EmbeddingGenerationError(AppException):
    def __init__(self, reason: str):
        super().__init__(f"Embedding generation failed: {reason}", status_code=502)


class RetrievalError(AppException):
    def __init__(self, reason: str):
        super().__init__(f"Retrieval failed: {reason}", status_code=502)


class LLMGenerationError(AppException):
    def __init__(self, reason: str):
        super().__init__(f"LLM generation failed: {reason}", status_code=502)
