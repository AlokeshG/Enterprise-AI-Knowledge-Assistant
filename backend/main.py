"""
FastAPI application entry point.

Wires together configuration, database initialization, routers, CORS, and
a global exception handler that converts our custom AppException hierarchy
into clean JSON error responses.
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config.settings import get_settings
from backend.database.db import init_db
from backend.routers import documents, query
from backend.utils.exceptions import AppException
from backend.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)

app = FastAPI(
    title=settings.app_name,
    description="Enterprise-grade RAG knowledge assistant: upload documents, "
                "ask grounded questions, get cited answers.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.cors_origins] if settings.cors_origins != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    logger.info("Starting %s in %s mode", settings.app_name, settings.environment)
    init_db()


@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    logger.warning("Handled exception on %s: %s", request.url.path, exc.message)
    return JSONResponse(status_code=exc.status_code, content={"error": exc.message})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s", request.url.path)
    return JSONResponse(status_code=500, content={"error": "An unexpected error occurred."})


@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok", "app": settings.app_name}


app.include_router(documents.router, prefix="/api")
app.include_router(query.router, prefix="/api")
