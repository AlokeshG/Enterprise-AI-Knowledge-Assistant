"""
Application configuration.

Centralizes all environment-driven settings using Pydantic's BaseSettings so
that config is validated at startup and never hardcoded in application code.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Google Gemini (free-tier friendly; no billing required to start) ---
    google_api_key: str = ""
    llm_model: str = "gemini-3.8-flash"
    embedding_model: str = "models/embedding-001"

    # --- App ---
    app_name: str = "Enterprise AI Knowledge Assistant"
    environment: str = "development"
    log_level: str = "INFO"

    # --- Storage paths ---
    upload_dir: str = "./data/uploads"
    chroma_persist_dir: str = "./data/chroma"
    sqlite_db_path: str = "./data/app.db"

    # --- RAG parameters ---
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_top_k: int = 5
    mmr_lambda: float = 0.5

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "*"

    # --- Cache ---
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor so we parse the environment only once."""
    return Settings()
