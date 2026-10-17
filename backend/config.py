"""
CareerPilot — Application Configuration
Loads all settings from environment variables (via .env).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Project root ──────────────────────────────────────────────────────────────
ROOT_DIR: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """All application settings, loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # ── LLM ─────────────────────────────────────────────────────────────────
    gemini_api_key: str = ""
    llm_model: str = "gemini-3.5-flash-lite"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 8192

    # ── Embeddings ───────────────────────────────────────────────────────────
    embedding_model: str = "models/gemini-embedding-001"

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    chroma_persist_dir: str = str(ROOT_DIR / "chroma_db")
    chroma_collection_name: str = "careerpilot_docs"

    # ── SQLite ───────────────────────────────────────────────────────────────
    database_url: str = f"sqlite:///{ROOT_DIR / 'database' / 'careerpilot.db'}"

    # ── RAG ─────────────────────────────────────────────────────────────────
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_top_k: int = 5

    # ── Streamlit ────────────────────────────────────────────────────────────
    streamlit_port: int = 8501

    # ── Derived helpers ──────────────────────────────────────────────────────
    @property
    def documents_dir(self) -> Path:
        return ROOT_DIR / "data" / "documents"

    @property
    def is_development(self) -> bool:
        return self.app_env.lower() == "development"

    def validate_required(self) -> None:
        """Raise ValueError if mandatory secrets are missing."""
        if not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Copy .env.example → .env and add your key."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
