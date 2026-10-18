"""
CareerPilot — Embeddings Factory (Phase 4 — RAG Integration)

Provides a cached singleton Embeddings instance using Google Generative AI embeddings.
Dynamically resolves Gemini API key.
"""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from loguru import logger

from backend.config import get_settings


def get_effective_embedding_api_key() -> str:
    """Resolve the Gemini API key from environment, settings, or Streamlit secrets."""
    key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    if key:
        return key

    settings = get_settings()
    if settings.gemini_api_key and settings.gemini_api_key.strip():
        return settings.gemini_api_key.strip()

    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if "GEMINI_API_KEY" in st.secrets and str(st.secrets["GEMINI_API_KEY"]).strip():
                return str(st.secrets["GEMINI_API_KEY"]).strip()
            if "GOOGLE_API_KEY" in st.secrets and str(st.secrets["GOOGLE_API_KEY"]).strip():
                return str(st.secrets["GOOGLE_API_KEY"]).strip()
    except Exception:
        pass

    return ""


@lru_cache(maxsize=8)
def _create_embeddings_instance(model: str, api_key: str) -> Embeddings:
    if not api_key:
        logger.warning("GEMINI_API_KEY is missing; embedding operations may fail.")
    logger.info(f"Initializing embeddings with model: {model}")
    return GoogleGenerativeAIEmbeddings(
        model=model,
        google_api_key=api_key or None,
    )


def get_embeddings() -> Embeddings:
    """
    Return a singleton Embeddings instance.
    Dynamically resolves API key so late secret updates work immediately.
    """
    settings = get_settings()
    api_key = get_effective_embedding_api_key()
    return _create_embeddings_instance(settings.embedding_model, api_key)
