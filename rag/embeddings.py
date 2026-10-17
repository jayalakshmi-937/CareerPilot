"""
CareerPilot — Embeddings Provider (Phase 4)

Provides a singleton embedding model instance via langchain-google-genai.
Uses GoogleGenerativeAIEmbeddings configured with the embedding model
from settings (e.g. models/gemini-embedding-001).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from loguru import logger

from backend.config import get_settings


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """
    Return a cached singleton Embeddings instance.
    Configured from project settings (.env).
    """
    settings = get_settings()

    if not settings.gemini_api_key:
        logger.warning("GEMINI_API_KEY is missing; embedding operations may fail.")

    logger.info(f"Initializing embeddings with model: {settings.embedding_model}")

    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.gemini_api_key,
    )
