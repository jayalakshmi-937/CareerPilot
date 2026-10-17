"""
CareerPilot — LLM Factory (Phase 2)

Provides a cached singleton ChatGoogleGenerativeAI instance.
All agent nodes import from here — only one LLM object is ever created.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI
from loguru import logger

from backend.config import get_settings


@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    """
    Return a cached Gemini LLM instance configured from settings.

    Uses langchain-google-genai so the LLM integrates cleanly with
    LangChain prompt templates and LangGraph nodes.
    """
    settings = get_settings()

    logger.info(f"Initialising LLM: {settings.llm_model}")

    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.gemini_api_key,
        temperature=settings.llm_temperature,
        max_output_tokens=settings.llm_max_tokens,
        convert_system_message_to_human=False,
    )
    return llm
