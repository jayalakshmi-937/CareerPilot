"""
CareerPilot — Semantic Retrieval Service (Phase 4)

Performs similarity search over indexed study materials in ChromaDB.
Provides:
  - retrieve_relevant_documents: returns list of Document objects with metadata
  - retrieve_context_string: formats retrieved chunks into a clean context prompt block
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from langchain_core.documents import Document
from loguru import logger

from backend.config import get_settings
from rag.vectorstore import get_vectorstore


def retrieve_relevant_documents(
    query: str,
    top_k: Optional[int] = None,
    collection_name: Optional[str] = None,
) -> List[Document]:
    """
    Search ChromaDB for the top_k chunks most semantically relevant to the query.
    """
    settings = get_settings()
    k = top_k or settings.retrieval_top_k

    if not query.strip():
        logger.warning("Empty query provided for document retrieval.")
        return []

    try:
        vectorstore = get_vectorstore(collection_name=collection_name)
        total_in_db = vectorstore._collection.count()

        if total_in_db == 0:
            logger.info("ChromaDB is empty; no documents retrieved.")
            return []

        # Adjust k if fewer documents exist in the collection
        actual_k = min(k, total_in_db)
        docs = vectorstore.similarity_search(query, k=actual_k)
        logger.info(f"Retrieved {len(docs)} relevant chunks for query: '{query[:50]}…'")
        return docs

    except Exception as exc:
        logger.error(f"Retrieval error for query '{query[:50]}': {exc}")
        return []


def retrieve_relevant_with_scores(
    query: str,
    top_k: Optional[int] = None,
    collection_name: Optional[str] = None,
) -> List[Tuple[Document, float]]:
    """
    Search ChromaDB and return documents along with similarity distance scores.
    """
    settings = get_settings()
    k = top_k or settings.retrieval_top_k

    if not query.strip():
        return []

    try:
        vectorstore = get_vectorstore(collection_name=collection_name)
        total = vectorstore._collection.count()
        if total == 0:
            return []

        actual_k = min(k, total)
        return vectorstore.similarity_search_with_score(query, k=actual_k)
    except Exception as exc:
        logger.error(f"Scored retrieval error: {exc}")
        return []


def retrieve_context_string(
    query: str,
    top_k: Optional[int] = None,
    collection_name: Optional[str] = None,
) -> str:
    """
    Retrieves relevant chunks and formats them as a clean text block
    ready to inject directly into LLM prompt templates.
    """
    docs = retrieve_relevant_documents(query, top_k=top_k, collection_name=collection_name)

    if not docs:
        return ""

    context_blocks: List[str] = []
    for idx, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Study Material")
        context_blocks.append(
            f"--- Document Source {idx}: {source} ---\n{doc.page_content.strip()}"
        )

    return "\n\n".join(context_blocks)
