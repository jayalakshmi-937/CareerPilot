"""
CareerPilot — Vector Storage with ChromaDB (Phase 4)

Manages the ChromaDB vector database:
  - Vectorstore initialization with GoogleGenerativeAIEmbeddings
  - Storing and indexing document chunks
  - Querying collection status and count
  - Resetting / clearing vectorstore
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from loguru import logger

from backend.config import get_settings
from rag.embeddings import get_embeddings


def get_vectorstore(
    collection_name: Optional[str] = None,
    persist_directory: Optional[str] = None,
) -> Chroma:
    """
    Get or initialize a Chroma vector store instance.
    Uses the configured GoogleGenerativeAIEmbeddings model.
    """
    settings = get_settings()
    c_name = collection_name or settings.chroma_collection_name
    p_dir = persist_directory or settings.chroma_persist_dir

    # Ensure persist directory exists
    Path(p_dir).mkdir(parents=True, exist_ok=True)
    embeddings = get_embeddings()

    vectorstore = Chroma(
        collection_name=c_name,
        embedding_function=embeddings,
        persist_directory=p_dir,
    )
    return vectorstore


def index_documents(
    documents: List[Document],
    collection_name: Optional[str] = None,
    persist_directory: Optional[str] = None,
) -> int:
    """
    Embeds and stores document chunks into ChromaDB.
    Returns the number of chunks successfully indexed.
    """
    if not documents:
        logger.warning("No documents provided to index into ChromaDB.")
        return 0

    vectorstore = get_vectorstore(collection_name, persist_directory)

    logger.info(f"Indexing {len(documents)} document chunks into ChromaDB collection '{vectorstore._collection.name}'...")
    # Add documents in batches to avoid payload limits
    batch_size = 50
    total_added = 0

    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        vectorstore.add_documents(batch)
        total_added += len(batch)
        logger.debug(f"Indexed batch {i // batch_size + 1}: {len(batch)} chunks (Total: {total_added}/{len(documents)})")

    logger.success(f"Successfully indexed {total_added} document chunks into ChromaDB.")
    return total_added


def get_collection_stats(
    collection_name: Optional[str] = None,
    persist_directory: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Returns statistics about the current ChromaDB collection.
    """
    try:
        vectorstore = get_vectorstore(collection_name, persist_directory)
        count = vectorstore._collection.count()
        return {
            "status": "ready",
            "collection_name": vectorstore._collection.name,
            "total_chunks": count,
            "persist_directory": str(Path(vectorstore._persist_directory).resolve()) if hasattr(vectorstore, "_persist_directory") else None,
        }
    except Exception as exc:
        logger.error(f"Failed to get ChromaDB stats: {exc}")
        return {
            "status": "error",
            "error": str(exc),
            "total_chunks": 0,
        }


def reset_collection(
    collection_name: Optional[str] = None,
    persist_directory: Optional[str] = None,
) -> bool:
    """
    Deletes all chunks in the specified collection for clean re-indexing.
    """
    try:
        vectorstore = get_vectorstore(collection_name, persist_directory)
        vectorstore.delete_collection()
        logger.info("ChromaDB collection reset.")
        return True
    except Exception as exc:
        logger.error(f"Failed to reset collection: {exc}")
        return False
