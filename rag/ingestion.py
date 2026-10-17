"""
CareerPilot — Document Ingestion & Chunking (Phase 4)

Responsible for:
  1. Discovering documents in data/documents/ (or custom directory).
  2. Loading common document types (.txt, .md, .pdf, .docx).
  3. Splitting documents into semantic chunks with overlap.
  4. Attaching metadata (source, chunk_id, total_chunks, timestamp).
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from backend.config import get_settings


def load_single_file(file_path: Path) -> List[Document]:
    """
    Load a single document file into LangChain Document objects.
    Supports .txt, .md, .pdf, and .docx.
    """
    suffix = file_path.suffix.lower()
    docs: List[Document] = []

    try:
        if suffix in [".txt", ".md", ".markdown"]:
            text = file_path.read_text(encoding="utf-8", errors="replace")
            docs = [
                Document(
                    page_content=text,
                    metadata={"source": file_path.name, "file_path": str(file_path)},
                )
            ]

        elif suffix == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()

        elif suffix in [".docx", ".doc"]:
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(str(file_path))
            docs = loader.load()

        else:
            logger.warning(f"Unsupported file format skipped: {file_path.name}")
            return []

        logger.info(f"Loaded '{file_path.name}' ({len(docs)} document pages/sections)")
        return docs

    except Exception as exc:
        logger.error(f"Failed to load file '{file_path.name}': {exc}")
        return []


def chunk_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """
    Split loaded documents into overlapping chunks using RecursiveCharacterTextSplitter.
    Enriches each chunk with chunk_id and ingestion timestamp metadata.
    """
    settings = get_settings()
    c_size = chunk_size or settings.chunk_size
    c_overlap = chunk_overlap or settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=c_size,
        chunk_overlap=c_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )

    raw_chunks = splitter.split_documents(documents)
    enriched_chunks: List[Document] = []
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    for idx, chunk in enumerate(raw_chunks):
        metadata = dict(chunk.metadata)
        metadata["chunk_index"] = idx
        metadata["total_chunks"] = len(raw_chunks)
        metadata["ingested_at"] = now_iso
        enriched_chunks.append(
            Document(page_content=chunk.page_content, metadata=metadata)
        )

    logger.info(f"Chunked {len(documents)} source docs into {len(enriched_chunks)} chunks (size={c_size}, overlap={c_overlap})")
    return enriched_chunks


def load_and_chunk_directory(
    directory_path: Optional[Path] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """
    Scans the given directory (defaults to settings.documents_dir),
    loads all supported files, and returns chunked documents.
    """
    settings = get_settings()
    target_dir = directory_path or settings.documents_dir
    target_dir = Path(target_dir)

    if not target_dir.exists():
        logger.warning(f"Documents directory does not exist: {target_dir}")
        target_dir.mkdir(parents=True, exist_ok=True)
        return []

    supported_extensions = {".txt", ".md", ".pdf", ".docx"}
    all_files = [
        f for f in target_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in supported_extensions
    ]

    logger.info(f"Discovered {len(all_files)} documents in '{target_dir}'")
    all_docs: List[Document] = []

    for file_path in all_files:
        docs = load_single_file(file_path)
        all_docs.extend(docs)

    if not all_docs:
        logger.info(f"No valid documents found to chunk in {target_dir}")
        return []

    return chunk_documents(all_docs, chunk_size, chunk_overlap)
