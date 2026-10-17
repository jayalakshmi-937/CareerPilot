"""
CareerPilot — Phase 4 Tests (RAG with ChromaDB)

Covers:
  - Document loading (.txt, .md) and error handling
  - Text chunking (RecursiveCharacterTextSplitter) and metadata enrichment
  - ChromaDB vector store initialization and stats
  - Semantic retriever service functions
  - LangGraph retrieve_study_material node integration
  - Documents API endpoints (/status, /query, /ingest)
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Document Ingestion & Chunking Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDocumentIngestion:
    def test_load_single_text_file(self, tmp_path):
        from rag.ingestion import load_single_file

        sample_file = tmp_path / "sample.txt"
        sample_file.write_text("Java is an object-oriented language.", encoding="utf-8")

        docs = load_single_file(sample_file)
        assert len(docs) == 1
        assert "Java is an object-oriented" in docs[0].page_content
        assert docs[0].metadata["source"] == "sample.txt"

    def test_load_unsupported_format_returns_empty(self, tmp_path):
        from rag.ingestion import load_single_file

        exe_file = tmp_path / "binary.bin"
        exe_file.write_bytes(b"\x00\x01\x02")

        docs = load_single_file(exe_file)
        assert docs == []

    def test_chunk_documents_splits_and_enriches(self):
        from rag.ingestion import chunk_documents

        long_text = "Java Collections Framework.\n\n" + ("HashMap stores key-value pairs. " * 50)
        doc = Document(page_content=long_text, metadata={"source": "test.txt"})

        chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=50)
        assert len(chunks) > 1
        # Verify metadata enrichment
        assert "chunk_index" in chunks[0].metadata
        assert "total_chunks" in chunks[0].metadata
        assert "ingested_at" in chunks[0].metadata
        assert chunks[0].metadata["source"] == "test.txt"

    def test_load_and_chunk_sample_guide_exists(self):
        """Verifies that the java_placement_guide.txt in data/documents is discoverable and chunkable."""
        from rag.ingestion import load_and_chunk_directory
        from backend.config import get_settings

        settings = get_settings()
        chunks = load_and_chunk_directory(settings.documents_dir)
        assert len(chunks) > 0
        assert any("java" in c.page_content.lower() for c in chunks)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Vectorstore & Retrieval Logic Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestVectorStoreAndRetriever:
    def test_get_collection_stats_structure(self):
        from rag.vectorstore import get_collection_stats

        stats = get_collection_stats()
        assert isinstance(stats, dict)
        assert "status" in stats
        assert "total_chunks" in stats

    def test_retrieve_context_string_formatting(self):
        from rag.retriever import retrieve_context_string

        mock_docs = [
            Document(page_content="Inheritance is an OOP pillar.", metadata={"source": "guide1.txt"}),
            Document(page_content="Polymorphism supports dynamic dispatch.", metadata={"source": "guide2.txt"}),
        ]

        with patch("rag.retriever.retrieve_relevant_documents", return_value=mock_docs):
            context = retrieve_context_string("OOP pillars")
            assert "--- Document Source 1: guide1.txt ---" in context
            assert "Inheritance is an OOP pillar." in context
            assert "--- Document Source 2: guide2.txt ---" in context

    def test_retrieve_empty_query_returns_empty_list(self):
        from rag.retriever import retrieve_relevant_documents

        docs = retrieve_relevant_documents("   ")
        assert docs == []


# ─────────────────────────────────────────────────────────────────────────────
# 3. LangGraph Retriever Node Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRetrieverNode:
    def test_retriever_node_populates_state(self):
        from agent.nodes.retriever import retrieve_study_material
        from agent.state import AgentState

        mock_context = "--- Document Source 1: java.txt ---\nJava is thread-safe."
        with patch("agent.nodes.retriever.retrieve_context_string", return_value=mock_context):
            state: AgentState = {
                "student_goal": "Prepare for Java",
                "current_topic": "Java Concurrency",
            }
            new_state = retrieve_study_material(state)
            assert new_state["retrieved_context"] == mock_context
            assert any("Retrieved study material" in m["content"] for m in new_state.get("messages", []))

    def test_retriever_node_skips_on_upstream_error(self):
        from agent.nodes.retriever import retrieve_study_material
        from agent.state import AgentState

        state: AgentState = {"error": "Invalid goal"}
        new_state = retrieve_study_material(state)
        # Should return state untouched
        assert new_state.get("retrieved_context") is None or new_state.get("retrieved_context") == ""


# ─────────────────────────────────────────────────────────────────────────────
# 4. Documents API Endpoints Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDocumentsAPIEndpoints:
    @pytest.fixture
    def client(self):
        from starlette.testclient import TestClient
        from backend.main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c

    def test_documents_status_endpoint(self, client):
        response = client.get("/api/documents/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "total_chunks_in_chroma" in data
        assert "documents_on_disk" in data
        assert "java_placement_guide.txt" in data["documents_on_disk"]

    def test_query_endpoint_validation(self, client):
        # Query too short should fail Pydantic validation
        response = client.post("/api/documents/query", json={"query": "a"})
        assert response.status_code == 422
