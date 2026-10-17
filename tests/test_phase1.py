"""
CareerPilot — Phase 1 Tests

Covers:
  - Configuration loading
  - Agent state schema validity
  - LangGraph graph compilation (no LLM calls)
  - FastAPI health endpoint
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ── Ensure project root is importable ─────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────────────────────────────────────

class TestConfiguration:
    def test_settings_load(self):
        """Settings object should load without raising."""
        from backend.config import get_settings
        settings = get_settings()
        assert settings is not None

    def test_default_llm_model(self):
        from backend.config import get_settings
        settings = get_settings()
        assert "gemini" in settings.llm_model.lower()

    def test_default_env(self):
        from backend.config import get_settings
        settings = get_settings()
        # Should default to development if .env is absent
        assert settings.app_env in ("development", "production", "test")

    def test_documents_dir_is_path(self):
        from backend.config import get_settings
        settings = get_settings()
        assert isinstance(settings.documents_dir, Path)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Agent State
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentState:
    def test_import(self):
        from agent.state import AgentState
        assert AgentState is not None

    def test_empty_state_is_valid(self):
        """An empty dict is a valid AgentState (all fields are optional)."""
        from agent.state import AgentState
        state: AgentState = {}
        assert isinstance(state, dict)

    def test_partial_state(self):
        from agent.state import AgentState
        state: AgentState = {
            "student_goal": "Prepare me for a Java interview",
            "weak_topics": ["inheritance", "generics"],
        }
        assert state["student_goal"].startswith("Prepare")
        assert len(state["weak_topics"]) == 2

    def test_all_expected_keys_present_in_annotations(self):
        from agent.state import AgentState
        expected_keys = {
            "student_goal", "student_name", "student_profile",
            "current_topic", "messages", "generated_plan",
            "current_question", "student_answer", "evaluation",
            "weak_topics", "progress", "retrieved_context",
            "next_action", "error", "session_id",
        }
        annotations = AgentState.__annotations__
        missing = expected_keys - set(annotations.keys())
        assert not missing, f"Missing keys in AgentState: {missing}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. LangGraph Graph (updated for Phase 2)
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentGraph:
    def test_graph_compiles(self):
        """Graph should compile without errors (no LLM calls needed)."""
        from agent.graph import build_graph
        compiled = build_graph()
        assert compiled is not None

    def test_graph_singleton(self):
        from agent.graph import graph
        assert graph is not None

    def test_graph_error_on_empty_goal(self):
        """
        Empty goal short-circuits at receive_goal (no LLM call).
        Final state must contain an error message.
        """
        from agent.graph import graph
        result = graph.invoke({"student_goal": ""})
        # receive_goal detects empty goal → sets error → conditional edge → END
        assert result.get("error") is not None

    def test_graph_preserves_goal(self):
        """
        Non-empty goal should survive through receive_goal into state.
        (The graph may fail at LLM nodes if no API key, but goal must be kept.)
        """
        from agent.graph import graph
        goal = "Prepare me for a Python interview in 3 days"
        result = graph.invoke({"student_goal": goal})
        # The goal must be echoed back in state regardless of LLM outcome
        assert result.get("student_goal") == goal


# ─────────────────────────────────────────────────────────────────────────────
# 4. FastAPI Endpoints
# ─────────────────────────────────────────────────────────────────────────────

class TestFastAPIEndpoints:
    """
    Uses Starlette's TestClient — the official synchronous ASGI test client
    for FastAPI. Avoids httpx.ASGITransport which became async-only in newer
    httpx releases.
    """

    @pytest.fixture
    def client(self):
        from starlette.testclient import TestClient
        from backend.main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["app"] == "CareerPilot"

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "CareerPilot" in data["message"]

    def test_health_contains_version(self, client):
        response = client.get("/health")
        data = response.json()
        assert "version" in data

    def test_docs_accessible(self, client):
        response = client.get("/docs")
        assert response.status_code == 200
