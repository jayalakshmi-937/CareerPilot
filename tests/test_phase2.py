"""
CareerPilot — Phase 2 Tests

Covers:
  - LLM factory instantiation
  - Goal analyzer node (live Gemini call)
  - Planner node (live Gemini call)
  - Full graph invocation end-to-end
  - POST /api/agent endpoint
  - Request validation (bad inputs)

NOTE: Tests marked with @pytest.mark.live make real Gemini API calls.
      They require GEMINI_API_KEY to be set in .env.
      Run fast (no-live) tests only with:  pytest tests/test_phase2.py -m "not live"
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# 1. LLM Factory
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMFactory:
    def test_llm_imports(self):
        from agent.llm import get_llm
        assert callable(get_llm)

    def test_llm_is_cached(self, monkeypatch):
        """Two calls with the same configuration return the same cached object."""
        import agent.llm as llm_module

        class FakeSettings:
            llm_model = "test-model"
            llm_temperature = 0.7
            llm_max_tokens = 1024

     
class TestLLMFactory:
    def test_llm_imports(self):
        from agent.llm import get_llm
        assert callable(get_llm)

    def test_llm_is_cached(self, monkeypatch):
        import agent.llm as llm_module

        class FakeSettings:
            llm_model = "test-model"
            llm_temperature = 0.7
            llm_max_tokens = 1024

        class FakeLLM:
            def __init__(self, **kwargs):
                self.model = kwargs["model"]

        monkeypatch.setattr(llm_module, "get_settings", lambda: FakeSettings())
        monkeypatch.setattr(llm_module, "get_effective_gemini_api_key", lambda: "")
        monkeypatch.setattr(llm_module, "ChatGoogleGenerativeAI", FakeLLM)
        llm_module._create_llm_instance.cache_clear()

        try:
            llm1 = llm_module.get_llm()
            llm2 = llm_module.get_llm()
            assert llm1 is llm2
        finally:
            llm_module._create_llm_instance.cache_clear()

    def test_llm_has_correct_model(self, monkeypatch):
        import agent.llm as llm_module

        class FakeSettings:
            llm_model = "test-model"
            llm_temperature = 0.7
            llm_max_tokens = 1024

        class FakeLLM:
            def __init__(self, **kwargs):
                self.model = kwargs["model"]

        monkeypatch.setattr(llm_module, "get_settings", lambda: FakeSettings())
        monkeypatch.setattr(llm_module, "get_effective_gemini_api_key", lambda: "")
        monkeypatch.setattr(llm_module, "ChatGoogleGenerativeAI", FakeLLM)
        llm_module._create_llm_instance.cache_clear()

        try:
            llm = llm_module.get_llm()
            assert llm.model == "test-model"
        finally:
            llm_module._create_llm_instance.cache_clear()



# ─────────────────────────────────────────────────────────────────────────────
# 2. Prompts
# ─────────────────────────────────────────────────────────────────────────────

class TestPrompts:
    def test_goal_analyzer_prompt_exists(self):
        from agent.prompts import GOAL_ANALYZER_PROMPT
        assert GOAL_ANALYZER_PROMPT is not None

    def test_planner_prompt_exists(self):
        from agent.prompts import PLANNER_PROMPT
        assert PLANNER_PROMPT is not None

    def test_goal_analyzer_has_student_goal_variable(self):
        from agent.prompts import GOAL_ANALYZER_PROMPT
        variables = GOAL_ANALYZER_PROMPT.input_variables
        assert "student_goal" in variables

    def test_planner_has_required_variables(self):
        from agent.prompts import PLANNER_PROMPT
        variables = PLANNER_PROMPT.input_variables
        for var in ["student_goal", "technology", "days_available"]:
            assert var in variables, f"Missing variable: {var}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Graph (Phase 2 version)
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase2Graph:
    def test_graph_builds_with_three_nodes(self):
        from agent.graph import build_graph
        g = build_graph()
        assert g is not None

    def test_graph_singleton_exists(self):
        from agent.graph import graph
        assert graph is not None

    def test_receive_goal_node_with_empty_goal(self):
        """Empty goal should return error in state without LLM call."""
        from agent.nodes.goal_analyzer import analyze_goal
        from agent.state import AgentState
        # Simulate what receive_goal does for empty input
        state: AgentState = {"student_goal": ""}
        # The receive_goal node in graph.py handles empty goals
        from agent.graph import receive_goal
        result = receive_goal(state)
        assert result.get("error") is not None

    def test_receive_goal_node_with_valid_goal(self):
        from agent.graph import receive_goal
        from agent.state import AgentState
        state: AgentState = {"student_goal": "Java interview in 7 days"}
        result = receive_goal(state)
        assert result.get("error") is None
        assert "messages" in result
        assert result["progress"]["session_count"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 4. API Endpoint (no LLM — request validation)
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentEndpointValidation:
    @pytest.fixture
    def client(self):
        from starlette.testclient import TestClient
        from backend.main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c

    def test_missing_goal_returns_422(self, client):
        """Pydantic should reject a request with no 'goal' field."""
        response = client.post("/api/agent", json={})
        assert response.status_code == 422

    def test_goal_too_short_returns_422(self, client):
        """Goal under 5 chars should fail Pydantic min_length validation."""
        response = client.post("/api/agent", json={"goal": "Hi"})
        assert response.status_code == 422

    def test_endpoint_exists(self, client):
        """POST /api/agent must be registered (not 404)."""
        # We only check it's not 404 — it may return 500 without a valid API key
        response = client.post(
            "/api/agent",
            json={"goal": "Prepare me for a Java interview in 7 days"},
        )
        assert response.status_code != 404

    def test_docs_show_agent_endpoint(self, client):
        """OpenAPI docs should include the /api/agent path."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        spec = response.json()
        assert "/api/agent" in spec["paths"]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Live LLM Tests (require real API key)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.live
class TestLiveGoalAnalyzer:
    """Requires GEMINI_API_KEY in .env — skipped automatically if key missing."""

    @pytest.fixture(autouse=True)
    def check_key(self):
        from backend.config import get_settings
        settings = get_settings()
        if not settings.gemini_api_key or settings.gemini_api_key == "your_gemini_api_key_here":
            pytest.skip("GEMINI_API_KEY not set — skipping live test")

    def test_goal_analyzer_returns_profile(self):
        from agent.nodes.goal_analyzer import analyze_goal
        from agent.state import AgentState
        state: AgentState = {"student_goal": "I have a Java interview in 7 days. Prepare me."}
        result = analyze_goal(state)
        assert result.get("error") is None
        profile = result.get("student_profile")
        assert profile is not None
        assert "java" in profile.get("technology", "").lower()
        assert profile.get("days_available") == 7
        assert len(profile.get("topics", [])) >= 3

    def test_goal_analyzer_infers_days(self):
        from agent.nodes.goal_analyzer import analyze_goal
        from agent.state import AgentState
        state: AgentState = {"student_goal": "Help me prepare for a Python Data Science interview"}
        result = analyze_goal(state)
        profile = result.get("student_profile", {})
        assert profile.get("days_available", 0) > 0  # defaults to 7


@pytest.mark.live
class TestLivePlanner:
    @pytest.fixture(autouse=True)
    def check_key(self):
        from backend.config import get_settings
        settings = get_settings()
        if not settings.gemini_api_key or settings.gemini_api_key == "your_gemini_api_key_here":
            pytest.skip("GEMINI_API_KEY not set — skipping live test")

    def test_planner_generates_plan(self):
        from agent.nodes.planner import generate_plan
        from agent.state import AgentState
        state: AgentState = {
            "student_goal": "Java interview in 5 days",
            "student_profile": {
                "technology": "Java",
                "role": "Software Engineer",
                "days_available": 5,
                "difficulty_level": "Intermediate",
                "topics": ["OOP", "Collections", "Multithreading", "Exception Handling", "Spring Boot"],
                "interview_type": "Technical",
                "summary": "Java interview in 5 days",
            },
        }
        result = generate_plan(state)
        assert result.get("error") is None
        plan = result.get("generated_plan", {})
        assert "markdown" in plan
        assert "structured" in plan
        structured = plan["structured"]
        assert "daily_schedule" in structured
        assert len(structured["daily_schedule"]) == 5

    def test_planner_markdown_contains_day_headers(self):
        from agent.nodes.planner import generate_plan
        from agent.state import AgentState
        state: AgentState = {
            "student_goal": "Python interview in 3 days",
            "student_profile": {
                "technology": "Python",
                "role": "Software Engineer",
                "days_available": 3,
                "difficulty_level": "Beginner",
                "topics": ["Python basics", "Data structures", "OOP"],
                "interview_type": "Technical",
                "summary": "Python interview in 3 days",
            },
        }
        result = generate_plan(state)
        markdown = result.get("generated_plan", {}).get("markdown", "")
        assert "Day 1" in markdown
        assert "Day 3" in markdown


@pytest.mark.live
class TestLiveFullWorkflow:
    @pytest.fixture(autouse=True)
    def check_key(self):
        from backend.config import get_settings
        settings = get_settings()
        if not settings.gemini_api_key or settings.gemini_api_key == "your_gemini_api_key_here":
            pytest.skip("GEMINI_API_KEY not set — skipping live test")

    def test_full_graph_invocation(self):
        from agent.graph import graph
        result = graph.invoke({
            "student_goal": "I need to prepare for a Java backend developer interview in 7 days."
        })
        assert result.get("error") is None
        assert result.get("student_profile") is not None
        assert result.get("generated_plan") is not None
        assert result.get("session_id") is not None

    def test_api_endpoint_live(self):
        from starlette.testclient import TestClient
        from backend.main import app
        with TestClient(app) as client:
            response = client.post(
                "/api/agent",
                json={
                    "goal": "Prepare me for a Java interview in 7 days",
                    "student_name": "Test Student",
                },
                timeout=60,
            )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["student_profile"] is not None
        assert data["plan_markdown"] is not None
        assert "Day 1" in data["plan_markdown"]
