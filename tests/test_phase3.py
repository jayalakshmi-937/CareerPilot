"""
CareerPilot — Phase 3 Tests

Covers:
  - Frontend app importability and syntax check
  - Frontend backend health check function with mocking
  - Frontend API call formatting (name, technology, days, goal)
  - Backend AgentRequest schema with new Phase 3 fields (target_technology, available_days)
  - Backend AgentResponse schema with new Phase 3 fields (recommended_next_task, progress)
  - Mocked end-to-end API response validation
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Frontend Module & Helper Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFrontendHelpers:
    def test_frontend_app_imports(self):
        """frontend.app module should be importable without syntax or top-level errors."""
        import frontend.app as app
        assert app is not None
        assert hasattr(app, "check_backend_health")
        assert hasattr(app, "call_agent_api")

    def test_check_backend_health_online(self):
        """When backend returns 200, check_backend_health returns (True, data)."""
        import frontend.app as app
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok", "app": "CareerPilot"}

        with patch("requests.get", return_value=mock_response):
            is_ok, data = app.check_backend_health()
            assert is_ok is True
            assert data["status"] == "ok"

    def test_check_backend_health_offline(self):
        """When backend raises connection error, check_backend_health returns (False, None)."""
        import frontend.app as app
        with patch("requests.get", side_effect=Exception("Connection refused")):
            is_ok, data = app.check_backend_health()
            assert is_ok is False
            assert data is None

    def test_call_agent_api_payload_structure(self):
        """call_agent_api formats payload correctly and handles 200 response."""
        import frontend.app as app
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "current_topic": "Java OOP",
            "recommended_next_task": "Practice inheritance problems",
            "progress": {"completed_topics": []},
            "plan_markdown": "# Plan",
        }

        with patch("requests.post", return_value=mock_response) as mock_post:
            success, data, err = app.call_agent_api(
                name="Arjun",
                goal="Prepare for Java",
                technology="Java",
                days=7,
            )
            assert success is True
            assert data["current_topic"] == "Java OOP"
            assert err is None

            # Verify call payload
            call_kwargs = mock_post.call_args[1]
            json_payload = call_kwargs["json"]
            assert json_payload["student_name"] == "Arjun"
            assert json_payload["target_technology"] == "Java"
            assert json_payload["available_days"] == 7
            assert "Prepare for Java" in json_payload["goal"]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Phase 3 Backend Schema Validations
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase3BackendSchemas:
    def test_agent_request_accepts_tech_and_days(self):
        from backend.routers.agent import AgentRequest
        req = AgentRequest(
            goal="I have a Python interview in 5 days. Prepare me.",
            student_name="Priya",
            target_technology="Python",
            available_days=5,
        )
        assert req.target_technology == "Python"
        assert req.available_days == 5
        assert req.student_name == "Priya"

    def test_agent_request_defaults(self):
        from backend.routers.agent import AgentRequest
        req = AgentRequest(goal="Prepare me for a placement test")
        assert req.target_technology is None
        assert req.available_days is None
        assert req.student_name is None

    def test_agent_response_contains_phase3_fields(self):
        from backend.routers.agent import AgentResponse
        resp = AgentResponse(
            success=True,
            current_topic="Collections Framework",
            recommended_next_task="Implement a custom ArrayList and HashMap in Java",
            progress={"completed_topics": [], "session_count": 1},
        )
        assert resp.current_topic == "Collections Framework"
        assert resp.recommended_next_task == "Implement a custom ArrayList and HashMap in Java"
        assert resp.progress["session_count"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 3. Agent State Schema Extension
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentStatePhase3:
    def test_agent_state_has_recommended_next_task(self):
        from agent.state import AgentState
        state: AgentState = {
            "student_goal": "Java in 7 days",
            "recommended_next_task": "Study Day 1: OOP concepts",
        }
        assert state["recommended_next_task"] == "Study Day 1: OOP concepts"
        assert "recommended_next_task" in AgentState.__annotations__
