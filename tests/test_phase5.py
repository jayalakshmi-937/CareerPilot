"""
CareerPilot — Phase 5 Tests (Agentic Learning Loop)

Covers:
  - Adaptive topic selection (weak topic prioritization vs curriculum advancement)
  - Answer evaluation logic and scoring
  - Weakness detection and progress metrics computation
  - Adaptive decision-making (repeat_weak_topic vs next_topic vs finish)
  - End-to-end evaluation API endpoint (/api/agent/evaluate)
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from agent.state import AgentState

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Topic Selector Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTopicSelector:
    def test_prioritizes_weak_topic(self):
        """When weak topics exist, the selector must prioritize them for remediation."""
        from agent.nodes.topic_selector import select_topic

        state: AgentState = {
            "student_profile": {"topics": ["OOP", "Collections", "Multithreading"]},
            "weak_topics": ["Collections"],
            "mastered_topics": ["OOP"],
        }
        res = select_topic(state)
        assert res["current_topic"] == "Collections"
        assert res["next_action"] == "repeat_weak_topic"

    def test_advances_to_next_unmastered_topic(self):
        """When no weak topics exist, it advances to the first unmastered topic."""
        from agent.nodes.topic_selector import select_topic

        state: AgentState = {
            "student_profile": {"topics": ["OOP", "Collections", "Multithreading"]},
            "weak_topics": [],
            "mastered_topics": ["OOP"],
        }
        res = select_topic(state)
        assert res["current_topic"] == "Collections"
        assert res["next_action"] == "continue"

    def test_finishes_when_all_topics_mastered(self):
        """When all topics are mastered, it sets next_action to finish."""
        from agent.nodes.topic_selector import select_topic

        state: AgentState = {
            "student_profile": {"topics": ["OOP", "Collections"]},
            "weak_topics": [],
            "mastered_topics": ["OOP", "Collections"],
        }
        res = select_topic(state)
        assert res["next_action"] == "finish"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Weakness Detector & Progress Updater Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestWeaknessDetector:
    def test_detects_and_flags_weak_topic(self):
        """A score < 7.0 must flag the topic as weak and remove from mastered."""
        from agent.nodes.weakness_detector import detect_weakness_and_update_progress

        state: AgentState = {
            "current_topic": "Java Inheritance",
            "evaluation": {"score": 5.0, "verdict": "Weak", "remediation_advice": "Review super() calls."},
            "weak_topics": [],
            "mastered_topics": ["Java Inheritance"],
            "student_profile": {"topics": ["Java Inheritance", "Collections"]},
        }
        res = detect_weakness_and_update_progress(state)
        assert "Java Inheritance" in res["weak_topics"]
        assert "Java Inheritance" not in res["mastered_topics"]
        assert res["topic_scores"]["Java Inheritance"] == [5.0]
        assert "Weakness identified" in res["remediation_notes"]

    def test_resolves_weak_topic_on_mastery(self):
        """A score >= 7.0 on a previously weak topic must clear it from weak_topics and add to mastered."""
        from agent.nodes.weakness_detector import detect_weakness_and_update_progress

        state: AgentState = {
            "current_topic": "Java Inheritance",
            "evaluation": {"score": 8.5, "verdict": "Mastered"},
            "weak_topics": ["Java Inheritance"],
            "mastered_topics": [],
            "student_profile": {"topics": ["Java Inheritance", "Collections"]},
        }
        res = detect_weakness_and_update_progress(state)
        assert "Java Inheritance" not in res["weak_topics"]
        assert "Java Inheritance" in res["mastered_topics"]
        assert res["progress"]["overall_progress_percent"] == 50


# ─────────────────────────────────────────────────────────────────────────────
# 3. Adaptive Decider Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAdaptiveDecider:
    def test_decides_repeat_weak_topic_on_low_score(self):
        from agent.nodes.adaptive_decider import decide_next_action

        state: AgentState = {
            "current_topic": "Multithreading",
            "evaluation": {"score": 4.5, "identified_weakness": "Deadlocks and Synchronized"},
            "student_profile": {"topics": ["OOP", "Multithreading", "Spring"]},
        }
        res = decide_next_action(state)
        assert res["next_action"] == "repeat_weak_topic"
        assert "Remediate weakness in 'Multithreading'" in res["recommended_next_task"]

    def test_decides_advance_topic_on_high_score(self):
        from agent.nodes.adaptive_decider import decide_next_action

        state: AgentState = {
            "current_topic": "OOP",
            "evaluation": {"score": 9.0},
            "mastered_topics": ["OOP"],
            "student_profile": {"topics": ["OOP", "Collections", "Spring"]},
        }
        res = decide_next_action(state)
        assert res["next_action"] == "next_topic"
        assert "Collections" in res["recommended_next_task"]

    def test_decides_finish_when_curriculum_complete(self):
        from agent.nodes.adaptive_decider import decide_next_action

        state: AgentState = {
            "current_topic": "Spring",
            "evaluation": {"score": 8.0},
            "mastered_topics": ["OOP", "Spring"],
            "student_profile": {"topics": ["OOP", "Spring"]},
        }
        res = decide_next_action(state)
        assert res["next_action"] == "finish"
        assert "Congratulations" in res["recommended_next_task"]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Evaluator Node Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluatorNode:
    def test_evaluator_handles_empty_answer_gracefully(self):
        from agent.nodes.evaluator import evaluate_student_answer

        state: AgentState = {
            "student_answer": "",
            "current_topic": "Java Exceptions",
            "current_question": "What is try-with-resources?",
        }
        res = evaluate_student_answer(state)
        assert res["evaluation"]["score"] == 0.0
        assert res["evaluation"]["is_passed"] is False
        assert "No answer" in res["evaluation"]["feedback"]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Evaluate API Endpoint Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluateAPIEndpoint:
    @pytest.fixture
    def client(self):
        from starlette.testclient import TestClient
        from backend.main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c

    def test_evaluate_endpoint_validation_empty_body(self, client):
        response = client.post("/api/agent/evaluate", json={})
        assert response.status_code == 422

    def test_evaluate_endpoint_mock_cycle(self, client):
        mock_return_state = {
            "session_id": "test-session-123",
            "student_profile": {"name": "Arjun", "technology": "Java", "topics": ["Collections", "Spring"]},
            "current_topic": "Collections",
            "current_question": "What is ConcurrentHashMap?",
            "question_details": {},
            "evaluation": {
                "score": 8.0,
                "is_passed": True,
                "verdict": "Satisfactory",
                "feedback": "Good answer explaining HashMap treeification.",
                "strengths": ["Correct threshold stated"],
                "areas_for_improvement": [],
                "ideal_answer": "Bucket size 8 triggers Red-Black Tree.",
            },
            "weak_topics": [],
            "mastered_topics": ["Collections"],
            "remediation_notes": None,
            "recommended_next_task": "Continue to Spring fundamentals.",
            "progress": {"completed_topics": ["Collections"], "overall_progress_percent": 50},
            "next_action": "next_topic",
            "messages": [],
            "error": None,
        }

        with patch("backend.routers.agent.graph.invoke", return_value=mock_return_state):
            payload = {
                "student_answer": "HashMap converts linked list to balanced tree when size reaches 8.",
                "current_question": "Explain HashMap collision handling.",
                "current_topic": "Collections",
                "student_profile": {"topics": ["Collections", "Spring"]},
                "weak_topics": [],
                "mastered_topics": [],
            }
            response = client.post("/api/agent/evaluate", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "Collections" in data["mastered_topics"]
            assert data["next_action"] == "next_topic"
            assert data["evaluation"]["score"] == 8.0

