"""
CareerPilot — Phase 8 Tests (Mock Interview Agent)

Covers:
  - Mock Interview schemas (InterviewStartRequest, TurnQuestion, TurnEvaluation, FinalSummary)
  - Interviewer Agent Node (Progressive question generation)
  - Evaluator Agent Node (Technical depth + Communication clarity scoring)
  - Hiring Committee Finalizer Node (Readiness score & recommendations synthesis)
  - LangGraph StateGraph compilation
  - Interview Service state coordinator
  - FastAPI Mock Interview endpoints (/api/interview/start, /api/interview/answer, /api/interview/{id}/report)
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from starlette.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.connection import Base, get_db
from interview.schemas import (
    InterviewAnswerRequest,
    InterviewFinalSummary,
    InterviewStartRequest,
    InterviewTurnEvaluation,
    InterviewTurnQuestion,
    TopicPerformance,
)
from interview.service import (
    get_interview_summary,
    start_interview_session,
    submit_interview_turn,
)
from interview.workflow import (
    InterviewState,
    build_interview_graph,
    evaluator_node,
    finalizer_node,
    interviewer_node,
)


# ── SQLite In-Memory Test Fixture ─────────────────────────────────────────────

@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(db_session):
    from backend.main import app

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.clear()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Schema Validation Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestInterviewSchemas:
    def test_start_request_validation(self):
        req = InterviewStartRequest(
            student_name="Aarav Sharma",
            target_role="Backend Engineer",
            technology="Java",
            total_questions=5,
            difficulty="Medium",
        )
        assert req.student_name == "Aarav Sharma"
        assert req.total_questions == 5

    def test_turn_evaluation_scores(self):
        eval_turn = InterviewTurnEvaluation(
            question_index=1,
            total_questions=5,
            topic="Memory Management",
            technical_score=8.5,
            communication_score=8.0,
            overall_score=8.3,
            verdict="Strong",
            feedback="Excellent explanation of garbage collection and generation heaps.",
            strengths=["Heap separation", "GC cycle explanation"],
            areas_for_improvement=[],
            is_finished=False,
            next_question=None,
        )
        assert eval_turn.technical_score == 8.5
        assert eval_turn.communication_score == 8.0
        assert eval_turn.verdict == "Strong"

    def test_final_summary_validation(self):
        summary = InterviewFinalSummary(
            session_id="test-session-123",
            student_name="Aarav Sharma",
            target_role="SDE-1",
            technology="Java",
            total_questions=5,
            technical_score=85.0,
            communication_score=80.0,
            overall_interview_score=83.0,
            interview_readiness_score=84.0,
            readiness_verdict="Hire",
            topic_wise_performance=[
                TopicPerformance(
                    topic="OOP",
                    average_technical_score=8.5,
                    average_communication_score=8.0,
                    verdict="Mastered",
                    observations="Strong design principles",
                )
            ],
            strong_areas=["OOP", "Memory"],
            weak_areas=["Concurrency"],
            recommendations=["Review race conditions"],
            interviewer_notes="Recommended for hire.",
            generated_at="2026-10-17 12:00:00 UTC",
        )
        assert summary.readiness_verdict == "Hire"
        assert summary.interview_readiness_score == 84.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. Workflow Node Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestInterviewWorkflowNodes:
    def test_interviewer_node_generates_turn(self):
        state: InterviewState = {
            "session_id": "sess-node-1",
            "student_name": "Priya Rao",
            "target_role": "Full Stack Engineer",
            "technology": "Java",
            "total_questions": 3,
            "difficulty": "Medium",
            "current_index": 0,
            "current_question": {},
            "student_answer": None,
            "last_evaluation": None,
            "history": [],
            "final_summary": None,
            "status": "in_progress",
            "error": None,
        }
        res = interviewer_node(state)
        assert res["current_index"] == 1
        assert "question_text" in res["current_question"]
        assert len(res["current_question"]["question_text"]) > 10
        assert res["status"] == "awaiting_answer"

    def test_evaluator_node_scores_answer(self):
        state: InterviewState = {
            "session_id": "sess-node-2",
            "student_name": "Priya Rao",
            "target_role": "SDE-1",
            "technology": "Java",
            "total_questions": 3,
            "difficulty": "Medium",
            "current_index": 1,
            "current_question": {
                "topic": "Collections & HashMaps",
                "difficulty": "Medium",
                "question_text": "Explain how HashMap handles key collisions in Java.",
            },
            "student_answer": (
                "In Java 8, HashMap uses separate chaining with linked lists. "
                "When a bucket reaches TREEIFY_THRESHOLD (8 items) and capacity is at least 64, "
                "it converts the list into a Red-Black tree to maintain O(log N) lookup instead of O(N)."
            ),
            "last_evaluation": None,
            "history": [],
            "final_summary": None,
            "status": "in_progress",
            "error": None,
        }
        res = evaluator_node(state)
        assert res["status"] == "evaluated"
        assert res["last_evaluation"] is not None
        assert 0.0 <= res["last_evaluation"]["technical_score"] <= 10.0
        assert 0.0 <= res["last_evaluation"]["communication_score"] <= 10.0
        assert len(res["history"]) == 1

    def test_evaluator_node_handles_empty_answer_gracefully(self):
        state: InterviewState = {
            "session_id": "sess-node-3",
            "student_name": "Priya Rao",
            "target_role": "SDE-1",
            "technology": "Java",
            "total_questions": 3,
            "difficulty": "Medium",
            "current_index": 1,
            "current_question": {
                "topic": "OOP",
                "difficulty": "Medium",
                "question_text": "Explain Polymorphism.",
            },
            "student_answer": "",
            "last_evaluation": None,
            "history": [],
            "final_summary": None,
            "status": "in_progress",
            "error": None,
        }
        res = evaluator_node(state)
        assert res["last_evaluation"]["technical_score"] < 5.0
        assert res["last_evaluation"]["verdict"] == "Needs Improvement"

    def test_finalizer_node_synthesizes_scorecard(self):
        state: InterviewState = {
            "session_id": "sess-node-4",
            "student_name": "Rohan Verma",
            "target_role": "Backend Engineer",
            "technology": "Java",
            "total_questions": 2,
            "difficulty": "Medium",
            "current_index": 2,
            "current_question": {},
            "student_answer": None,
            "last_evaluation": None,
            "history": [
                {
                    "question_index": 1,
                    "topic": "Core Java",
                    "difficulty": "Medium",
                    "question_text": "What is JVM memory model?",
                    "student_answer": "JVM divides memory into Heap, Stack, Method Area, PC Register, and Native Stack.",
                    "technical_score": 8.5,
                    "communication_score": 8.0,
                    "overall_score": 8.3,
                    "verdict": "Strong",
                    "feedback": "Great overview.",
                    "strengths": ["Structured answer"],
                    "areas_for_improvement": [],
                },
                {
                    "question_index": 2,
                    "topic": "Concurrency",
                    "difficulty": "Medium",
                    "question_text": "Explain volatile keyword.",
                    "student_answer": "Volatile guarantees memory visibility across threads but not atomicity.",
                    "technical_score": 9.0,
                    "communication_score": 8.5,
                    "overall_score": 8.8,
                    "verdict": "Strong",
                    "feedback": "Clear explanation of memory visibility.",
                    "strengths": ["Understands atomicity distinction"],
                    "areas_for_improvement": [],
                },
            ],
            "final_summary": None,
            "status": "evaluated",
            "error": None,
        }
        res = finalizer_node(state)
        assert res["status"] == "finished"
        assert res["final_summary"] is not None
        assert res["final_summary"]["technical_score"] >= 70.0
        assert res["final_summary"]["communication_score"] >= 70.0
        assert res["final_summary"]["readiness_verdict"] in ("Strong Hire", "Hire", "Leaning Hire", "No Hire")

    def test_langgraph_interview_graph_compiles(self):
        graph = build_interview_graph()
        assert graph is not None


# ─────────────────────────────────────────────────────────────────────────────
# 3. Service Layer Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestInterviewService:
    def test_start_and_complete_short_interview_session(self, db_session):
        # 1. Start 2-question interview
        turn1 = start_interview_session(
            student_name="Devika Menon",
            target_role="Java Developer",
            technology="Java",
            total_questions=2,
            difficulty="Medium",
            db=db_session,
        )
        assert turn1.question_index == 1
        assert turn1.total_questions == 2
        assert len(turn1.question_text) > 10

        # 2. Answer Question 1
        eval1, final1 = submit_interview_turn(
            session_id=turn1.session_id,
            question_index=1,
            student_answer="Encapsulation is bundling data and methods inside a single class and restricting direct access using private fields.",
            db=db_session,
        )
        assert eval1.is_finished is False
        assert eval1.next_question is not None
        assert eval1.next_question.question_index == 2
        assert final1 is None

        # 3. Answer Question 2 (Final Question)
        eval2, final2 = submit_interview_turn(
            session_id=turn1.session_id,
            question_index=2,
            student_answer="ConcurrentHashMap achieves thread-safety using fine-grained bucket synchronization (synchronized on head node in Java 8) and CAS operations.",
            db=db_session,
        )
        assert eval2.is_finished is True
        assert eval2.next_question is None
        assert final2 is not None
        assert 0.0 <= final2.technical_score <= 100.0
        assert 0.0 <= final2.communication_score <= 100.0
        assert len(final2.topic_wise_performance) >= 1
        assert len(final2.interviewer_notes) > 20

        # 4. Retrieve summary via service
        cached_summary = get_interview_summary(turn1.session_id)
        assert cached_summary is not None
        assert cached_summary.session_id == turn1.session_id


# ─────────────────────────────────────────────────────────────────────────────
# 4. FastAPI REST Endpoints Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestInterviewAPIEndpoints:
    def test_start_interview_endpoint(self, client):
        resp = client.post(
            "/api/interview/start",
            json={
                "student_name": "Sahil Verma",
                "target_role": "Software Engineer",
                "technology": "Java",
                "total_questions": 3,
                "difficulty": "Medium",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "session_id" in data
        assert data["question_index"] == 1
        assert data["total_questions"] == 3
        assert "question_text" in data

    def test_submit_interview_answer_endpoint(self, client):
        # Start session
        start_resp = client.post(
            "/api/interview/start",
            json={
                "student_name": "Sahil Verma",
                "target_role": "Software Engineer",
                "technology": "Java",
                "total_questions": 3,
                "difficulty": "Medium",
            },
        )
        session_id = start_resp.json()["session_id"]

        # Submit answer
        ans_resp = client.post(
            "/api/interview/answer",
            json={
                "session_id": session_id,
                "question_index": 1,
                "student_answer": "Java uses bytecode executed on the JVM, providing cross-platform portability through the Write Once Run Anywhere model.",
            },
        )
        assert ans_resp.status_code == 200
        data = ans_resp.json()
        assert "evaluation" in data
        assert data["is_finished"] is False
        assert data["evaluation"]["technical_score"] >= 0.0

    def test_invalid_session_returns_404(self, client):
        resp = client.post(
            "/api/interview/answer",
            json={
                "session_id": "non-existent-session-xyz",
                "question_index": 1,
                "student_answer": "Some answer",
            },
        )
        assert resp.status_code == 404
