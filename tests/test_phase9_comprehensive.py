"""
CareerPilot — Phase 9 Comprehensive System Test Suite

Full verification across:
  1. API Surface & Robust Error Handling (HTTP 200, 404, 422)
  2. LangGraph State Machine & Adaptive Decider Nodes
  3. Strict Score Boundary Decisions (6.9 vs 7.0)
  4. Database Relational Integrity & Student Memory Aggregation
  5. Placement Preparation Services (MCQ, Coding, Readiness Scorecard)
  6. Mock Interview State Lifecycle
  7. Document Chunking & Ingestion Safety
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from starlette.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.nodes.adaptive_decider import decide_next_action
from agent.nodes.evaluator import evaluate_student_answer
from agent.nodes.topic_selector import select_topic
from agent.nodes.weakness_detector import detect_weakness_and_update_progress
from agent.state import AgentState
from database.models import Answer, Evaluation, Goal, Progress, Question, Student, Topic
from database.repository import (
    get_or_create_goal,
    get_or_create_student,
    get_student_memory,
    record_answer_and_evaluation,
    record_question,
    sync_goal_topics,
    update_progress,
)
from interview.schemas import InterviewStartRequest
from interview.service import start_interview_session
from placement.generator import (
    generate_readiness_report,
    generate_topic_question,
)
from placement.schemas import (
    CodingDetails,
    CodingTestCase,
    MCQDetails,
    MCQOption,
    ReadinessReport,
    TopicPracticeResponse,
)
from rag.ingestion import chunk_documents
from tests.sample_data import SAMPLE_ANSWERS, SAMPLE_DOCUMENTS, SAMPLE_STUDENTS


# ─────────────────────────────────────────────────────────────────────────────
# 1. API Surface & Robustness (FastAPI Endpoints)
# ─────────────────────────────────────────────────────────────────────────────

class TestAPISurfaceAndErrorHandling:
    def test_health_check_endpoint(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_root_endpoint(self, client: TestClient):
        response = client.get("/")
        assert response.status_code == 200
        assert "CareerPilot" in response.json()["message"]

    def test_student_lifecycle_endpoints(self, client: TestClient, db_session):
        # Create student in DB
        student = get_or_create_student(
            db_session,
            name="Devika Sen",
            email="devika.sen@example.com",
            target_technology="Python",
            available_days=10,
        )
        assert student.id is not None
        student_id = student.id

        # List students
        res_list = client.get("/api/students")
        assert res_list.status_code == 200
        students = res_list.json()
        assert any(s["id"] == student_id for s in students)

        # Get individual student
        res_get = client.get(f"/api/students/{student_id}")
        assert res_get.status_code == 200
        assert res_get.json()["student"]["email"] == "devika.sen@example.com"

        # Get history
        res_hist = client.get(f"/api/students/{student_id}/history")
        assert res_hist.status_code == 200
        assert isinstance(res_hist.json(), list)

        # Get progress
        res_prog = client.get(f"/api/students/{student_id}/progress")
        assert res_prog.status_code == 200
        assert "mastered_topics" in res_prog.json()

    def test_nonexistent_student_returns_404(self, client: TestClient):
        res_get = client.get("/api/students/999999")
        assert res_get.status_code == 404

        res_hist = client.get("/api/students/999999/history")
        assert res_hist.status_code == 404

    def test_placement_mcq_check_endpoint(self, client: TestClient):
        # Correct answer
        res_correct = client.post(
            "/api/placement/mcq/check",
            json={
                "student_name": "Devika Sen",
                "topic": "Algorithms",
                "question_text": "What is the time complexity of binary search?",
                "correct_option": "B",
                "selected_option": "B",
                "explanation": "Binary search is O(log N).",
            },
        )
        assert res_correct.status_code == 200
        data_correct = res_correct.json()
        assert data_correct["is_correct"] is True
        assert data_correct["score"] == 10.0

        # Incorrect answer
        res_wrong = client.post(
            "/api/placement/mcq/check",
            json={
                "student_name": "Devika Sen",
                "topic": "Algorithms",
                "question_text": "What is the time complexity of binary search?",
                "correct_option": "B",
                "selected_option": "A",
                "explanation": "Binary search is O(log N).",
            },
        )
        assert res_wrong.status_code == 200
        data_wrong = res_wrong.json()
        assert data_wrong["is_correct"] is False
        assert data_wrong["score"] < 5.0

    def test_placement_mcq_validation_error_422(self, client: TestClient):
        # Missing selected_option
        bad_payload = {
            "topic": "Algorithms",
            "question_text": "Sample Question",
            "correct_option": "A",
        }
        response = client.post("/api/placement/mcq/check", json=bad_payload)
        assert response.status_code == 422

    def test_mock_interview_nonexistent_session_404(self, client: TestClient):
        res_answer = client.post(
            "/api/interview/answer",
            json={"session_id": "nonexistent-uuid", "question_index": 0, "student_answer": "Java is object-oriented"},
        )
        assert res_answer.status_code == 404

        res_report = client.get("/api/interview/nonexistent-uuid/report")
        assert res_report.status_code == 404

    def test_documents_status_endpoint(self, client: TestClient):
        response = client.get("/api/documents/status")
        assert response.status_code == 200
        assert "status" in response.json()
        assert "total_chunks_in_chroma" in response.json()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Adaptive Decider Logic & Exact Boundary Testing
# ─────────────────────────────────────────────────────────────────────────────

class TestAdaptiveDeciderLogic:
    def test_score_below_7_repeats_weak_topic(self):
        """Strict boundary: score 6.9 must trigger repeat_weak_topic."""
        state: AgentState = {
            "current_topic": "OOP",
            "evaluation": {"score": 6.9, "identified_weakness": "Polymorphism vs Encapsulation"},
            "student_profile": {"topics": ["OOP", "Collections", "Concurrency"]},
            "weak_topics": ["OOP"],
            "mastered_topics": [],
        }
        updated = decide_next_action(state)
        assert updated["next_action"] == "repeat_weak_topic"
        assert "Remediate weakness in 'OOP'" in updated["recommended_next_task"]

    def test_score_at_7_advances_to_next_topic(self):
        """Strict boundary: score 7.0 must advance to next_topic."""
        state: AgentState = {
            "current_topic": "OOP",
            "evaluation": {"score": 7.0, "verdict": "Pass"},
            "student_profile": {"topics": ["OOP", "Collections", "Concurrency"]},
            "weak_topics": [],
            "mastered_topics": ["OOP"],
        }
        updated = decide_next_action(state)
        assert updated["next_action"] == "next_topic"
        assert "Collections" in updated["recommended_next_task"]

    def test_last_topic_passed_finishes_workflow(self):
        """When the last curriculum topic is passed, next_action must be 'finish'."""
        state: AgentState = {
            "current_topic": "Collections",
            "evaluation": {"score": 8.5, "verdict": "Pass"},
            "student_profile": {"topics": ["OOP", "Collections"]},
            "weak_topics": [],
            "mastered_topics": ["OOP", "Collections"],
        }
        updated = decide_next_action(state)
        assert updated["next_action"] == "finish"
        assert "Congratulations" in updated["recommended_next_task"]

    def test_evaluator_handles_empty_answer_gracefully(self):
        """Empty student answers must be caught and evaluated to 0.0 score."""
        state: AgentState = {
            "student_answer": "",
            "current_topic": "Java Exceptions",
            "current_question": "What is try-with-resources?",
        }
        res = evaluate_student_answer(state)
        assert res["evaluation"]["score"] == 0.0
        assert res["evaluation"]["is_passed"] is False


# ─────────────────────────────────────────────────────────────────────────────
# 3. Database Relational Integrity & Longitudinal Memory
# ─────────────────────────────────────────────────────────────────────────────

class TestDatabaseRelationalIntegrity:
    def test_student_and_goal_cascade_sync(self, db_session):
        data = SAMPLE_STUDENTS[1]
        student = get_or_create_student(
            db_session,
            name=data["name"],
            email=data["email"],
            target_technology=data["target_technology"],
            available_days=data["available_days"],
        )
        assert student.id is not None

        goal = get_or_create_goal(
            db_session,
            student_id=student.id,
            goal_text=data["goal"],
            target_technology=data["target_technology"],
            days_available=data["available_days"],
        )
        assert goal.id is not None
        assert goal.student_id == student.id

        topics = sync_goal_topics(
            db_session,
            goal.id,
            ["Python Data Structures", "Algorithms", "Dynamic Programming"],
        )
        assert len(topics) == 3
        assert topics[0].name == "Python Data Structures"

    def test_memory_aggregation_and_topic_mastery(self, db_session, sample_student_with_goal):
        student, goal = sample_student_with_goal

        # Record a question on Concurrency
        q = record_question(
            db_session,
            goal_id=goal.id,
            topic_name="Concurrency",
            question_text="Explain synchronized blocks vs locks.",
            difficulty="Medium",
        )
        assert q.id is not None

        # Record a strong answer (score 8.5)
        ans, ev = record_answer_and_evaluation(
            db_session,
            student_id=student.id,
            question_id=q.id,
            answer_text="Synchronized is intrinsic lock; ReentrantLock gives timed lock and fair queueing.",
            eval_data={
                "score": 8.5,
                "strengths": ["Clear distinction of features"],
                "weaknesses": ["None"],
                "verdict": "Pass",
                "is_passed": True,
            },
        )
        assert ans.id is not None
        assert ev.score == 8.5

        # Update progress for Concurrency
        prog = update_progress(
            db=db_session,
            student_id=student.id,
            goal_id=goal.id,
            completed_topics=["Concurrency"],
            weak_topics=[],
            mastered_topics=["Concurrency"],
            topic_scores={"Concurrency": [8.5]},
        )
        assert prog.mastered_topics == ["Concurrency"]

        # Check memory aggregation
        memory = get_student_memory(db_session, student.name)
        assert memory["has_memory"] is True
        assert "Concurrency" in memory["mastered_topics"]

    def test_weak_topic_detection_in_memory(self, db_session, sample_student_with_goal):
        student, goal = sample_student_with_goal

        q = record_question(
            db_session,
            goal_id=goal.id,
            topic_name="Garbage Collection",
            question_text="Explain G1 GC phases.",
        )
        ans, ev = record_answer_and_evaluation(
            db_session,
            student_id=student.id,
            question_id=q.id,
            answer_text="GC deletes memory automatically.",
            eval_data={
                "score": 4.5,
                "strengths": ["Knows GC cleans memory"],
                "weaknesses": ["Missing marking, copying, concurrent mark sweep phases"],
                "verdict": "Weak",
                "is_passed": False,
            },
        )
        update_progress(
            db=db_session,
            student_id=student.id,
            goal_id=goal.id,
            completed_topics=[],
            weak_topics=["Garbage Collection"],
            mastered_topics=[],
            topic_scores={"Garbage Collection": [4.5]},
        )

        memory = get_student_memory(db_session, student.name)
        assert "Garbage Collection" in memory["weak_topics"]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Placement Features & Question Verification
# ─────────────────────────────────────────────────────────────────────────────

class TestPlacementFeatures:
    def test_generate_mcq_format(self):
        res = generate_topic_question(
            topic="Polymorphism",
            technology="Java",
            question_type="mcq",
            difficulty="Medium",
            context_override="Dynamic method dispatch resolves overridden methods at runtime.",
        )
        assert res.success is True
        assert res.question_type == "mcq"
        assert res.mcq_details is not None
        assert len(res.mcq_details.options) == 4
        assert res.mcq_details.correct_option in ("A", "B", "C", "D")

    def test_generate_coding_format(self):
        res = generate_topic_question(
            topic="Two Pointers",
            technology="Java",
            question_type="coding",
            difficulty="Medium",
            context_override="Two pointer technique solves pair sum in O(N).",
        )
        assert res.success is True
        assert res.question_type == "coding"
        assert res.coding_details is not None
        assert len(res.coding_details.sample_test_cases) >= 1
        assert res.coding_details.starter_code is not None

    def test_generate_conceptual_format(self):
        res = generate_topic_question(
            topic="JVM Architecture",
            technology="Java",
            question_type="conceptual",
            difficulty="Medium",
            context_override="JVM memory includes Heap, Stack, Method Area.",
        )
        assert res.success is True
        assert res.question_type == "conceptual"
        assert len(res.question_text) > 10

    def test_placement_readiness_scorecard_calculation(self, client: TestClient, db_session, sample_student_with_goal):
        student, goal = sample_student_with_goal

        # Simulate 2 evaluations: one 9.0, one 8.0 -> average 8.5
        q1 = record_question(db_session, goal_id=goal.id, topic_name="OOP", question_text="Explain Polymorphism.")
        record_answer_and_evaluation(
            db_session,
            student_id=student.id,
            question_id=q1.id,
            answer_text="Polymorphism allows objects to take multiple forms.",
            eval_data={"score": 9.0, "is_passed": True, "verdict": "Pass"},
        )

        q2 = record_question(db_session, goal_id=goal.id, topic_name="Collections", question_text="Explain HashMap.")
        record_answer_and_evaluation(
            db_session,
            student_id=student.id,
            question_id=q2.id,
            answer_text="HashMap uses hashing with buckets.",
            eval_data={"score": 8.0, "is_passed": True, "verdict": "Pass"},
        )
        update_progress(
            db=db_session,
            student_id=student.id,
            goal_id=goal.id,
            completed_topics=["OOP", "Collections"],
            weak_topics=[],
            mastered_topics=["OOP", "Collections"],
            topic_scores={"OOP": [9.0], "Collections": [8.0]},
        )

        # Generate readiness report via endpoint
        payload = {
            "student_name": student.name,
            "target_technology": "Java",
            "target_role": "Systems Engineer",
        }
        res = client.post("/api/placement/readiness-report", json=payload)
        assert res.status_code == 200
        report = res.json()
        assert report["overall_readiness_score"] >= 60.0
        assert report["target_role"] == "Systems Engineer"
        assert "readiness_level" in report
        assert "placement_advisor_verdict" in report


# ─────────────────────────────────────────────────────────────────────────────
# 5. RAG Document Ingestion & Chunking Safety
# ─────────────────────────────────────────────────────────────────────────────

class TestDocumentChunkingAndSafety:
    def test_chunking_empty_list_returns_empty_list(self):
        chunks = chunk_documents([])
        assert chunks == []

    def test_chunking_short_document_preserves_content(self):
        doc = Document(page_content="This is a concise placement guide snippet.", metadata={"source": "test.txt"})
        chunks = chunk_documents([doc], chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1
        assert chunks[0].page_content == doc.page_content
        assert chunks[0].metadata["chunk_index"] == 0
        assert chunks[0].metadata["total_chunks"] == 1
        assert "ingested_at" in chunks[0].metadata

    def test_chunking_long_document_splits_with_metadata(self):
        sample_doc = SAMPLE_DOCUMENTS[0]
        doc = Document(page_content=sample_doc["content"], metadata={"source": sample_doc["filename"]})
        chunks = chunk_documents([doc], chunk_size=120, chunk_overlap=20)
        assert len(chunks) >= 2
        for idx, c in enumerate(chunks):
            assert c.metadata["chunk_index"] == idx
            assert c.metadata["total_chunks"] == len(chunks)
            assert c.metadata["source"] == sample_doc["filename"]
