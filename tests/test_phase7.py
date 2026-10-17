"""
CareerPilot — Phase 7 Tests (Placement Features)

Covers:
  - MCQ question generation & validation schemas
  - Coding challenge generation schemas
  - Final Placement Readiness Report calculation & schema
  - FastAPI placement endpoints (/api/placement/question, /api/placement/mcq/check, /api/placement/readiness-report)
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
from database.models import Progress, Student, Topic
from database.repository import (
    get_or_create_student,
    record_answer_and_evaluation,
    record_question,
    update_progress,
)
from placement.generator import generate_readiness_report, generate_topic_question
from placement.schemas import (
    CodingDetails,
    CodingTestCase,
    MCQDetails,
    MCQOption,
    ReadinessReport,
    TopicPracticeResponse,
)


# ── Test Database Fixture ─────────────────────────────────────────────────────

@pytest.fixture(name="db_session")
def fixture_db_session():
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


@pytest.fixture(name="client")
def fixture_client(db_session):
    from backend.main import app

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    app.dependency_overrides.clear()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Placement Schemas Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPlacementSchemas:
    def test_mcq_details_validation(self):
        opts = [
            MCQOption(label="A", text="Encapsulation"),
            MCQOption(label="B", text="Inheritance"),
            MCQOption(label="C", text="Polymorphism"),
            MCQOption(label="D", text="Abstraction"),
        ]
        mcq = MCQDetails(
            options=opts,
            correct_option="A",
            explanation="Encapsulation wraps data and methods.",
        )
        assert mcq.correct_option == "A"
        assert len(mcq.options) == 4

    def test_coding_details_validation(self):
        tc = CodingTestCase(input="[1, 2, 3]", expected_output="6")
        coding = CodingDetails(
            problem_statement="Calculate sum of array elements.",
            constraints=["1 <= N <= 10^5"],
            starter_code="int sum(int[] arr) {}",
            sample_test_cases=[tc],
            optimal_time_complexity="O(N)",
            optimal_space_complexity="O(1)",
        )
        assert coding.optimal_time_complexity == "O(N)"
        assert len(coding.sample_test_cases) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 2. Generator Logic Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPlacementGenerator:
    def test_generate_mcq_question_fallback_or_mock(self):
        """Test MCQ generation produces valid structured options."""
        res = generate_topic_question(
            topic="Polymorphism",
            technology="Java",
            question_type="mcq",
            difficulty="Medium",
            context_override="Polymorphism is resolved via dynamic method dispatch.",
        )
        assert res.success is True
        assert res.question_type == "mcq"
        assert res.mcq_details is not None
        assert len(res.mcq_details.options) == 4
        assert res.mcq_details.correct_option in ("A", "B", "C", "D")

    def test_generate_coding_question_fallback_or_mock(self):
        """Test coding challenge generation produces test cases and starter code."""
        res = generate_topic_question(
            topic="Binary Search",
            technology="Java",
            question_type="coding",
            difficulty="Medium",
            context_override="Binary search runs in O(log N).",
        )
        assert res.success is True
        assert res.question_type == "coding"
        assert res.coding_details is not None
        assert len(res.coding_details.sample_test_cases) >= 1
        assert res.coding_details.starter_code is not None

    def test_generate_conceptual_question(self):
        res = generate_topic_question(
            topic="Garbage Collection",
            technology="Java",
            question_type="conceptual",
            difficulty="Medium",
            context_override="JVM uses G1GC and ZGC.",
        )
        assert res.success is True
        assert res.question_type == "conceptual"
        assert len(res.question_text) > 10

    def test_readiness_report_deterministic_calculation(self, db_session):
        student = get_or_create_student(db_session, "Kavya Patel", "Java")
        q = record_question(
            db=db_session,
            goal_id=1,
            topic_name="OOP",
            question_text="Explain encapsulation in Java.",
            difficulty="Medium",
            category="conceptual",
        )
        record_answer_and_evaluation(
            db=db_session,
            student_id=student.id,
            question_id=q.id,
            answer_text="Encapsulation is data hiding using private fields and public getter/setter methods.",
            eval_data={
                "score": 9.0,
                "is_passed": True,
                "verdict": "Pass",
                "feedback": "Accurate conceptual answer.",
                "strengths": ["Data hiding", "Getter/Setter accessors"],
                "areas_for_improvement": [],
            },
        )
        update_progress(
            db=db_session,
            student_id=student.id,
            goal_id=1,
            completed_topics=["OOP", "Collections"],
            weak_topics=[],
            mastered_topics=["OOP", "Collections"],
            topic_scores={"OOP": [9.0], "Collections": [8.5]},
        )

        report = generate_readiness_report(
            db=db_session,
            student_name="Kavya Patel",
            target_technology="Java",
            target_role="Software Engineer",
        )
        assert report.student_name == "Kavya Patel"
        assert report.overall_readiness_score >= 40
        assert report.total_topics_mastered >= 2
        assert len(report.competencies) >= 2
        assert len(report.action_plan_next_steps) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 3. Placement API Endpoints Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPlacementAPIEndpoints:
    def test_placement_question_endpoint(self, client):
        payload = {
            "student_name": "Test Student",
            "technology": "Java",
            "topic": "Inheritance",
            "question_type": "mcq",
            "difficulty": "Medium",
        }
        resp = client.post("/api/placement/question", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["question_type"] == "mcq"
        assert data["mcq_details"] is not None

    def test_mcq_check_endpoint_correct(self, client):
        payload = {
            "student_name": "Test Student",
            "topic": "Java Interfaces",
            "question_text": "Can an interface have private methods in Java 9+?",
            "selected_option": "A",
            "correct_option": "A",
            "explanation": "Java 9 introduced private interface methods.",
        }
        resp = client.post("/api/placement/mcq/check", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_correct"] is True
        assert data["score"] == 10.0
        assert data["verdict"] == "Mastered"

    def test_mcq_check_endpoint_incorrect(self, client):
        payload = {
            "student_name": "Test Student",
            "topic": "Java Interfaces",
            "question_text": "Can an interface have private methods in Java 9+?",
            "selected_option": "B",
            "correct_option": "A",
            "explanation": "Java 9 introduced private interface methods.",
        }
        resp = client.post("/api/placement/mcq/check", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_correct"] is False
        assert data["score"] == 2.0
        assert data["verdict"] == "Weak"

    def test_readiness_report_endpoint(self, client, db_session):
        student = get_or_create_student(db_session, "Rahul Roy", "Java")
        payload = {
            "student_name": "Rahul Roy",
            "target_technology": "Java",
            "target_role": "Backend Engineer",
        }
        resp = client.post("/api/placement/readiness-report", json=payload)
        assert resp.status_code == 200
        report = resp.json()
        assert report["student_name"] == "Rahul Roy"
        assert "overall_readiness_score" in report
        assert "company_fit" in report
        assert len(report["competencies"]) >= 1
