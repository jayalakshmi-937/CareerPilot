"""
CareerPilot — Phase 6 Tests (Persistent Memory with SQLite)

Covers:
  - SQLAlchemy model definitions (Student, Goal, Topic, Question, Answer, Evaluation, Progress, StudySession)
  - Database repository CRUD operations & relationships
  - Student memory retrieval & persistent state injection
  - FastAPI student & history endpoints (/api/students)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.connection import Base, get_db
from database.models import (
    Answer,
    Evaluation,
    Goal,
    Progress,
    Question,
    Student,
    StudySession,
    Topic,
)
from database.repository import (
    get_or_create_goal,
    get_or_create_student,
    get_student_memory,
    record_answer_and_evaluation,
    record_question,
    save_study_session,
    sync_goal_topics,
    update_progress,
)


# ── Test Database Fixture ─────────────────────────────────────────────────────

from sqlalchemy.pool import StaticPool

@pytest.fixture(name="db_session")
def fixture_db_session():
    """In-memory SQLite session for fast isolated testing."""
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
    """FastAPI TestClient with overridden get_db dependency."""
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
# 1. SQLAlchemy Model Unit Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDatabaseModels:
    def test_student_model_creation(self, db_session):
        student = Student(name="Aman Varma", target_technology="Java", available_days=10)
        db_session.add(student)
        db_session.commit()
        db_session.refresh(student)

        assert student.id is not None
        assert student.name == "Aman Varma"
        d = student.to_dict()
        assert d["name"] == "Aman Varma"
        assert d["target_technology"] == "Java"
        assert d["available_days"] == 10

    def test_question_hints_json_property(self, db_session):
        q = Question(question_text="What is volatile in Java?", hints=["Memory visibility", "Happens-before"])
        db_session.add(q)
        db_session.commit()
        db_session.refresh(q)

        assert len(q.hints) == 2
        assert "Memory visibility" in q.hints

    def test_evaluation_json_properties(self, db_session):
        q = Question(question_text="Sample Q")
        db_session.add(q)
        db_session.commit()

        ans = Answer(question_id=q.id, student_id=1, answer_text="Sample answer")
        db_session.add(ans)
        db_session.commit()

        ev = Evaluation(
            answer_id=ans.id,
            score=8.5,
            verdict="Mastered",
            is_passed=True,
            strengths=["Concise", "Accurate"],
            areas_for_improvement=["Add code example"],
        )
        db_session.add(ev)
        db_session.commit()
        db_session.refresh(ev)

        assert ev.score == 8.5
        assert ev.is_passed is True
        assert len(ev.strengths) == 2
        assert ev.areas_for_improvement == ["Add code example"]

    def test_progress_lists_json_property(self, db_session):
        prog = Progress(
            student_id=1,
            goal_id=1,
            completed_topics=["OOP", "Generics"],
            weak_topics=["Concurrency"],
            mastered_topics=["OOP"],
            topic_scores={"OOP": [8.0, 9.0], "Concurrency": [4.0]},
        )
        db_session.add(prog)
        db_session.commit()
        db_session.refresh(prog)

        assert prog.completed_topics == ["OOP", "Generics"]
        assert prog.weak_topics == ["Concurrency"]
        assert prog.topic_scores["OOP"] == [8.0, 9.0]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Repository & CRUD Layer Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDatabaseRepository:
    def test_get_or_create_student_idempotency(self, db_session):
        s1 = get_or_create_student(db_session, "Neha Singh", "Python", 5)
        s2 = get_or_create_student(db_session, "neha singh", "Python Full Stack", 7)

        assert s1.id == s2.id
        assert s2.target_technology == "Python Full Stack"
        assert s2.available_days == 7

    def test_sync_goal_topics(self, db_session):
        student = get_or_create_student(db_session, "Kiran Rao", "Java")
        goal = get_or_create_goal(db_session, student.id, "Java Prep", "Java", 7)

        topics = sync_goal_topics(db_session, goal.id, ["OOP", "Collections", "Multithreading"])
        assert len(topics) == 3
        assert topics[0].name == "OOP"

        # Re-syncing same topics does not duplicate
        topics_v2 = sync_goal_topics(db_session, goal.id, ["OOP", "Collections", "Spring Boot"])
        all_db_topics = db_session.query(Topic).filter(Topic.goal_id == goal.id).all()
        assert len(all_db_topics) == 4

    def test_record_answer_and_evaluation_updates_topic_mastery(self, db_session):
        student = get_or_create_student(db_session, "Vikram Seth", "Java")
        goal = get_or_create_goal(db_session, student.id, "Java Prep", "Java", 7)
        q = record_question(db_session, goal.id, "Polymorphism", "Explain runtime polymorphism")

        # 1. Weak evaluation
        ans1, ev1 = record_answer_and_evaluation(
            db=db_session,
            student_id=student.id,
            question_id=q.id,
            answer_text="It just means many forms.",
            eval_data={"score": 4.0, "is_passed": False, "verdict": "Weak"},
        )
        topic = db_session.query(Topic).filter(Topic.id == q.topic_id).first()
        assert topic.is_weak is True
        assert topic.is_mastered is False

        # 2. Strong evaluation
        ans2, ev2 = record_answer_and_evaluation(
            db=db_session,
            student_id=student.id,
            question_id=q.id,
            answer_text="Runtime polymorphism is resolved at runtime via dynamic method dispatch using vtables.",
            eval_data={"score": 9.0, "is_passed": True, "verdict": "Mastered"},
        )
        db_session.refresh(topic)
        assert topic.is_weak is False
        assert topic.is_mastered is True

    def test_student_memory_retrieval(self, db_session):
        student = get_or_create_student(db_session, "Rohan Das", "C++")
        goal = get_or_create_goal(db_session, student.id, "C++ Prep", "C++", 5)

        # Before any progress recorded
        mem_empty = get_student_memory(db_session, "Rohan Das")
        assert mem_empty["has_memory"] is True
        assert mem_empty["weak_topics"] == []

        # Record progress
        update_progress(
            db=db_session,
            student_id=student.id,
            goal_id=goal.id,
            completed_topics=["Pointers"],
            weak_topics=["Smart Pointers"],
            mastered_topics=["Pointers"],
            topic_scores={"Pointers": [8.0], "Smart Pointers": [4.5]},
        )

        mem = get_student_memory(db_session, "Rohan Das")
        assert "Smart Pointers" in mem["weak_topics"]
        assert "Pointers" in mem["mastered_topics"]
        assert mem["topic_scores"]["Smart Pointers"] == [4.5]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Students API Endpoints Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestStudentsAPIEndpoints:
    def test_list_students_empty(self, client):
        resp = client.get("/api/students")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_student_detail_and_404(self, client, db_session):
        # 404 test
        resp_404 = client.get("/api/students/NonExistentStudent")
        assert resp_404.status_code == 404

        # 200 test
        student = get_or_create_student(db_session, "Priya Nair", "Python", 7)
        resp_200 = client.get(f"/api/students/{student.id}")
        assert resp_200.status_code == 200
        data = resp_200.json()
        assert data["student"]["name"] == "Priya Nair"

    def test_get_student_history_and_progress(self, client, db_session):
        student = get_or_create_student(db_session, "Amit Kumar", "Java")
        goal = get_or_create_goal(db_session, student.id, "Goal", "Java", 7)
        q = record_question(db_session, goal.id, "Generics", "Why use generics?")
        ans, ev = record_answer_and_evaluation(
            db_session, student.id, q.id, "Type safety at compile time",
            {"score": 8.0, "is_passed": True, "verdict": "Pass", "strengths": ["Clear definition"]},
        )

        resp_hist = client.get(f"/api/students/{student.id}/history")
        assert resp_hist.status_code == 200
        history = resp_hist.json()
        assert len(history) == 1
        assert history[0]["score"] == 8.0
        assert history[0]["topic_name"] == "Generics"

        resp_prog = client.get(f"/api/students/{student.id}/progress")
        assert resp_prog.status_code == 200
        prog = resp_prog.json()
        assert prog["student_id"] == student.id

    def test_delete_student(self, client, db_session):
        student = get_or_create_student(db_session, "To Delete", "Java")
        resp = client.delete(f"/api/students/{student.id}")
        assert resp.status_code == 200
        deleted = db_session.query(Student).filter(Student.id == student.id).first()
        assert deleted is None
