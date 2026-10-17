"""
CareerPilot — Pytest Shared Fixtures (Phase 9 — Test Suite)

Provides globally available pytest fixtures:
  - In-memory SQLite database session (isolated, fast, thread-safe with StaticPool)
  - FastAPI TestClient with overridden get_db dependency
  - Pre-seeded sample student & goal models from sample_data
  - Reusable mock LLM generators for isolated unit testing
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool
from starlette.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app
from database.connection import Base, get_db
from database.models import Goal, Student, Topic
from database.repository import get_or_create_goal, get_or_create_student
from tests.sample_data import SAMPLE_ANSWERS, SAMPLE_DOCUMENTS, SAMPLE_STUDENTS


# ── Database Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(name="db_session")
def fixture_db_session() -> Generator[Session, None, None]:
    """Provides a fresh, isolated in-memory SQLite database for each test."""
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
def fixture_client(db_session: Session) -> Generator[TestClient, None, None]:
    """FastAPI TestClient wired to the in-memory SQLite database."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app, raise_server_exceptions=True) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# ── Seeded Model Fixtures ─────────────────────────────────────────────────────

@pytest.fixture(name="sample_student")
def fixture_sample_student(db_session: Session) -> Student:
    """Creates and returns a persisted sample Student in db_session."""
    data = SAMPLE_STUDENTS[0]
    student = get_or_create_student(
        db_session,
        name=data["name"],
        email=data["email"],
        target_technology=data["target_technology"],
        available_days=data["available_days"],
    )
    return student


@pytest.fixture(name="sample_student_with_goal")
def fixture_sample_student_with_goal(db_session: Session, sample_student: Student) -> tuple[Student, Goal]:
    """Creates and returns a persisted Student and associated Goal in db_session."""
    data = SAMPLE_STUDENTS[0]
    goal = get_or_create_goal(
        db_session,
        student_id=sample_student.id,
        goal_text=data["goal"],
        target_technology=sample_student.target_technology,
        days_available=sample_student.available_days,
    )
    return sample_student, goal


# ── Mock LLM Fixture ──────────────────────────────────────────────────────────

@pytest.fixture(name="mock_llm_factory")
def fixture_mock_llm_factory():
    """Factory helper to construct mock LLM instances returning controlled JSON or text."""
    def _create_mock(return_content: str):
        mock = MagicMock()
        mock_response = MagicMock()
        mock_response.content = return_content
        mock.invoke.return_value = mock_response
        return mock
    return _create_mock
