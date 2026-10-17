"""
CareerPilot — Database Package (Phase 6 — Persistent Memory)

Provides SQLite persistence using SQLAlchemy ORM for students, preparation goals,
topics, practice questions, answers, evaluations, progress, and study sessions.
"""

from database.connection import Base, engine, get_db, init_db
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

__all__ = [
    "Base",
    "engine",
    "get_db",
    "init_db",
    "Student",
    "Goal",
    "Topic",
    "Question",
    "Answer",
    "Evaluation",
    "Progress",
    "StudySession",
]
