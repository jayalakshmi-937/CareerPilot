"""
CareerPilot — Students & Historical Performance API Router (Phase 6 — Persistent Memory)

Provides REST endpoints to query:
  - Registered students
  - Student preparation goals and active sessions
  - Complete history of attempted questions, answers, and evaluations
  - Longitudinal mastery and weak topic metrics
"""

from __future__ import annotations

from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func
from sqlalchemy.orm import Session

from database.connection import get_db
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

router = APIRouter(prefix="/api/students", tags=["Students & Persistence"])


# ── Response Schemas ──────────────────────────────────────────────────────────

class StudentSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    email: Optional[str] = None
    target_technology: Optional[str] = None
    available_days: int
    created_at: Optional[str] = None


class AnswerHistoryItem(BaseModel):
    answer_id: int
    question_id: int
    question_text: str
    difficulty: str
    category: str
    topic_name: str
    answer_text: str
    submitted_at: Optional[str] = None
    score: float
    verdict: str
    is_passed: bool
    feedback: Optional[str] = None
    strengths: list[str] = []
    areas_for_improvement: list[str] = []
    remediation_advice: Optional[str] = None


class StudentProfileDetail(BaseModel):
    student: StudentSummary
    active_goal: Optional[dict[str, Any]] = None
    progress: Optional[dict[str, Any]] = None
    latest_session: Optional[dict[str, Any]] = None
    topics: list[dict[str, Any]] = []


# ── Helper ────────────────────────────────────────────────────────────────────

def _find_student(identifier: str, db: Session) -> Student:
    """Find student by numeric ID or case-insensitive name."""
    if identifier.isdigit():
        student = db.query(Student).filter(Student.id == int(identifier)).first()
    else:
        student = db.query(Student).filter(func.lower(Student.name) == identifier.strip().lower()).first()

    if not student:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student '{identifier}' not found in database.",
        )
    return student


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("", response_model=list[dict[str, Any]], summary="List all registered students")
def list_students(db: Session = Depends(get_db)):
    """Retrieve all students stored in the SQLite database."""
    students = db.query(Student).order_by(Student.created_at.desc()).all()
    return [s.to_dict() for s in students]


@router.get("/{identifier}", response_model=dict[str, Any], summary="Get student profile & active goal")
def get_student_detail(identifier: str, db: Session = Depends(get_db)):
    """Retrieve comprehensive profile, active goal, topics, and latest session."""
    student = _find_student(identifier, db)

    goal = (
        db.query(Goal)
        .filter(Goal.student_id == student.id, Goal.is_active == True)
        .order_by(Goal.created_at.desc())
        .first()
    )

    prog = db.query(Progress).filter(Progress.student_id == student.id).first()
    session = (
        db.query(StudySession)
        .filter(StudySession.student_id == student.id)
        .order_by(StudySession.created_at.desc())
        .first()
    )

    topics = []
    if goal:
        topics = [t.to_dict() for t in db.query(Topic).filter(Topic.goal_id == goal.id).all()]

    return {
        "student": student.to_dict(),
        "active_goal": goal.to_dict() if goal else None,
        "progress": prog.to_dict() if prog else None,
        "latest_session": session.to_dict() if session else None,
        "topics": topics,
    }


@router.get("/{identifier}/history", response_model=list[AnswerHistoryItem], summary="Get answer & evaluation history")
def get_student_history(identifier: str, db: Session = Depends(get_db)):
    """
    Retrieve chronological history of all questions attempted by the student,
    along with their answers, scores, verdicts, and AI diagnostic feedback.
    """
    student = _find_student(identifier, db)

    answers = (
        db.query(Answer)
        .filter(Answer.student_id == student.id)
        .order_by(Answer.submitted_at.desc())
        .all()
    )

    history: list[AnswerHistoryItem] = []
    for a in answers:
        q = a.question
        ev = a.evaluation
        topic_name = q.topic.name if q and q.topic else "General"

        history.append(
            AnswerHistoryItem(
                answer_id=a.id,
                question_id=q.id if q else 0,
                question_text=q.question_text if q else "Question deleted",
                difficulty=q.difficulty if q else "Medium",
                category=q.category if q else "General",
                topic_name=topic_name,
                answer_text=a.answer_text,
                submitted_at=a.submitted_at.isoformat() if a.submitted_at else None,
                score=ev.score if ev else 0.0,
                verdict=ev.verdict if ev else "Pending",
                is_passed=ev.is_passed if ev else False,
                feedback=ev.feedback if ev else None,
                strengths=ev.strengths if ev else [],
                areas_for_improvement=ev.areas_for_improvement if ev else [],
                remediation_advice=ev.remediation_advice if ev else None,
            )
        )

    return history


@router.get("/{identifier}/progress", response_model=dict[str, Any], summary="Get student progress record")
def get_student_progress(identifier: str, db: Session = Depends(get_db)):
    """Get aggregated progress record, mastered topics, and weak topics."""
    student = _find_student(identifier, db)
    prog = db.query(Progress).filter(Progress.student_id == student.id).first()
    if not prog:
        return {
            "student_id": student.id,
            "completed_topics": [],
            "weak_topics": [],
            "mastered_topics": [],
            "topic_scores": {},
            "mastered_count": 0,
            "overall_progress_percent": 0,
            "total_questions_attempted": 0,
            "average_score": 0.0,
        }
    return prog.to_dict()


@router.delete("/{identifier}", summary="Reset / delete student record")
def delete_student(identifier: str, db: Session = Depends(get_db)):
    """Delete a student and cascade delete all their goals, progress, answers, and sessions."""
    student = _find_student(identifier, db)
    db.delete(student)
    db.commit()
    logger.info(f"Deleted student record: {student.name} (id={student.id})")
    return {"message": f"Student '{student.name}' successfully deleted."}
