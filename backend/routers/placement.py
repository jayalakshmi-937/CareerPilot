"""
CareerPilot — Placement Features API Router (Phase 7)

Exposes:
  - POST /api/placement/question         — Generate specific question format (MCQ, Coding, Conceptual) for a topic
  - POST /api/placement/mcq/check        — Instant client-side MCQ answer validation & scoring
  - POST /api/placement/readiness-report — Generate candidate's comprehensive Placement Readiness Report
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database.connection import get_db
from database.repository import get_or_create_student, record_answer_and_evaluation, record_question
from placement.generator import generate_readiness_report, generate_topic_question
from placement.schemas import (
    ReadinessReport,
    ReadinessReportRequest,
    TopicPracticeRequest,
    TopicPracticeResponse,
)

router = APIRouter(prefix="/api/placement", tags=["Placement Features"])


class MCQCheckRequest(BaseModel):
    student_name: str = "Student"
    topic: str
    question_text: str
    selected_option: str = Field(..., description="Option chosen by student: A, B, C, or D")
    correct_option: str = Field(..., description="The known correct option: A, B, C, or D")
    explanation: Optional[str] = None


class MCQCheckResponse(BaseModel):
    is_correct: bool
    score: float
    verdict: str
    selected_option: str
    correct_option: str
    explanation: str


@router.post(
    "/question",
    response_model=TopicPracticeResponse,
    summary="Generate targeted placement question (MCQ, Coding, Conceptual)",
)
def get_topic_question(request: TopicPracticeRequest):
    """
    Generate an authentic placement question tailored to the chosen topic and format:
    - mcq: Multiple Choice with 4 options and automated answer key
    - coding: Algorithmic challenge with problem description, constraints, and test cases
    - conceptual: Deep-dive interview question on architecture and mechanics
    """
    try:
        return generate_topic_question(
            topic=request.topic,
            technology=request.technology,
            question_type=request.question_type,
            difficulty=request.difficulty,
        )
    except Exception as exc:
        logger.exception(f"Error generating {request.question_type} question: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Question generation failed: {str(exc)}",
        )


@router.post(
    "/mcq/check",
    response_model=MCQCheckResponse,
    summary="Check student's MCQ selection and persist result",
)
def check_mcq_answer(
    request: MCQCheckRequest,
    db: Session = Depends(get_db),
):
    """
    Verify student's chosen option, compute immediate score, and record into SQLite.
    """
    is_correct = request.selected_option.strip().upper() == request.correct_option.strip().upper()
    score = 10.0 if is_correct else 2.0
    verdict = "Mastered" if is_correct else "Weak"

    try:
        student = get_or_create_student(db, request.student_name)
        # Record question in database if not recorded
        q_record = record_question(
            db=db,
            goal_id=student.id,
            topic_name=request.topic,
            question_text=request.question_text,
            difficulty="Medium",
            category="MCQ",
        )
        record_answer_and_evaluation(
            db=db,
            student_id=student.id,
            question_id=q_record.id,
            answer_text=f"Selected Option: {request.selected_option.upper()}",
            eval_data={
                "score": score,
                "is_passed": is_correct,
                "verdict": verdict,
                "feedback": request.explanation or ("Correct answer!" if is_correct else "Incorrect selection."),
            },
        )
    except Exception as p_err:
        logger.warning(f"Could not persist MCQ result to SQLite: {p_err}")

    return MCQCheckResponse(
        is_correct=is_correct,
        score=score,
        verdict=verdict,
        selected_option=request.selected_option.upper(),
        correct_option=request.correct_option.upper(),
        explanation=request.explanation or "Standard placement test question.",
    )


@router.post(
    "/readiness-report",
    response_model=ReadinessReport,
    summary="Generate comprehensive Placement Readiness Report",
)
def get_readiness_report(
    request: ReadinessReportRequest,
    db: Session = Depends(get_db),
):
    """
    Synthesize candidate's longitudinal evaluation records, topic mastery, and
    generate an executive Placement Readiness Report with company fit and action plan.
    """
    try:
        return generate_readiness_report(
            db=db,
            student_name=request.student_name,
            target_technology=request.target_technology or "Java",
            target_role=request.target_role or "Software Engineer",
        )
    except Exception as exc:
        logger.exception(f"Error generating readiness report: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Readiness report generation failed: {str(exc)}",
        )
