"""
CareerPilot — Mock Interview Agent Router (Phase 8)

REST API Endpoints for the Mock Interview Workflow:
  - POST /api/interview/start   -> Initialize interview session & ask opening question
  - POST /api/interview/answer  -> Submit candidate answer, score turn (tech + comm), get next Q / final report
  - GET  /api/interview/{id}/report -> Retrieve final hiring scorecard
"""

from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database.connection import get_db
from interview.schemas import (
    InterviewAnswerRequest,
    InterviewFinalSummary,
    InterviewStartRequest,
    InterviewTurnEvaluation,
    InterviewTurnQuestion,
)
from interview.service import (
    get_interview_summary,
    start_interview_session,
    submit_interview_turn,
)

interview_router = APIRouter()


@interview_router.post(
    "/start",
    response_model=InterviewTurnQuestion,
    status_code=status.HTTP_201_CREATED,
    summary="Start a new mock interview session and pose the opening question",
)
def start_interview(
    payload: InterviewStartRequest,
    db: Session = Depends(get_db),
):
    """
    Launch a simulated placement interview session.
    The Interviewer Agent formulates a role-tailored opening technical question.
    """
    try:
        turn_q = start_interview_session(
            student_name=payload.student_name,
            target_role=payload.target_role,
            technology=payload.technology,
            total_questions=payload.total_questions,
            difficulty=payload.difficulty,
            db=db,
        )
        return turn_q
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start mock interview: {str(exc)}",
        )


@interview_router.post(
    "/answer",
    summary="Submit candidate answer to current interview question",
)
def submit_answer(
    payload: InterviewAnswerRequest,
    db: Session = Depends(get_db),
):
    """
    Candidate submits their verbal/written answer to the active interview question.
    The Evaluator Agent scores:
      - Technical depth & correctness (0-10)
      - Communication clarity & structure (0-10)
    If more questions remain, the Interviewer Agent poses the next question.
    If the session is complete, returns the final hiring committee readiness scorecard.
    """
    try:
        turn_eval, final_summary = submit_interview_turn(
            session_id=payload.session_id,
            question_index=payload.question_index,
            student_answer=payload.student_answer,
            db=db,
        )
        return {
            "evaluation": turn_eval,
            "final_summary": final_summary,
            "is_finished": turn_eval.is_finished,
        }
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(val_err),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to evaluate interview answer: {str(exc)}",
        )


@interview_router.get(
    "/{session_id}/report",
    response_model=InterviewFinalSummary,
    summary="Retrieve final mock interview performance report and hiring verdict",
)
def get_report(session_id: str):
    """Fetch complete hiring committee debrief for a finished mock interview."""
    summary = get_interview_summary(session_id)
    if not summary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No finalized interview report found for session '{session_id}'.",
        )
    return summary
