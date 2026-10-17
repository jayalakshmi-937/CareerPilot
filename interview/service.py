"""
CareerPilot — Mock Interview Service & Session Manager (Phase 8)

Coordinates interview state machines, SQLite persistence, and LangGraph nodes.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Optional

from loguru import logger
from sqlalchemy.orm import Session

from database.models import Answer, Evaluation, Question, Student, StudySession
from database.repository import get_or_create_student, record_answer_and_evaluation, record_question
from interview.schemas import (
    InterviewFinalSummary,
    InterviewTurnEvaluation,
    InterviewTurnQuestion,
    TopicPerformance,
)
from interview.workflow import (
    InterviewState,
    evaluator_node,
    finalizer_node,
    interviewer_node,
)


# In-memory interview session cache: session_id -> InterviewState
_ACTIVE_SESSIONS: Dict[str, InterviewState] = {}


def start_interview_session(
    student_name: str,
    target_role: str = "Software Development Engineer (SDE-1)",
    technology: str = "Java",
    total_questions: int = 5,
    difficulty: str = "Medium",
    db: Optional[Session] = None,
) -> InterviewTurnQuestion:
    """
    Initialize a new mock interview session and generate the opening interview question.
    """
    clean_name = student_name.strip() or "Candidate"
    session_id = f"interview-{uuid.uuid4().hex[:8]}"

    # Ensure student profile exists in SQLite if db is provided
    student_id = None
    if db:
        student = get_or_create_student(db, clean_name, technology)
        student_id = student.id

    initial_state: InterviewState = {
        "session_id": session_id,
        "student_name": clean_name,
        "target_role": target_role,
        "technology": technology,
        "total_questions": total_questions,
        "difficulty": difficulty,
        "current_index": 0,
        "current_question": {},
        "student_answer": None,
        "last_evaluation": None,
        "history": [],
        "final_summary": None,
        "status": "in_progress",
        "error": None,
    }

    # Generate Question 1 via Interviewer Node
    step_result = interviewer_node(initial_state)
    initial_state.update(step_result)

    q_data = initial_state["current_question"]

    # Persist question in SQLite if db is present
    if db and student_id:
        try:
            record_question(
                db=db,
                goal_id=1,
                topic_name=q_data.get("topic", "Interview Question"),
                question_text=q_data.get("question_text", ""),
                difficulty=q_data.get("difficulty", "Medium"),
                category="Mock Interview",
                hints=q_data.get("hints", []),
            )
        except Exception as exc:
            logger.warning(f"Could not log interview question to SQLite: {exc}")

    _ACTIVE_SESSIONS[session_id] = initial_state
    logger.info(f"Initialized mock interview session {session_id} for '{clean_name}'")

    return InterviewTurnQuestion(
        session_id=session_id,
        question_index=q_data["question_index"],
        total_questions=q_data["total_questions"],
        topic=q_data["topic"],
        difficulty=q_data["difficulty"],
        question_text=q_data["question_text"],
        interviewer_preface=q_data.get("interviewer_preface"),
        hints=q_data.get("hints", []),
    )


def submit_interview_turn(
    session_id: str,
    question_index: int,
    student_answer: str,
    db: Optional[Session] = None,
) -> tuple[InterviewTurnEvaluation, Optional[InterviewFinalSummary]]:
    """
    Process candidate answer:
      1. Evaluates technical correctness & communication quality
      2. If more questions remain, generates next question
      3. If all questions complete, finalizes comprehensive interview scorecard
    """
    state = _ACTIVE_SESSIONS.get(session_id)
    if not state:
        raise ValueError(f"Interview session '{session_id}' not found or expired.")

    state["student_answer"] = student_answer.strip()

    # 1. Run Evaluator Agent Node
    eval_result = evaluator_node(state)
    state.update(eval_result)

    eval_data = state["last_evaluation"]
    current_q = state.get("current_question") or {}
    topic_name = current_q.get("topic", "General")

    # 2. Persist answer & evaluation in SQLite
    if db:
        try:
            student = get_or_create_student(db, state["student_name"], state["technology"])
            q_rec = record_question(
                db=db,
                goal_id=1,
                topic_name=topic_name,
                question_text=current_q.get("question_text", ""),
                difficulty=current_q.get("difficulty", "Medium"),
                category="Mock Interview",
            )
            record_answer_and_evaluation(
                db=db,
                student_id=student.id,
                question_id=q_rec.id,
                answer_text=student_answer.strip(),
                eval_data={
                    "score": eval_data["technical_score"],
                    "is_passed": eval_data["technical_score"] >= 7.0,
                    "verdict": eval_data["verdict"],
                    "feedback": eval_data["feedback"],
                    "strengths": eval_data["strengths"],
                    "areas_for_improvement": eval_data["areas_for_improvement"],
                },
            )
        except Exception as exc:
            logger.warning(f"Failed to record interview turn to SQLite: {exc}")

    # 3. Check if all questions completed
    total_q = state.get("total_questions", 5)
    is_finished = state["current_index"] >= total_q

    next_question_obj: Optional[InterviewTurnQuestion] = None
    final_summary_obj: Optional[InterviewFinalSummary] = None

    if is_finished:
        # Finalize interview
        fin_result = finalizer_node(state)
        state.update(fin_result)
        raw_final = state["final_summary"]

        perf_items = [
            TopicPerformance(
                topic=tp.get("topic", "General"),
                average_technical_score=tp.get("average_technical_score", 7.0),
                average_communication_score=tp.get("average_communication_score", 7.0),
                verdict=tp.get("verdict", "Competent"),
                observations=tp.get("observations", "Solid effort."),
            )
            for tp in raw_final.get("topic_wise_performance", [])
        ]

        final_summary_obj = InterviewFinalSummary(
            session_id=session_id,
            student_name=state["student_name"],
            target_role=state["target_role"],
            technology=state["technology"],
            total_questions=total_q,
            technical_score=raw_final.get("technical_score", 75.0),
            communication_score=raw_final.get("communication_score", 75.0),
            overall_interview_score=raw_final.get("overall_interview_score", 75.0),
            interview_readiness_score=raw_final.get("interview_readiness_score", 75.0),
            readiness_verdict=raw_final.get("readiness_verdict", "Hire"),
            topic_wise_performance=perf_items,
            strong_areas=raw_final.get("strong_areas", []),
            weak_areas=raw_final.get("weak_areas", []),
            recommendations=raw_final.get("recommendations", []),
            interviewer_notes=raw_final.get("interviewer_notes", "Good performance."),
            generated_at=raw_final.get("generated_at", ""),
        )
    else:
        # Generate next question via Interviewer Agent Node
        next_step = interviewer_node(state)
        state.update(next_step)
        next_q_data = state["current_question"]

        next_question_obj = InterviewTurnQuestion(
            session_id=session_id,
            question_index=next_q_data["question_index"],
            total_questions=next_q_data["total_questions"],
            topic=next_q_data["topic"],
            difficulty=next_q_data["difficulty"],
            question_text=next_q_data["question_text"],
            interviewer_preface=next_q_data.get("interviewer_preface"),
            hints=next_q_data.get("hints", []),
        )

    # Save state back
    _ACTIVE_SESSIONS[session_id] = state

    turn_eval = InterviewTurnEvaluation(
        question_index=question_index,
        total_questions=total_q,
        topic=topic_name,
        technical_score=eval_data["technical_score"],
        communication_score=eval_data["communication_score"],
        overall_score=eval_data["overall_score"],
        verdict=eval_data["verdict"],
        feedback=eval_data["feedback"],
        strengths=eval_data["strengths"],
        areas_for_improvement=eval_data["areas_for_improvement"],
        is_finished=is_finished,
        next_question=next_question_obj,
    )

    return turn_eval, final_summary_obj


def get_interview_summary(session_id: str) -> Optional[InterviewFinalSummary]:
    """Retrieve finalized mock interview summary by session ID."""
    state = _ACTIVE_SESSIONS.get(session_id)
    if not state or not state.get("final_summary"):
        return None

    raw = state["final_summary"]
    perf_items = [
        TopicPerformance(
            topic=tp.get("topic", "General"),
            average_technical_score=tp.get("average_technical_score", 7.0),
            average_communication_score=tp.get("average_communication_score", 7.0),
            verdict=tp.get("verdict", "Competent"),
            observations=tp.get("observations", "Solid effort."),
        )
        for tp in raw.get("topic_wise_performance", [])
    ]

    return InterviewFinalSummary(
        session_id=session_id,
        student_name=state["student_name"],
        target_role=state["target_role"],
        technology=state["technology"],
        total_questions=state.get("total_questions", 5),
        technical_score=raw.get("technical_score", 75.0),
        communication_score=raw.get("communication_score", 75.0),
        overall_interview_score=raw.get("overall_interview_score", 75.0),
        interview_readiness_score=raw.get("interview_readiness_score", 75.0),
        readiness_verdict=raw.get("readiness_verdict", "Hire"),
        topic_wise_performance=perf_items,
        strong_areas=raw.get("strong_areas", []),
        weak_areas=raw.get("weak_areas", []),
        recommendations=raw.get("recommendations", []),
        interviewer_notes=raw.get("interviewer_notes", ""),
        generated_at=raw.get("generated_at", ""),
    )
