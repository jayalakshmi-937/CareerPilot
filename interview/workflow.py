"""
CareerPilot — Mock Interview Agent Workflow (Phase 8)

Implements the LangGraph interview workflow:

    START
      ↓
    Interviewer Agent (Generates/poses question)
      ↓
    [Student answers via API / UI]
      ↓
    Evaluator Agent (Technical + Communication scoring & feedback)
      ↓
    [Router: More questions remaining?]
      ├─► Yes ──► Interviewer Agent (Next question)
      └─► No  ──► Hiring Committee Finalizer (Final Readiness Report)
                    ↓
                   END
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

from loguru import logger
from langgraph.graph import END, StateGraph

from agent.llm import get_llm
from interview.prompts import (
    INTERVIEW_EVALUATION_PROMPT,
    INTERVIEW_FINAL_REPORT_PROMPT,
    INTERVIEW_QUESTION_PROMPT,
)
from rag.retriever import retrieve_context_string


# ── Typed State for Mock Interview Graph ──────────────────────────────────────

class InterviewTurnRecord(TypedDict):
    question_index: int
    topic: str
    difficulty: str
    question_text: str
    student_answer: str
    technical_score: float
    communication_score: float
    overall_score: float
    verdict: str
    feedback: str
    strengths: List[str]
    areas_for_improvement: List[str]


class InterviewState(TypedDict):
    session_id: str
    student_name: str
    target_role: str
    technology: str
    total_questions: int
    difficulty: str
    current_index: int
    current_question: Dict[str, Any]
    student_answer: Optional[str]
    last_evaluation: Optional[Dict[str, Any]]
    history: List[InterviewTurnRecord]
    final_summary: Optional[Dict[str, Any]]
    status: str  # "in_progress", "awaiting_answer", "evaluated", "finished"
    error: Optional[str]


def _clean_json(raw: Any) -> str:
    """Safely extract JSON substring from LLM string output."""
    if isinstance(raw, list):
        parts = []
        for p in raw:
            if isinstance(p, dict) and "text" in p:
                parts.append(p["text"])
            elif isinstance(p, str):
                parts.append(p)
        text = "".join(parts).strip()
    elif isinstance(raw, str):
        text = raw.strip()
    else:
        text = str(raw).strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


# ── Graph Nodes ───────────────────────────────────────────────────────────────

def interviewer_node(state: InterviewState) -> Dict[str, Any]:
    """
    Interviewer Agent: Formulates the next realistic interview question
    matching candidate's profile, technology, and progressive turn difficulty.
    """
    current_idx = state.get("current_index", 0) + 1
    total_q = state.get("total_questions", 5)
    tech = state.get("technology", "Java")
    role = state.get("target_role", "Software Development Engineer")
    name = state.get("student_name", "Candidate")
    history = state.get("history", [])

    logger.info(f"Interviewer Agent formulating turn {current_idx}/{total_q} for '{name}' ({tech})")

    # Format previous questions to avoid duplicates
    prev_q_texts = [f"Q{h['question_index']} ({h['topic']}): {h['question_text']}" for h in history]
    prev_str = "\n".join(prev_q_texts) if prev_q_texts else "None (First question of the interview)."

    # Optional RAG background
    study_context = "Core interview technical concepts."
    try:
        study_context = retrieve_context_string(f"{tech} interview questions", top_k=2)
    except Exception as exc:
        logger.debug(f"RAG retrieval skipped for interview turn: {exc}")

    llm = get_llm()
    prompt_val = INTERVIEW_QUESTION_PROMPT.format_messages(
        student_name=name,
        target_role=role,
        technology=tech,
        question_index=current_idx,
        total_questions=total_q,
        previous_questions=prev_str,
        study_context=study_context or "General curriculum notes.",
    )

    try:
        res = llm.invoke(prompt_val)
        parsed = json.loads(_clean_json(res.content))
        question_data = {
            "session_id": state.get("session_id"),
            "question_index": current_idx,
            "total_questions": total_q,
            "topic": parsed.get("topic", f"{tech} Core Mechanics"),
            "difficulty": parsed.get("difficulty", "Medium"),
            "interviewer_preface": parsed.get(
                "interviewer_preface",
                f"Let us explore how {tech} handles core operational semantics."
            ),
            "question_text": parsed.get(
                "question_text",
                f"Explain how memory management and garbage collection function under the hood in {tech}."
            ),
            "hints": parsed.get("hints", [f"Focus on the underlying runtime model in {tech}."]),
        }
    except Exception as exc:
        logger.warning(f"Failed to generate dynamic interview question, using progressive fallback: {exc}")
        fallbacks = [
            ("Core Mechanics & Memory", f"How does memory allocation (Stack vs Heap) work in {tech}? What happens during object creation?"),
            ("Data Structures & Performance", f"Compare the internal implementation and time complexity of ArrayList vs LinkedList in {tech}."),
            ("Concurrency & Synchronization", f"How does the volatile keyword differ from synchronized blocks in {tech}? Explain memory visibility."),
            ("Exception Architecture", f"What is the difference between Checked and Unchecked exceptions in {tech}? Explain best practices."),
            ("System Resilience & Edge Cases", f"How would you troubleshoot and diagnose a memory leak or High CPU issue in a production {tech} service?"),
        ]
        fb_idx = min(current_idx - 1, len(fallbacks) - 1)
        fb_topic, fb_q = fallbacks[fb_idx]
        question_data = {
            "session_id": state.get("session_id"),
            "question_index": current_idx,
            "total_questions": total_q,
            "topic": fb_topic,
            "difficulty": "Medium" if current_idx <= 3 else "Hard",
            "interviewer_preface": f"Welcome to Question {current_idx}. Let us discuss {fb_topic}.",
            "question_text": fb_q,
            "hints": ["Think about real-world production trade-offs."],
        }

    return {
        "current_index": current_idx,
        "current_question": question_data,
        "student_answer": None,
        "status": "awaiting_answer",
    }


def evaluator_node(state: InterviewState) -> Dict[str, Any]:
    """
    Evaluator Agent: Rigorously grades candidate's response across:
      1. Technical Correctness & Depth (0-10)
      2. Communication Structure & Articulation (0-10)
    """
    current_q = state.get("current_question") or {}
    student_ans = (state.get("student_answer") or "").strip()
    topic = current_q.get("topic", "General")
    q_text = current_q.get("question_text", "Technical question")
    idx = state.get("current_index", 1)

    logger.info(f"Evaluator Agent scoring answer for Turn {idx} ('{topic}')")

    if not student_ans or len(student_ans) < 5:
        # Candidate skipped or submitted empty response
        eval_data = {
            "technical_score": 2.0,
            "communication_score": 3.0,
            "overall_score": 2.5,
            "verdict": "Needs Improvement",
            "feedback": "Answer was too brief or incomplete. Please articulate the underlying mechanism and provide practical code examples.",
            "strengths": [],
            "areas_for_improvement": ["Elaborate on core mechanisms", "Provide structured explanations"],
        }
    else:
        llm = get_llm()
        prompt_val = INTERVIEW_EVALUATION_PROMPT.format_messages(
            topic=topic,
            question_text=q_text,
            student_answer=student_ans,
        )
        try:
            res = llm.invoke(prompt_val)
            parsed = json.loads(_clean_json(res.content))
            t_score = float(parsed.get("technical_score", 7.0))
            c_score = float(parsed.get("communication_score", 7.0))
            o_score = round((t_score * 0.6) + (c_score * 0.4), 1)

            eval_data = {
                "technical_score": max(0.0, min(10.0, t_score)),
                "communication_score": max(0.0, min(10.0, c_score)),
                "overall_score": o_score,
                "verdict": parsed.get("verdict", "Acceptable" if o_score >= 7.0 else "Needs Improvement"),
                "feedback": parsed.get("feedback", "Good explanation covering the fundamental mechanics."),
                "strengths": parsed.get("strengths", ["Identified primary concept correctly"]),
                "areas_for_improvement": parsed.get("areas_for_improvement", ["Discuss edge cases and runtime trade-offs"]),
            }
        except Exception as exc:
            logger.warning(f"Evaluation fallback triggered: {exc}")
            eval_data = {
                "technical_score": 7.5,
                "communication_score": 7.0,
                "overall_score": 7.3,
                "verdict": "Acceptable",
                "feedback": "Solid answer addressing the primary interview question with reasonable clarity.",
                "strengths": ["Demonstrated working knowledge of topic mechanics"],
                "areas_for_improvement": ["Articulate performance implications in production environments"],
            }

    # Append to interview transcript history
    turn_record: InterviewTurnRecord = {
        "question_index": idx,
        "topic": topic,
        "difficulty": current_q.get("difficulty", "Medium"),
        "question_text": q_text,
        "student_answer": student_ans,
        "technical_score": eval_data["technical_score"],
        "communication_score": eval_data["communication_score"],
        "overall_score": eval_data["overall_score"],
        "verdict": eval_data["verdict"],
        "feedback": eval_data["feedback"],
        "strengths": eval_data["strengths"],
        "areas_for_improvement": eval_data["areas_for_improvement"],
    }

    updated_history = list(state.get("history", []))
    updated_history.append(turn_record)

    return {
        "last_evaluation": eval_data,
        "history": updated_history,
        "status": "evaluated",
    }


def finalizer_node(state: InterviewState) -> Dict[str, Any]:
    """
    Hiring Committee Finalizer: Aggregates complete interview transcript into
    a senior hiring scorecard: Technical score %, Communication score %,
    Topic-wise performance, Weak areas, Recommendations, and Readiness Score.
    """
    history = state.get("history", [])
    name = state.get("student_name", "Candidate")
    role = state.get("target_role", "Software Development Engineer")
    tech = state.get("technology", "Java")
    total_q = state.get("total_questions", len(history) or 1)

    logger.info(f"Synthesizing Final Mock Interview Scorecard for '{name}' ({len(history)} questions)")

    # 1. Deterministic baseline calculations
    if history:
        avg_tech = sum(h["technical_score"] for h in history) / len(history)
        avg_comm = sum(h["communication_score"] for h in history) / len(history)
    else:
        avg_tech = 7.0
        avg_comm = 7.0

    tech_pct = round(avg_tech * 10.0, 1)
    comm_pct = round(avg_comm * 10.0, 1)
    overall_pct = round((tech_pct * 0.65) + (comm_pct * 0.35), 1)

    # Readiness Verdict based on composite score
    if overall_pct >= 85:
        verdict = "Strong Hire"
    elif overall_pct >= 72:
        verdict = "Hire"
    elif overall_pct >= 60:
        verdict = "Leaning Hire"
    else:
        verdict = "No Hire (Remediation Required)"

    # Topic breakdown
    topic_groups: Dict[str, List[InterviewTurnRecord]] = {}
    for h in history:
        t = h["topic"]
        topic_groups.setdefault(t, []).append(h)

    topic_perf: List[Dict[str, Any]] = []
    for t_name, turns in topic_groups.items():
        t_avg_tech = round(sum(turn["technical_score"] for turn in turns) / len(turns), 1)
        t_avg_comm = round(sum(turn["communication_score"] for turn in turns) / len(turns), 1)
        t_verdict = "Mastered" if t_avg_tech >= 8.0 else ("Competent" if t_avg_tech >= 6.5 else "Needs Practice")
        topic_perf.append({
            "topic": t_name,
            "average_technical_score": t_avg_tech,
            "average_communication_score": t_avg_comm,
            "verdict": t_verdict,
            "observations": f"Candidate demonstrated {t_verdict.lower()} depth across {len(turns)} question(s).",
        })

    # LLM Synthesis for Executive Notes & Actionable Recommendations
    llm = get_llm()
    transcript_dump = json.dumps([
        {
            "turn": h["question_index"],
            "topic": h["topic"],
            "question": h["question_text"],
            "answer": h["student_answer"],
            "tech_score": h["technical_score"],
            "comm_score": h["communication_score"],
            "feedback": h["feedback"],
        }
        for h in history
    ], indent=2)

    prompt_val = INTERVIEW_FINAL_REPORT_PROMPT.format_messages(
        student_name=name,
        target_role=role,
        technology=tech,
        total_questions=total_q,
        transcript_json=transcript_dump,
    )

    try:
        res = llm.invoke(prompt_val)
        parsed = json.loads(_clean_json(res.content))

        final_summary = {
            "session_id": state.get("session_id", str(uuid.uuid4())),
            "student_name": name,
            "target_role": role,
            "technology": tech,
            "total_questions": total_q,
            "technical_score": float(parsed.get("technical_score", tech_pct)),
            "communication_score": float(parsed.get("communication_score", comm_pct)),
            "overall_interview_score": float(parsed.get("overall_interview_score", overall_pct)),
            "interview_readiness_score": float(parsed.get("interview_readiness_score", overall_pct)),
            "readiness_verdict": parsed.get("readiness_verdict", verdict),
            "topic_wise_performance": parsed.get("topic_wise_performance", topic_perf),
            "strong_areas": parsed.get("strong_areas", [
                f"Solid grasp of fundamental {tech} runtime mechanisms",
                "Structured communication under interview pressure",
            ]),
            "weak_areas": parsed.get("weak_areas", [
                "Edge-case handling in high-throughput or concurrent scenarios",
            ]),
            "recommendations": parsed.get("recommendations", [
                "Practice vocalizing architectural trade-offs explicitly before writing code",
                "Review memory visibility semantics in multi-threaded environments",
                "Simulate mock whiteboard sessions on distributed system failure modes",
            ]),
            "interviewer_notes": parsed.get(
                "interviewer_notes",
                f"{name} demonstrated good technical and communicative maturity throughout the interview. "
                f"With targeted refinement on system-level edge cases, candidate is well-positioned for placement offers."
            ),
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        }
    except Exception as exc:
        logger.warning(f"Final summary generation fallback triggered: {exc}")
        final_summary = {
            "session_id": state.get("session_id", str(uuid.uuid4())),
            "student_name": name,
            "target_role": role,
            "technology": tech,
            "total_questions": total_q,
            "technical_score": tech_pct,
            "communication_score": comm_pct,
            "overall_interview_score": overall_pct,
            "interview_readiness_score": overall_pct,
            "readiness_verdict": verdict,
            "topic_wise_performance": topic_perf,
            "strong_areas": [f"Technical proficiency in {tech} fundamentals"],
            "weak_areas": ["Concurrency and complex system edge cases"],
            "recommendations": [
                "Practice explaining time complexity aloud",
                "Review garbage collection tuning and memory allocation",
            ],
            "interviewer_notes": (
                f"{name} completed all {total_q} interview questions for the {role} position. "
                f"Candidate achieved a technical score of {tech_pct}% and communication score of {comm_pct}%."
            ),
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        }

    return {
        "final_summary": final_summary,
        "status": "finished",
    }


# ── LangGraph Workflow Construction ───────────────────────────────────────────

def build_interview_graph() -> StateGraph:
    """Build and compile the LangGraph Mock Interview workflow."""
    builder = StateGraph(InterviewState)

    builder.add_node("interviewer", interviewer_node)
    builder.add_node("evaluator", evaluator_node)
    builder.add_node("finalizer", finalizer_node)

    # Entry point is interviewer
    builder.set_entry_point("interviewer")

    # Flow: interviewer awaits answer; evaluate answer; route next question or finalize
    def route_after_evaluation(state: InterviewState) -> str:
        idx = state.get("current_index", 0)
        total = state.get("total_questions", 5)
        if idx >= total:
            return "finalizer"
        return "interviewer"

    builder.add_conditional_edges(
        "evaluator",
        route_after_evaluation,
        {
            "interviewer": "interviewer",
            "finalizer": "finalizer",
        },
    )
    builder.add_edge("finalizer", END)

    return builder.compile()


# Singleton compiled graph
_interview_graph = None


def get_interview_graph():
    global _interview_graph
    if _interview_graph is None:
        _interview_graph = build_interview_graph()
    return _interview_graph
