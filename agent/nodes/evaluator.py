"""
CareerPilot — Answer Evaluator Node (Phase 5)

Rigorously evaluates the student's answer:
  - Technical accuracy & core principle mastery
  - Edge cases, algorithmic efficiency, and syntax
  - Assigns quantitative score (0.0 to 10.0)
  - Identifies specific conceptual weaknesses
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict
from loguru import logger

from agent.llm import get_llm
from agent.prompts import EVALUATOR_PROMPT
from agent.state import AgentState


def _clean_json(raw: Any) -> str:
    """Extract string and strip accidental markdown fences the model might add."""
    if isinstance(raw, list):
        parts = []
        for item in raw:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif hasattr(item, "text"):
                parts.append(getattr(item, "text"))
            else:
                parts.append(str(item))
        text = "\n".join(parts)
    else:
        text = str(raw)

    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def evaluate_student_answer(state: AgentState) -> AgentState:
    """
    LangGraph node: Evaluate the student's submitted answer.

    Reads:  state["student_answer"], state["current_question"], state["question_details"], state["retrieved_context"]
    Writes: state["evaluation"], state["messages"]
    """
    if state.get("error"):
        return state

    student_ans = (state.get("student_answer") or "").strip()
    question = state.get("current_question") or "Explain technical concepts."
    q_details = state.get("question_details") or {}
    topic = q_details.get("topic") or state.get("current_topic") or "General Technical"
    expected = ", ".join(q_details.get("expected_key_points", [])) or "Correct technical explanation and examples."
    study_mat = state.get("retrieved_context") or "Standard reference notes."

    if not student_ans:
        logger.warning("Empty student answer provided for evaluation.")
        eval_dict = {
            "score": 0.0,
            "is_passed": False,
            "verdict": "Inadequate",
            "feedback": "No answer was provided. Attempting to answer is essential for placement readiness.",
            "strengths": [],
            "areas_for_improvement": ["Please write an answer before submitting."],
            "ideal_answer": q_details.get("sample_ideal_answer", "Review the study notes for full solution."),
            "identified_weakness": topic,
            "remediation_advice": f"Review fundamental concepts for {topic}.",
        }
        return {**state, "evaluation": eval_dict}

    logger.info(f"Evaluating answer for '{topic}' | answer_len={len(student_ans)}")

    try:
        llm = get_llm()
        chain = EVALUATOR_PROMPT | llm
        response = chain.invoke({
            "topic": topic,
            "question": question,
            "expected_points": expected,
            "student_answer": student_ans,
            "study_material": study_mat,
        })

        raw_content = response.content if hasattr(response, "content") else str(response)
        cleaned = _clean_json(raw_content)
        eval_data: Dict[str, Any] = json.loads(cleaned)

        score = float(eval_data.get("score", 5.0))
        eval_data["score"] = score
        eval_data["is_passed"] = score >= 7.0

        logger.success(f"Evaluation complete for '{topic}' → Score: {score}/10 (Verdict: {eval_data.get('verdict')})")

        messages = list(state.get("messages") or [])
        messages.append({"role": "human", "content": student_ans})
        messages.append({
            "role": "ai",
            "content": f"📊 **Evaluation ({score}/10 — {eval_data.get('verdict')}):**\n\n{eval_data.get('feedback')}",
        })

        return {
            **state,
            "evaluation": eval_data,
            "messages": messages,
        }

    except Exception as exc:
        logger.error(f"Failed to evaluate answer: {exc}")
        fallback_eval = {
            "score": 6.0,
            "is_passed": False,
            "verdict": "Needs Improvement",
            "feedback": f"Evaluator encountered an error ({exc}). Answer received: review core mechanics for {topic}.",
            "strengths": ["Submitted answer"],
            "areas_for_improvement": ["Elaborate on edge cases and code syntax."],
            "ideal_answer": "Refer to official placement documentation.",
            "identified_weakness": topic,
            "remediation_advice": "Practice explaining this concept aloud.",
        }
        return {**state, "evaluation": fallback_eval}
