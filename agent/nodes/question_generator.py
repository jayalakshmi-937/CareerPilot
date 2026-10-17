"""
CareerPilot — Question Generator Node (Phase 5)

Generates placement-ready questions grounded in study materials:
  - Conceptual interview questions
  - Code challenges & implementations
  - Debugging scenarios
Tailors difficulty based on whether this is a remedial question for a weak topic.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict
from loguru import logger

from agent.llm import get_llm
from agent.prompts import QUESTION_GENERATOR_PROMPT
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


def generate_question(state: AgentState) -> AgentState:
    """
    LangGraph node: Generate an interview question on the current topic.

    Reads:  state["current_topic"], state["student_profile"], state["retrieved_context"], state["weak_topics"]
    Writes: state["current_question"], state["question_details"], state["messages"]
    """
    if state.get("error"):
        return state

    topic = state.get("current_topic") or "General Programming Concepts"
    profile = state.get("student_profile") or {}
    tech = profile.get("technology", "Software Engineering")
    diff = profile.get("difficulty_level", "Intermediate")
    weak_topics = state.get("weak_topics") or []
    is_remedial = topic in weak_topics

    study_mat = state.get("retrieved_context") or "Standard industry placement curriculum."

    logger.info(f"Generating interview question | topic='{topic}' | is_remedial={is_remedial}")

    try:
        llm = get_llm()
        chain = QUESTION_GENERATOR_PROMPT | llm
        response = chain.invoke({
            "technology": tech,
            "topic": topic,
            "difficulty": diff if not is_remedial else "Beginner to Intermediate",
            "is_remedial": is_remedial,
            "study_material": study_mat,
        })

        raw_content = response.content if hasattr(response, "content") else str(response)
        cleaned = _clean_json(raw_content)
        q_dict: Dict[str, Any] = json.loads(cleaned)

        question_text = q_dict.get("question_text", f"Explain the core mechanics and best practices of {topic}.")
        q_type = q_dict.get("question_type", "Conceptual")

        logger.success(f"Generated {q_type} question on '{topic}'")

        messages = list(state.get("messages") or [])
        messages.append({
            "role": "ai",
            "content": f"📝 **Question ({q_type} - {topic}):**\n\n{question_text}",
        })

        return {
            **state,
            "current_question": question_text,
            "question_details": q_dict,
            "messages": messages,
        }

    except Exception as exc:
        logger.error(f"Failed to generate question: {exc}")
        fallback_q = f"Explain the key principles, common pitfalls, and real-world usage of {topic} in {tech}."
        return {
            **state,
            "current_question": fallback_q,
            "question_details": {
                "question_text": fallback_q,
                "question_type": "Conceptual",
                "topic": topic,
                "difficulty": diff,
                "expected_key_points": [f"Definition of {topic}", "Internal mechanics", "Real-world trade-offs"],
            },
        }
