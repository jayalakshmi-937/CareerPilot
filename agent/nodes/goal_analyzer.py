"""
CareerPilot — Goal Analyzer Node (Phase 2)

Analyses the raw student goal string and extracts structured profile data:
  - technology (Java, Python, etc.)
  - role (Software Engineer, etc.)
  - days_available (int)
  - difficulty_level (Beginner | Intermediate | Advanced)
  - topics (list of core topics)
  - interview_type (Technical | HR | Both)
  - summary (friendly 1-sentence summary)

This structured profile drives every downstream node.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from loguru import logger

from agent.llm import get_llm
from agent.prompts import GOAL_ANALYZER_PROMPT
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
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def analyze_goal(state: AgentState) -> AgentState:
    """
    LangGraph node: Analyse the student's raw goal and populate student_profile.

    Reads:  state["student_goal"]
    Writes: state["student_profile"], state["session_id"], state["messages"]
    """
    goal = state.get("student_goal", "").strip()
    if not goal:
        logger.warning("analyze_goal: empty student_goal received")
        return {
            **state,
            "error": "No goal provided. Please describe your placement goal.",
        }

    logger.info(f"Analysing goal: '{goal[:80]}…'" if len(goal) > 80 else f"Analysing goal: '{goal}'")

    try:
        llm = get_llm()
        chain = GOAL_ANALYZER_PROMPT | llm
        response = chain.invoke({"student_goal": goal})
        raw_content = response.content if hasattr(response, "content") else str(response)

        cleaned = _clean_json(raw_content)
        profile: dict[str, Any] = json.loads(cleaned)

        # Ensure required keys are present with sensible defaults
        profile.setdefault("technology", "General Software Engineering")
        profile.setdefault("role", "Software Engineer")
        profile.setdefault("days_available", 7)
        profile.setdefault("difficulty_level", "Intermediate")
        profile.setdefault("topics", [])
        profile.setdefault("interview_type", "Technical")
        profile.setdefault("summary", goal)

        session_id = str(uuid.uuid4())
        logger.success(f"Goal analysed → technology={profile['technology']}, days={profile['days_available']}")

        # Append to conversation history
        messages = list(state.get("messages") or [])
        messages.append({"role": "human", "content": goal})
        messages.append({
            "role": "ai",
            "content": f"Goal understood! You want to prepare for {profile['technology']} in {profile['days_available']} days. Let me create your personalised plan.",
        })

        return {
            **state,
            "student_profile": profile,
            "session_id": session_id,
            "messages": messages,
            "current_topic": profile["topics"][0] if profile["topics"] else profile["technology"],
            "error": None,
        }

    except json.JSONDecodeError as exc:
        logger.error(f"Failed to parse goal analysis JSON: {exc}")
        return {
            **state,
            "error": f"Goal analysis parsing error: {exc}. Please rephrase your goal.",
        }
    except Exception as exc:
        logger.error(f"Goal analysis LLM error: {exc}")
        return {
            **state,
            "error": f"Goal analysis failed: {str(exc)}",
        }
