"""
CareerPilot — Topic Selector Node (Phase 5)

Selects the next learning or testing topic adaptively based on:
  1. Priority 1: Unresolved weak topics (score < 7.0 previously).
  2. Priority 2: Next scheduled topic from profile/plan not yet mastered.
  3. Priority 3: Fallback to general technology topic.
"""

from __future__ import annotations

from typing import List
from loguru import logger

from agent.state import AgentState


def select_topic(state: AgentState) -> AgentState:
    """
    LangGraph node: Select the next topic adaptively based on performance.

    Reads:  state["weak_topics"], state["mastered_topics"], state["student_profile"]
    Writes: state["current_topic"], state["next_action"]
    """
    if state.get("error"):
        return state

    profile = state.get("student_profile") or {}
    all_topics: List[str] = profile.get("topics", [])
    weak_topics: List[str] = list(state.get("weak_topics") or [])
    mastered_topics: List[str] = list(state.get("mastered_topics") or [])

    # Priority 1: If there are identified weak topics, remediate the first one
    if weak_topics:
        selected = weak_topics[0]
        logger.info(f"Adaptive selector prioritized weak topic for remediation: '{selected}'")
        return {
            **state,
            "current_topic": selected,
            "next_action": "repeat_weak_topic",
        }

    # Priority 2: Find the first unmastered topic in the curriculum
    for t in all_topics:
        if t not in mastered_topics:
            logger.info(f"Adaptive selector advanced to next unmastered topic: '{t}'")
            return {
                **state,
                "current_topic": t,
                "next_action": "continue",
            }

    # Priority 3: All topics mastered!
    fallback = all_topics[0] if all_topics else profile.get("technology", "Core Technical Review")
    logger.info("All topics mastered or completed in curriculum.")
    return {
        **state,
        "current_topic": fallback,
        "next_action": "finish",
    }
