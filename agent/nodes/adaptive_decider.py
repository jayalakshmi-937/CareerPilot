"""
CareerPilot — Adaptive Decider Node (Phase 5)

The central agentic brain for dynamic decision-making:
  - Evaluates recent performance score and weakness flags
  - DECISION A: If student struggled (score < 7.0):
      -> Directs remediation on the weak topic
      -> Schedules additional practice
  - DECISION B: If student succeeded (score >= 7.0) and unmastered topics remain:
      -> Advances to the next syllabus topic
  - DECISION C: If all topics are completed/mastered:
      -> Transitions to completion / mock interview readiness
"""

from __future__ import annotations

from typing import List
from loguru import logger

from agent.state import AgentState


def decide_next_action(state: AgentState) -> AgentState:
    """
    LangGraph node: Make an adaptive pedagogical decision based on student performance.

    Reads:  state["evaluation"], state["weak_topics"], state["mastered_topics"], state["current_topic"], state["student_profile"]
    Writes: state["next_action"], state["recommended_next_task"], state["messages"]
    """
    if state.get("error"):
        return state

    eval_data = state.get("evaluation") or {}
    score = float(eval_data.get("score", 0.0))
    current_topic = state.get("current_topic") or "General Programming"
    weak_topics: List[str] = list(state.get("weak_topics") or [])
    mastered_topics: List[str] = list(state.get("mastered_topics") or [])
    profile = state.get("student_profile") or {}
    all_topics: List[str] = profile.get("topics", [])

    messages = list(state.get("messages") or [])

    # DECISION A: Student struggled on current topic (score < 7.0)
    if score < 7.0:
        next_action = "repeat_weak_topic"
        rec_task = (
            f"Remediate weakness in '{current_topic}'. "
            f"Review reference notes on {eval_data.get('identified_weakness') or current_topic} "
            f"and attempt another practice challenge."
        )
        logger.info(f"Adaptive Decider: Weak performance ({score}/10). Scheduling remediation for '{current_topic}'.")
        messages.append({
            "role": "ai",
            "content": f"🔄 **Adaptive Decision:** Score ({score}/10) indicates gaps in **{current_topic}**. Re-routing to targeted practice.",
        })

    # DECISION B: Student passed and there are more topics left
    else:
        remaining_topics = [t for t in all_topics if t not in mastered_topics and t != current_topic]

        if remaining_topics:
            next_topic = remaining_topics[0]
            next_action = "next_topic"
            rec_task = f"Advance to next topic: Study '{next_topic}' and practice placement questions."
            logger.info(f"Adaptive Decider: Passed '{current_topic}' ({score}/10). Advancing to '{next_topic}'.")
            messages.append({
                "role": "ai",
                "content": f"🎉 **Adaptive Decision:** Excellent! **{current_topic}** mastered. Advancing to **{next_topic}**.",
            })

        # DECISION C: All curriculum topics mastered!
        else:
            next_action = "finish"
            rec_task = "🎉 Congratulations! All placement preparation topics mastered. Take a full Mock Interview."
            logger.success("Adaptive Decider: All topics mastered. Goal achieved.")
            messages.append({
                "role": "ai",
                "content": "🏆 **Adaptive Decision:** You have completed all scheduled topics! Ready for placement interviews.",
            })

    return {
        **state,
        "next_action": next_action,
        "recommended_next_task": rec_task,
        "messages": messages,
    }
