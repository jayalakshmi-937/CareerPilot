"""
CareerPilot — Planner Node (Phase 2)

Uses the student profile (populated by goal_analyzer) to generate a
detailed, personalised, day-by-day preparation plan via Gemini.

The plan is stored as a structured dict in state["generated_plan"] and
also as a formatted Markdown string for the frontend to display.
"""

from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger

from agent.llm import get_llm
from agent.prompts import PLANNER_PROMPT
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


def _plan_to_markdown(plan: dict[str, Any], profile: dict[str, Any]) -> str:
    """Convert the structured plan dict to a clean Markdown summary."""
    lines: list[str] = []
    lines.append(f"# 🎯 {plan.get('plan_title', 'Your Preparation Plan')}")
    lines.append(f"\n**Technology:** {profile.get('technology', '—')} | "
                 f"**Role:** {profile.get('role', '—')} | "
                 f"**Level:** {profile.get('difficulty_level', '—')}\n")
    lines.append(f"## Overview\n{plan.get('overview', '')}\n")

    lines.append("## 📅 Day-by-Day Schedule\n")
    for day in plan.get("daily_schedule", []):
        lines.append(f"### Day {day['day']} — {day['focus']}")
        lines.append(f"**⏱ Estimated:** {day.get('estimated_hours', 2)} hrs")
        if day.get("goals"):
            lines.append("**Goals:**")
            for g in day["goals"]:
                lines.append(f"  - {g}")
        if day.get("topics"):
            lines.append("**Topics:**")
            for t in day["topics"]:
                lines.append(f"  - {t}")
        lines.append(f"**Practice:** {day.get('practice', '')}\n")

    if plan.get("key_resources"):
        lines.append("## 📚 Key Resources")
        for r in plan["key_resources"]:
            lines.append(f"  - {r}")
        lines.append("")

    if plan.get("success_tips"):
        lines.append("## 💡 Success Tips")
        for tip in plan["success_tips"]:
            lines.append(f"  - {tip}")
        lines.append("")

    if plan.get("readiness_indicators"):
        lines.append("## ✅ You're Ready When…")
        for ind in plan["readiness_indicators"]:
            lines.append(f"  - {ind}")

    return "\n".join(lines)


def generate_plan(state: AgentState) -> AgentState:
    """
    LangGraph node: Generate a personalised preparation plan.

    Reads:  state["student_goal"], state["student_profile"]
    Writes: state["generated_plan"], state["messages"]
    """
    profile = state.get("student_profile")
    if not profile:
        logger.warning("generate_plan: no student_profile in state — skipping")
        return {
            **state,
            "error": "Cannot generate plan without an analysed profile. Try again.",
        }

    # If goal analysis failed earlier, skip
    if state.get("error"):
        logger.warning("generate_plan: upstream error detected — skipping")
        return state

    goal = state.get("student_goal", "")
    technology = profile.get("technology", "Software Engineering")
    role = profile.get("role", "Software Engineer")
    difficulty_level = profile.get("difficulty_level", "Intermediate")
    days_available = profile.get("days_available", 7)
    topics = ", ".join(profile.get("topics", []))
    interview_type = profile.get("interview_type", "Technical")

    # Extract retrieved context from RAG if present
    study_material = (
        state.get("retrieved_context")
        or "No custom study materials provided. Use industry-standard placement preparation guidelines."
    )

    try:
        llm = get_llm()
        chain = PLANNER_PROMPT | llm
        response = chain.invoke({
            "student_goal": goal,
            "technology": technology,
            "role": role,
            "difficulty_level": difficulty_level,
            "days_available": days_available,
            "topics": topics,
            "interview_type": interview_type,
            "study_material": study_material,
        })
        raw_content = response.content if hasattr(response, "content") else str(response)

        cleaned = _clean_json(raw_content)
        plan_dict: dict[str, Any] = json.loads(cleaned)

        # Build human-readable Markdown version
        plan_markdown = _plan_to_markdown(plan_dict, profile)

        # Determine first topic to study and recommended next task
        daily = plan_dict.get("daily_schedule", [])
        first_topic = daily[0]["focus"] if daily else profile.get("technology", "")
        recommended_task = (
            daily[0].get("practice")
            if daily and daily[0].get("practice")
            else f"Review core fundamentals of {first_topic}."
        )

        logger.success(f"Plan generated: {plan_dict.get('plan_title', 'untitled')}")

        # Append assistant message to history
        messages = list(state.get("messages") or [])
        messages.append({
            "role": "ai",
            "content": f"✅ Your personalised {days_available}-day plan is ready! We start with: **{first_topic}**.",
        })

        return {
            **state,
            "generated_plan": {
                "structured": plan_dict,
                "markdown": plan_markdown,
            },
            "current_topic": first_topic,
            "recommended_next_task": recommended_task,
            "messages": messages,
            "next_action": "continue",
            "error": None,
        }

    except json.JSONDecodeError as exc:
        logger.error(f"Failed to parse plan JSON: {exc}")
        return {**state, "error": f"Plan generation parsing error: {exc}"}
    except Exception as exc:
        logger.error(f"Plan generation LLM error: {exc}")
        return {**state, "error": f"Plan generation failed: {str(exc)}"}
