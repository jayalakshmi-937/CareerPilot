"""
CareerPilot — LangGraph Agentic Learning Loop (Phase 5)

Complete Agentic Learning Workflow:

    START
      ↓
    route_entry (check if initial planning or answer evaluation)
      ├─► [Initial Session]
      │     ↓
      │   receive_goal
      │     ↓
      │   analyze_goal
      │     ↓
      │   generate_plan
      │     ↓
      │   select_topic
      │     ↓
      │   retrieve_study_material
      │     ↓
      │   generate_question
      │     ↓
      │    END (returns question & plan to student)
      │
      └─► [Student Answer Submitted]
            ↓
          evaluate_student_answer
            ↓
          detect_weakness_and_update_progress
            ↓
          decide_next_action
            ↓
          [Conditional Router: repeat_weak_topic | next_topic | finish]
            ├─► finish ───────► END
            └─► continue/remediate
                  ↓
                select_topic
                  ↓
                retrieve_study_material
                  ↓
                generate_question
                  ↓
                 END (returns evaluation + next practice question!)
"""

from __future__ import annotations

from typing import Literal
from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes.goal_analyzer import analyze_goal
from agent.nodes.planner import generate_plan
from agent.nodes.retriever import retrieve_study_material
from agent.nodes.topic_selector import select_topic
from agent.nodes.question_generator import generate_question
from agent.nodes.evaluator import evaluate_student_answer
from agent.nodes.weakness_detector import detect_weakness_and_update_progress
from agent.nodes.adaptive_decider import decide_next_action


# ── Entry Node ────────────────────────────────────────────────────────────────

def receive_goal(state: AgentState) -> AgentState:
    """
    Entry node — initializes state variables and conversation history.
    """
    goal = (state.get("student_goal") or "").strip()
    if not goal:
        return {
            **state,
            "error": "Please provide a placement preparation goal.",
            "messages": [{"role": "system", "content": "No goal provided."}],
        }

    return {
        **state,
        "student_goal": goal,
        "messages": list(state.get("messages") or []) + [
            {"role": "system", "content": "CareerPilot session started."},
        ],
        "weak_topics": list(state.get("weak_topics") or []),
        "mastered_topics": list(state.get("mastered_topics") or []),
        "topic_scores": dict(state.get("topic_scores") or {}),
        "progress": dict(state.get("progress") or {
            "completed_topics": [],
            "weak_topics": [],
            "mastered_count": 0,
            "overall_progress_percent": 0,
            "total_questions_attempted": 0,
            "average_score": 0.0,
            "session_count": 1,
        }),
        "retrieved_context": state.get("retrieved_context") or "",
        "error": None,
    }


# ── Conditional Edges & Routing ───────────────────────────────────────────────

def _entry_router(state: AgentState) -> Literal["evaluate_student_answer", "receive_goal"]:
    """
    Route to evaluation if the student submitted an answer,
    otherwise begin standard preparation planning.
    """
    if state.get("student_answer"):
        return "evaluate_student_answer"
    return "receive_goal"


def _should_continue(state: AgentState) -> Literal["continue", "end"]:
    """Short-circuit on error to prevent unnecessary LLM invocations."""
    return "end" if state.get("error") else "continue"


def _adaptive_action_router(state: AgentState) -> Literal["select_topic", "end"]:
    """
    Route based on the decision made by decide_next_action node:
      - finish -> END
      - repeat_weak_topic or next_topic -> select_topic (fetches material & generates question)
    """
    action = state.get("next_action")
    if action == "finish" or state.get("error"):
        return "end"
    return "select_topic"


# ── Graph Factory ──────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Construct and compile the Phase 5 Agentic Learning Loop workflow.
    """
    workflow = StateGraph(AgentState)

    # ── Register all nodes ──────────────────────────────────────────────────
    workflow.add_node("receive_goal", receive_goal)
    workflow.add_node("analyze_goal", analyze_goal)
    workflow.add_node("generate_plan", generate_plan)
    workflow.add_node("select_topic", select_topic)
    workflow.add_node("retrieve_study_material", retrieve_study_material)
    workflow.add_node("generate_question", generate_question)
    workflow.add_node("evaluate_student_answer", evaluate_student_answer)
    workflow.add_node("detect_weakness_and_update_progress", detect_weakness_and_update_progress)
    workflow.add_node("decide_next_action", decide_next_action)

    # ── Entry conditional router ───────────────────────────────────────────
    workflow.set_conditional_entry_point(
        _entry_router,
        {
            "receive_goal": "receive_goal",
            "evaluate_student_answer": "evaluate_student_answer",
        },
    )

    # ── Initial Planning Flow ───────────────────────────────────────────────
    workflow.add_conditional_edges(
        "receive_goal",
        _should_continue,
        {"continue": "analyze_goal", "end": END},
    )
    workflow.add_conditional_edges(
        "analyze_goal",
        _should_continue,
        {"continue": "generate_plan", "end": END},
    )
    workflow.add_conditional_edges(
        "generate_plan",
        _should_continue,
        {"continue": "select_topic", "end": END},
    )
    workflow.add_conditional_edges(
        "select_topic",
        _should_continue,
        {"continue": "retrieve_study_material", "end": END},
    )
    workflow.add_conditional_edges(
        "retrieve_study_material",
        _should_continue,
        {"continue": "generate_question", "end": END},
    )
    workflow.add_edge("generate_question", END)

    # ── Evaluation & Adaptive Learning Loop Flow ────────────────────────────
    workflow.add_edge("evaluate_student_answer", "detect_weakness_and_update_progress")
    workflow.add_edge("detect_weakness_and_update_progress", "decide_next_action")
    workflow.add_conditional_edges(
        "decide_next_action",
        _adaptive_action_router,
        {
            "select_topic": "select_topic",
            "end": END,
        },
    )

    return workflow.compile()


# ── Singleton compiled graph ───────────────────────────────────────────────────
graph = build_graph()
