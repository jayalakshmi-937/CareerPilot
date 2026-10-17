"""
CareerPilot — Retriever Node (Phase 4)

LangGraph node that queries the ChromaDB vector store for study material
relevant to the student's target technology or current topic.

Reads:  state["current_topic"], state["student_goal"]
Writes: state["retrieved_context"], state["messages"]
"""

from __future__ import annotations

from loguru import logger

from agent.state import AgentState
from rag.retriever import retrieve_context_string


def retrieve_study_material(state: AgentState) -> AgentState:
    """
    LangGraph node: Retrieve relevant study material chunks from ChromaDB.

    Uses state["current_topic"] if available, otherwise falls back to
    state["student_goal"].
    Populates state["retrieved_context"] with retrieved document chunks.
    """
    if state.get("error"):
        logger.warning("retrieve_study_material: upstream error detected — skipping")
        return state

    topic = state.get("current_topic")
    goal = state.get("student_goal", "")

    # Build effective query
    query = topic or goal
    if not query:
        logger.info("retrieve_study_material: no topic or goal to search for.")
        return {**state, "retrieved_context": ""}

    logger.info(f"Retrieving study materials for: '{query}'")

    try:
        context_str = retrieve_context_string(query=query)

        messages = list(state.get("messages") or [])
        if context_str:
            num_chars = len(context_str)
            logger.success(f"Retrieved {num_chars} chars of relevant context from ChromaDB.")
            messages.append({
                "role": "system",
                "content": f"Retrieved study material for '{query}' ({num_chars} characters).",
            })
        else:
            logger.info("No documents matched or vector store is unpopulated.")

        return {
            **state,
            "retrieved_context": context_str,
            "messages": messages,
        }

    except Exception as exc:
        logger.error(f"Error in retrieve_study_material node: {exc}")
        # Non-fatal: if retrieval fails, the agent continues using parametric LLM knowledge
        return {
            **state,
            "retrieved_context": "",
        }
