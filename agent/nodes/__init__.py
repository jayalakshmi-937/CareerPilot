# agent/nodes/__init__.py
"""
Node package for the CareerPilot LangGraph workflow.

Each module in this package implements one node function.
Nodes are imported into agent/graph.py.

Phase 1  — empty (nodes added in Phase 2+)
Phase 2  — goal_analyzer.py, planner.py
Phase 4  — retriever.py
Phase 5  — question_generator.py, evaluator.py, adaptive_decider.py
Phase 6  — progress_updater.py
Phase 7  — mock_interviewer.py
"""
