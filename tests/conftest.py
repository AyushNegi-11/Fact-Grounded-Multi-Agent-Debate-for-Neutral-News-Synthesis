# conftest.py — shared helpers for all tests
from app import GraphState


def make_state(**overrides) -> GraphState:
    """Return a minimal valid GraphState with optional field overrides."""
    base: GraphState = {
        "original_article": "",
        "agent_a_round1": "",
        "agent_b_round1": "",
        "agent_a_summary": "",
        "agent_b_summary": "",
        "a_score": 0.0,
        "b_score": 0.0,
        "final_summary": "",
        "iteration": 0,
        "transparency_report": {},
    }
    base.update(overrides)
    return base
