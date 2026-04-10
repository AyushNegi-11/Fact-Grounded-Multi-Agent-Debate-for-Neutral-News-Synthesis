"""
state.py
Defines the shared state dictionary passed between agents in the LangGraph pipeline.
"""
from typing import TypedDict

class GraphState(TypedDict):
    original_article: str      # The source text being debated
    persona_a: str             # Dynamically generated persona for Challenger
    persona_b: str             # Dynamically generated persona for Supporter
    agent_a_summary: str       # Challenger's argument
    agent_b_summary: str       # Supporter's argument
    a_score: float             # Authority Score for Challenger (0.0 to 1.0)
    b_score: float             # Authority Score for Supporter (0.0 to 1.0)
    iteration: int             # Tracks how many refinement loops we've done
    highlighted_text_a: str    # HTML string with verified/hallucinated entities colored
    highlighted_text_b: str    # HTML string with verified/hallucinated entities colored
    jury_feedback: str         # Captured human-in-the-loop feedback (if ever needed again)
    final_summary: str         # The final neutral synthesis by the Mediator
    debate_log: list           # History of all rounds
    synthesis_rouge: dict      # ROUGE metrics for Final Synthesis
    synthesis_neutral: dict    # Neutrality metrics (sentiment)
    debate_influence: dict     # Influence % of each agent on the final synthesis
    critique_a: str            # Chain-of-Thought Critique from Fallacy Checker for A
    critique_b: str            # Chain-of-Thought Critique from Fallacy Checker for B
