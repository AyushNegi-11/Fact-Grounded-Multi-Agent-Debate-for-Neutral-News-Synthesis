"""
agents/fallacy_checker.py
Agent Node: The Fallacy Checker.
Penalizes Authority Scores if Ad Hominem or Emotional Appeals are detected.
"""
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from state import GraphState
from config import get_llm

def fallacy_checker_node(state: GraphState) -> dict:
    sum_a = state.get("agent_a_summary", "")
    sum_b = state.get("agent_b_summary", "")
    a_score, b_score = state.get("a_score", 0.0), state.get("b_score", 0.0)
    llm = get_llm("MEDIATOR", max_tokens=150)
    
    class FallacyCheckResult(BaseModel):
        has_fallacy: bool
        reasoning: str
        penalty: float

    def check_fallacy(text: str, current_score: float) -> tuple[float, str]:
        prompt = (
            f"Analyze this argument for severe logical fallacies (Ad Hominem, Straw Man). "
            f"Argument: {text}\n"
            "If it contains a severe fallacy, set 'has_fallacy' to true, explain the 'reasoning', "
            "and set 'penalty' to 0.2. Otherwise, set 'penalty' to 0.0 and 'has_fallacy' to false."
        )
        try:
            structured_llm = llm.with_structured_output(FallacyCheckResult)
            res = structured_llm.invoke([HumanMessage(content=prompt)])
            if res.has_fallacy:
                return round(max(0.0, current_score - res.penalty), 3), res.reasoning
        except Exception:
            pass
        return current_score, ""

    new_a, critique_a = check_fallacy(sum_a, a_score)
    new_b, critique_b = check_fallacy(sum_b, b_score)
    
    # Store the log after ALL score penalties are calculated
    log = state.get("debate_log", [])
    log.append({
        "iteration": state.get("iteration", 1),
        "a_score": new_a,
        "b_score": new_b,
        "highlighted_text_a": state.get("highlighted_text_a", ""),
        "highlighted_text_b": state.get("highlighted_text_b", "")
    })
    
    return {"a_score": new_a, "b_score": new_b, "debate_log": log, "critique_a": critique_a, "critique_b": critique_b}
