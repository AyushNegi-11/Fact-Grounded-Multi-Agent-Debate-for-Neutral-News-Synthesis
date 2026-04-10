"""
agents/supporter.py
Agent B: The Supporter.
Defends the article against the Challenger.
"""
from langchain_core.messages import HumanMessage
from state import GraphState
from config import get_llm

def supporter_node(state: GraphState) -> dict:
    article = state["original_article"]
    persona = state.get("persona_b", "Supporter")
    iteration = state.get("iteration", 0)
    llm = get_llm("B") # Heterogeneous model (e.g., Gemma)
    
    challenger_arg = state.get("agent_a_summary", "")
    
    prompt = f"You are playing the role of a {persona}.\n\n"
    prompt += f"Original Article:\n{article[:2000]}\n\n"
    prompt += f"Challenger ({state.get('persona_a', 'Opponent')}) argued:\n{challenger_arg}\n\n"
    
    if iteration > 0:
        b_score = state.get("b_score", 0.0)
        prompt += f"Your previous factual score was low ({b_score}). You must use MORE specific names and dates from the article.\n\n"
        
        critique = state.get("critique_b")
        if critique:
            prompt += f"Additionally, your previous argument was rejected by the Fallacy Checker for the following reason:\n{critique}\n"
            prompt += "Please fix this logical error and try again.\n\n"

    prompt += (
        "Write a 150-word defensive argument supporting the article's core claims "
        "and countering the Challenger. Cite specific facts. Plain text only. No bullet points.\n"
        "IMPORTANT: DO NOT output any internal thoughts, reasoning, or introductions (e.g., 'Okay, let's tackle this...'). "
        "Return ONLY the final, polished argument."
    )
    
    result = llm.invoke([HumanMessage(content=prompt)])
    return {"agent_b_summary": result.content.strip()}
