"""
agents/challenger.py
Agent A: The Challenger.
Uses Web Search to bring in external facts and challenges the article.
"""
from langchain_core.messages import HumanMessage
from state import GraphState
from config import get_llm
from utils.tools import perform_web_search

def challenger_node(state: GraphState) -> dict:
    article = state["original_article"]
    persona = state.get("persona_a", "Challenger")
    iteration = state.get("iteration", 0)
    llm = get_llm("A") # Heterogeneous model (e.g., Mixtral)
    
    # Run a web search based on the article's first 100 characters to get external context
    search_context = perform_web_search(article[:100] + " controversy criticism")
    
    prompt = f"You are playing the role of a {persona}.\n\n"
    prompt += f"Original Article:\n{article[:2000]}\n\n"
    prompt += f"External Web Search Context:\n{search_context}\n\n"
    
    if iteration > 0:
        a_score = state.get("a_score", 0.0)
        prompt += f"Your previous factual score was low ({a_score}). You must use MORE specific names and dates from the article.\n\n"
        
        critique = state.get("critique_a")
        if critique:
            prompt += f"Additionally, your previous argument was rejected by the Fallacy Checker for the following reason:\n{critique}\n"
            prompt += "Please fix this logical error and try again.\n\n"
            
        prompt += f"Opponent's argument:\n{state.get('agent_b_summary', '')}\n\n"
        
    prompt += (
        "Write a 150-word critical argument challenging the article's main claims. "
        "You must cite specific facts from both the article and the external context. "
        "Plain text only. No bullet points.\n"
        "IMPORTANT: DO NOT output any internal thoughts, reasoning, or introductions (e.g., 'Okay, let's tackle this...'). "
        "Return ONLY the final, polished argument."
    )
    
    result = llm.invoke([HumanMessage(content=prompt)])
    return {"agent_a_summary": result.content.strip()}
