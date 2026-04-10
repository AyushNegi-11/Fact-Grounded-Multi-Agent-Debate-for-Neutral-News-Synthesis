"""
agents/analyst.py
Agent 0: The Analyst. Reads the article and dynamically decides what two perspectives should debate it.
"""
import json
import re
from langchain_core.messages import HumanMessage
from state import GraphState
from config import get_llm

def analyst_node(state: GraphState) -> dict:
    article = state["original_article"]
    llm = get_llm("MEDIATOR") # Use a smart model to plan
    
    prompt = (
        "You are a News Analyst. Read the following article snippet and identify two "
        "distinct, opposing professional personas (e.g., 'Economist', 'Privacy Advocate', 'Doctor') "
        "who would have a fierce but factual debate about this topic.\n\n"
        f"Article:\n{article[:1500]}\n\n"
        "Return EXACTLY a JSON dictionary like this: "
        '{"persona_a": "First Persona Name", "persona_b": "Second Persona Name"}'
    )
    
    try:
        result = llm.invoke([HumanMessage(content=prompt)])
        data = json.loads(re.search(r"\{.*\}", result.content.strip(), re.DOTALL).group(0))
        return {"persona_a": data.get("persona_a", "Challenger"), "persona_b": data.get("persona_b", "Supporter")}
    except Exception:
        return {"persona_a": "Skeptic", "persona_b": "Defending Authority"}
