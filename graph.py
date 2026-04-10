"""
graph.py
Compiles the LangGraph StateMachine and manages edges/routing.
"""
import functools
from langgraph.graph import StateGraph, END

from state import GraphState
from agents.analyst import analyst_node
from agents.challenger import challenger_node
from agents.supporter import supporter_node
from agents.fact_checker import fact_checker_node
from agents.fallacy_checker import fallacy_checker_node
from agents.mediator import mediator_node

def router_logic(state: GraphState) -> str:
    """Decides where to go next based on scores and iteration."""
    it = state.get("iteration", 1)
    a_s, b_s = state.get("a_score", 0.0), state.get("b_score", 0.0)
    
    # If scores are terrible and we haven't looped too many times, loop back to Challenger
    if (a_s < 0.35 or b_s < 0.35) and it <= 1:
        return "rewrite"
    return "mediator"

@functools.lru_cache(maxsize=1)
def build_graph():
    """Builds the LangGraph state machine. Cached so Streamlit doesn't rebuild it."""
    builder = StateGraph(GraphState)
    
    builder.add_node("analyst", analyst_node)
    builder.add_node("challenger", challenger_node)
    builder.add_node("supporter", supporter_node)
    builder.add_node("fact_checker", fact_checker_node)
    builder.add_node("fallacy_checker", fallacy_checker_node)
    builder.add_node("mediator", mediator_node)
    
    # Draw the edges
    builder.set_entry_point("analyst")
    builder.add_edge("analyst", "challenger")
    builder.add_edge("challenger", "supporter")
    builder.add_edge("supporter", "fact_checker")
    builder.add_edge("fact_checker", "fallacy_checker")
    
    # Conditional Routing
    builder.add_conditional_edges("fallacy_checker", router_logic, {
        "rewrite": "challenger",
        "mediator": "mediator"
    })
    
    builder.add_edge("mediator", END)
    
    # Compile the graph (stateless, no memory checkpointing needed since we removed pause)
    graph = builder.compile()
    return graph
