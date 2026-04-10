"""
agents/mediator.py
Agent C: The Mediator.
Writes the final synthesis.
"""
from langchain_core.messages import HumanMessage
from textblob import TextBlob
from rouge_score import rouge_scorer
from state import GraphState
from config import get_llm

def mediator_node(state: GraphState) -> dict:
    a_score, b_score = state.get("a_score", 0.0), state.get("b_score", 0.0)
    a_sum, b_sum = state.get("agent_a_summary", ""), state.get("agent_b_summary", "")
    llm = get_llm("MEDIATOR", max_tokens=400)
    
    prompt = (
        "You are the Mediator. Write a 200-word final, neutral synthesis of the debate.\n\n"
        f"Persona A ({state.get('persona_a', 'A')}) Score: {a_score}\nArgument:\n{a_sum}\n\n"
        f"Persona B ({state.get('persona_b', 'B')}) Score: {b_score}\nArgument:\n{b_sum}\n\n"
        "Write a balanced journalistic summary. Proportionally weight the facts of the higher scoring agent. "
        "Do not declare a 'winner'. Output plain paragraphs."
    )
    
    result = llm.invoke([HumanMessage(content=prompt)])
    final_text = result.content.strip()
    
    # 1. ROUGE Score (Information Conservation)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    r_scores = scorer.score(state["original_article"], final_text)
    synthesis_rouge = {
        "rouge1": round(r_scores['rouge1'].fmeasure, 3),
        "rougeL": round(r_scores['rougeL'].fmeasure, 3)
    }
    
    # 2. Sentiment Neutrality (Bias Reduction)
    article_blob = TextBlob(state["original_article"])
    syn_blob = TextBlob(final_text)
    synthesis_neutral = {
        "original_polarity": round(article_blob.sentiment.polarity, 3),
        "synthesis_polarity": round(syn_blob.sentiment.polarity, 3)
    }

    # 3. Debate Influence (Whose points did the mediator use?)
    score_a = scorer.score(a_sum, final_text)['rougeL'].fmeasure
    score_b = scorer.score(b_sum, final_text)['rougeL'].fmeasure
    total_influence = score_a + score_b if (score_a + score_b) > 0 else 1.0
    influence = {
        "challenger": round((score_a / total_influence) * 100, 1),
        "supporter": round((score_b / total_influence) * 100, 1)
    }

    return {
        "final_summary": final_text,
        "synthesis_rouge": synthesis_rouge,
        "synthesis_neutral": synthesis_neutral,
        "debate_influence": influence
    }
