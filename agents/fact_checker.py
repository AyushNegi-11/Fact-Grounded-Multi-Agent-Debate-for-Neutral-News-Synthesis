"""
agents/fact_checker.py
Algorithmic Node (No LLM): The Fact Checker.
Uses spaCy for Entity overlap, DeBERTa for Contradiction, and generates Highlighted HTML.
"""
import numpy as np
from state import GraphState
from config import load_spacy_model, load_nli_model
from utils.tools import perform_web_search

def fact_checker_node(state: GraphState) -> dict:
    article, sum_a, sum_b = state["original_article"], state.get("agent_a_summary", ""), state.get("agent_b_summary", "")
    nlp = load_spacy_model()
    nli_model = load_nli_model()
    iteration = state.get("iteration", 0)
    
    # 1. Extract Named Entities from the Original Article
    doc_art = nlp(article)
    article_ents = {ent.text.lower() for ent in doc_art.ents if ent.label_ not in ["CARDINAL", "ORDINAL"]}
    
    def score_and_highlight(summary_text: str) -> tuple[float, str]:
        if not summary_text:
            return 0.0, ""
            
        doc_sum = nlp(summary_text)
        cited = 0
        total_ents = 0
        highlighted_words = []
        last_idx = 0
        
        # Iterate over characters and inject HTML tags around entities
        for ent in doc_sum.ents:
            if ent.label_ in ["CARDINAL", "ORDINAL"]:
                continue
            total_ents += 1
            highlighted_words.append(summary_text[last_idx:ent.start_char])
            
            # Check if this entity was actually in the original article!
            if ent.text.lower() in article_ents:
                cited += 1
                highlighted_words.append(f'<span style="background-color: #d4edda; color: #155724; padding: 2px; border-radius: 3px;" title="Verified Entity">{ent.text}</span>')
            else:
                # Multi-Source Fact Check
                search_res = perform_web_search(ent.text + " " + article[:50])
                if ent.text.lower() in search_res.lower() and "No external" not in search_res:
                    cited += 1
                    highlighted_words.append(f'<span style="background-color: #cce5ff; color: #004085; padding: 2px; border-radius: 3px;" title="Verified via Web">{ent.text}</span>')
                else:
                    highlighted_words.append(f'<span style="background-color: #f8d7da; color: #721c24; padding: 2px; border-radius: 3px;" title="Unverified/Hallucinated">{ent.text}</span>')
            last_idx = ent.end_char
            
        highlighted_words.append(summary_text[last_idx:])
        
        base_score = (cited / total_ents) if total_ents > 0 else 0.5
        
        # 2. Check NLI Contradiction
        penalty = 0.0
        if nli_model:
            sentences = [sent.text for sent in doc_sum.sents]
            for sent in sentences:
                logits = nli_model.predict([[article[:1000], sent]])
                probs = np.exp(logits) / np.sum(np.exp(logits))
                if probs[0][0] > 0.85: # Contradiction label
                    penalty = 0.5
                    break
                    
        final_score = max(0.0, min(1.0, base_score * (1.0 - penalty)))
        return round(float(final_score), 3), "".join(highlighted_words)

    a_score, html_a = score_and_highlight(sum_a)
    b_score, html_b = score_and_highlight(sum_b)
    
    return {
        "a_score": a_score, 
        "b_score": b_score, 
        "highlighted_text_a": html_a,
        "highlighted_text_b": html_b,
        "iteration": iteration + 1
    }
