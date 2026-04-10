"""
app.py
The main Streamlit front-end entry point.
Run with: py -m streamlit run app.py
"""
import os
import warnings
import logging
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import streamlit as st
from utils.tools import auto_fetch_text
from graph import build_graph

def main():
    st.set_page_config(page_title="Dialectic AI", layout="wide")
    
    if not os.getenv("GROQ_API_KEY"):
        st.error("Missing GROQ_API_KEY in .env file!")
        st.stop()
        
    st.title("Dialectic AI: Agentic News Neutralizer")
    st.caption("A dynamic, RAG-enabled debate with zero-hallucination NLP checks.")
    
    # Session state initialization for UI flow
    if "debate_started" not in st.session_state:
        st.session_state.debate_started = False
    
    # Input Area
    article_input = st.text_area("Paste Article Here (or a URL):", height=150)
    
    if st.button("Start Debate", type="primary"):
        if not article_input:
            st.warning("Please paste an article or URL.")
            st.stop()
            
        # Optional URL extraction
        if "processed_article" not in st.session_state or st.session_state.get("last_input") != article_input:
            with st.spinner("Processing input (fetching URL if needed)..."):
                st.session_state.processed_article = auto_fetch_text(article_input)
                st.session_state.last_input = article_input
                
        st.session_state.debate_started = True
        
    if not st.session_state.debate_started:
        st.stop()
        
    actual_article = st.session_state.processed_article
        
    # --- Execute Graph ---
    graph = build_graph()
    
    # Stream the graph updates
    st.write("---")
    status_text = st.empty()
    
    # Run the graph!
    final_end_state = None
    for event in graph.stream({"original_article": actual_article}, stream_mode="values"):
        final_end_state = event
        if "final_summary" in event and event["final_summary"]:
            status_text.success("Synthesis Complete.")
        else:
            status_text.info(f"Graph Processing... Iteration {event.get('iteration', 0)}")
            
    # --- Dashboard Layout ---
    col1, col2, col3 = st.columns([1.5, 2, 1.5])
    
    with col1:
        st.subheader("Original Article")
        st.markdown(f"<div style='padding:10px; border:1px solid #ccc;'>{actual_article[:1500]}...</div>", unsafe_allow_html=True)
        
    with col2:
        st.subheader("Live Debate & Hallucination Check")
        if final_end_state and final_end_state.get("debate_log"):
            persona_a = final_end_state.get('persona_a', 'Challenger')
            persona_b = final_end_state.get('persona_b', 'Supporter')
            
            for round_data in final_end_state["debate_log"]:
                with st.expander(f"Round {round_data['iteration']}", expanded=True):
                    st.markdown(f"**Challenger Persona:** {persona_a} (Score: `{round_data.get('a_score', 0)}`)")
                    if round_data.get("highlighted_text_a"):
                        st.markdown(round_data["highlighted_text_a"], unsafe_allow_html=True)
                    
                    st.write("---")
                    
                    st.markdown(f"**Supporter Persona:** {persona_b} (Score: `{round_data.get('b_score', 0)}`)")
                    if round_data.get("highlighted_text_b"):
                        st.markdown(round_data["highlighted_text_b"], unsafe_allow_html=True)
                        
            st.markdown("<br>**Legend:** <span style='color: #155724; background-color: #d4edda; padding: 2px 4px; border-radius: 3px; font-size: 0.85em;'>Verified Entity</span> <span style='color: #004085; background-color: #cce5ff; padding: 2px 4px; border-radius: 3px; font-size: 0.85em;'>Verified Web</span> <span style='color: #721c24; background-color: #f8d7da; padding: 2px 4px; border-radius: 3px; font-size: 0.85em;'>Unverified/Hallucinated</span>", unsafe_allow_html=True)

    with col3:
        st.subheader("Final Synthesis")
        if final_end_state and final_end_state.get("final_summary"):
            st.success("Mediator's Report:")
            st.write(final_end_state["final_summary"])
            
            st.markdown("#### Evaluation Metrics")
            if final_end_state.get("synthesis_rouge"):
                st.markdown("**ROUGE Scores (Info Conservation):**")
                st.json(final_end_state["synthesis_rouge"])
                
            if final_end_state.get("synthesis_neutral"):
                st.markdown("**Sentiment Analysis (Bias Reduction):**")
                st.caption("Values closer to 0 are more Neutral.")
                st.json(final_end_state["synthesis_neutral"])
                
            if final_end_state.get("debate_influence"):
                inf = final_end_state["debate_influence"]
                st.markdown("**Debate Influence (Winner):**")
                st.caption("Mathematically proves whose points the Mediator retained.")
                st.progress(inf['challenger'] / 100.0, text=f"Challenger Influence: {inf['challenger']}%")
                st.progress(inf['supporter'] / 100.0, text=f"Supporter Influence: {inf['supporter']}%")
            
if __name__ == "__main__":
    main()
