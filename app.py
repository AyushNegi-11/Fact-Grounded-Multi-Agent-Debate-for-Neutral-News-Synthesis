# app.py — Fact-Grounded Multi-Agent Debate
import os
import spacy
import streamlit as st
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
import operator

load_dotenv()

st.set_page_config(page_title="⚖️ AI Debate Arena", layout="wide")

st.markdown("""
<style>
.bubble-a {
    background: #1e3a5f;
    border-radius: 18px 18px 18px 4px;
    padding: 10px 16px;
    margin: 4px 0;
    max-width: 75%;
    color: #e8f4fd;
    font-size: 0.9rem;
    line-height: 1.5;
    display: inline-block;
}
.bubble-b {
    background: #3a1e1e;
    border-radius: 18px 18px 4px 18px;
    padding: 10px 16px;
    margin: 4px 0;
    max-width: 75%;
    color: #fde8e8;
    font-size: 0.9rem;
    line-height: 1.5;
    display: inline-block;
}
.row-a { text-align: left; margin-bottom: 8px; }
.row-b { text-align: right; margin-bottom: 8px; }
.label-a { color: #5ba3d9; font-size: 0.72rem; font-weight: 700; margin-bottom: 2px; }
.label-b { color: #d95b5b; font-size: 0.72rem; font-weight: 700; margin-bottom: 2px; }
.score-tag { font-size: 0.68rem; color: #888; margin-top: 2px; }
.judge-box {
    background: #111827;
    border: 1px solid #374151;
    border-radius: 12px;
    padding: 18px 20px;
    color: #f3f4f6;
    font-size: 0.9rem;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)


# ── LLM ───────────────────────────────────────────────────────────────────────

def get_llm():
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    return ChatGroq(model=model, temperature=0.75)


# ── spaCy ─────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_md")
    except OSError:
        from spacy.cli import download
        download("en_core_web_md")
        return spacy.load("en_core_web_md")


# ── Authority Score ───────────────────────────────────────────────────────────

def compute_authority_score(text: str, nlp) -> float:
    """
    Score a single debate message on its own factual density:
      - Count named entities (people, orgs, dates, places, numbers)
      - Count specific numbers/years mentioned
      - Normalize by message length
    Higher = more specific factual content cited.
    """
    if not text.strip():
        return 0.0

    doc = nlp(text)

    # Count named entities of factual types
    factual_types = {"PERSON", "ORG", "GPE", "DATE", "CARDINAL", "PERCENT",
                     "MONEY", "LAW", "EVENT", "NORP", "FAC", "LOC"}
    entity_count = sum(1 for e in doc.ents if e.label_ in factual_types)

    # Count explicit numbers/years in text (e.g. "2002", "35%", "₹500 crore")
    import re
    number_hits = len(re.findall(r'\b\d{4}\b|\d+%|\d+\s*crore|\d+\s*lakh|\d+\s*billion', text))

    # Word count for normalization
    word_count = max(len(text.split()), 1)

    # Score = (entities + numbers) per 10 words, capped at 1.0
    raw = (entity_count + number_hits) / (word_count / 10)
    return round(min(1.0, raw), 3)


# ── LangGraph State ───────────────────────────────────────────────────────────

class GraphState(TypedDict):
    article: str
    messages: Annotated[list[dict], operator.add]
    round: int
    max_rounds: int
    a_scores: Annotated[list[float], operator.add]
    b_scores: Annotated[list[float], operator.add]
    verdict: str


# ── Nodes ─────────────────────────────────────────────────────────────────────

def challenger_node(state: GraphState) -> dict:
    article = state["article"]
    round_num = state["round"]
    messages = state.get("messages", [])
    nlp = load_spacy_model()
    llm = get_llm()

    last_supporter = next(
        (m["text"] for m in reversed(messages) if m["role"] == "supporter"), ""
    )

    if round_num == 1:
        prompt = (
            f"You are the CHALLENGER in a debate. Read this article:\n\"{article[:1000]}\"\n\n"
            f"Give a SHORT opening argument (strictly 1-2 sentences) that challenges the "
            f"article's main claim. Mention ONE specific fact, name, or date from the article. "
            f"Be punchy. No bullet points. No intro phrases like 'I believe' or 'As a challenger'."
        )
    else:
        prompt = (
            f"Article context: \"{article[:400]}\"\n\n"
            f"Supporter said: \"{last_supporter}\"\n\n"
            f"You are the CHALLENGER. Reply in 1-2 sentences only. "
            f"Counter with ONE specific fact. Be direct and punchy. No filler phrases."
        )

    resp = llm.invoke([HumanMessage(content=prompt)])
    text = resp.content.strip()
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    text = ". ".join(sentences[:2]) + ("." if sentences else "")
    score = compute_authority_score(text, nlp)

    return {
        "messages": [{"role": "challenger", "text": text, "round": round_num, "score": score}],
        "a_scores": [score],
    }


def supporter_node(state: GraphState) -> dict:
    article = state["article"]
    round_num = state["round"]
    messages = state.get("messages", [])
    nlp = load_spacy_model()
    llm = get_llm()

    last_challenger = next(
        (m["text"] for m in reversed(messages) if m["role"] == "challenger"), ""
    )

    if round_num == 1:
        prompt = (
            f"You are the SUPPORTER in a debate. Read this article:\n\"{article[:1000]}\"\n\n"
            f"Challenger said: \"{last_challenger}\"\n\n"
            f"Defend the article in 1-2 sentences only. Cite ONE specific fact or name. "
            f"Be punchy. No bullet points. No intro phrases."
        )
    else:
        prompt = (
            f"Article context: \"{article[:400]}\"\n\n"
            f"Challenger said: \"{last_challenger}\"\n\n"
            f"You are the SUPPORTER. Reply in 1-2 sentences only. "
            f"Counter with ONE specific fact from the article. Be direct. No filler phrases."
        )

    resp = llm.invoke([HumanMessage(content=prompt)])
    text = resp.content.strip()
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    text = ". ".join(sentences[:2]) + ("." if sentences else "")
    score = compute_authority_score(text, nlp)

    return {
        "messages": [{"role": "supporter", "text": text, "round": round_num, "score": score}],
        "b_scores": [score],
        "round": round_num + 1,
    }


def judge_node(state: GraphState) -> dict:
    article = state["article"]
    messages = state.get("messages", [])
    a_scores = state.get("a_scores", [])
    b_scores = state.get("b_scores", [])
    llm = get_llm()

    transcript = ""
    for m in messages:
        speaker = "Challenger" if m["role"] == "challenger" else "Supporter"
        transcript += f"{speaker} (Round {m['round']}): {m['text']}\n\n"

    avg_a = round(sum(a_scores) / len(a_scores), 3) if a_scores else 0.0
    avg_b = round(sum(b_scores) / len(b_scores), 3) if b_scores else 0.0

    prompt = (
        f"You are a neutral Judge evaluating a debate about: \"{article[:600]}\"\n\n"
        f"Debate transcript:\n{transcript}\n"
        f"Factual keyword scores — Challenger: {avg_a}, Supporter: {avg_b}\n\n"
        f"Based on the arguments made AND the scores above, decide who won. "
        f"Write a 3-4 sentence verdict. End your verdict with exactly one of these lines:\n"
        f"WINNER: Challenger\n"
        f"WINNER: Supporter\n"
        f"WINNER: Tie\n"
        f"Be concise and direct. No bullet points."
    )
    resp = llm.invoke([HumanMessage(content=prompt)])
    verdict_text = resp.content.strip()

    # Parse winner from LLM response
    winner = "Tie"
    for line in verdict_text.splitlines():
        line = line.strip()
        if line.startswith("WINNER:"):
            declared = line.replace("WINNER:", "").strip()
            if "Challenger" in declared:
                winner = "Challenger"
            elif "Supporter" in declared:
                winner = "Supporter"
            else:
                winner = "Tie"
            break
    # Fallback to score comparison if LLM didn't declare
    if winner == "Tie" and abs(avg_a - avg_b) > 0.01:
        winner = "Challenger" if avg_a > avg_b else "Supporter"

    # Remove the WINNER: line from display text
    clean_text = "\n".join(
        l for l in verdict_text.splitlines() if not l.strip().startswith("WINNER:")
    ).strip()

    verdict = (
        f"**Challenger Score:** `{avg_a}`  |  **Supporter Score:** `{avg_b}`\n\n"
        f"### 🏆 Winner: {winner}\n\n"
        f"{clean_text}"
    )
    return {"verdict": verdict}


def should_continue(state: GraphState) -> str:
    if state["round"] > state["max_rounds"]:
        return "judge"
    return "challenger"


# ── Build Graph ───────────────────────────────────────────────────────────────

@st.cache_resource
def build_graph():
    wf = StateGraph(GraphState)
    wf.add_node("challenger", challenger_node)
    wf.add_node("supporter", supporter_node)
    wf.add_node("judge", judge_node)
    wf.set_entry_point("challenger")
    wf.add_edge("challenger", "supporter")
    wf.add_conditional_edges("supporter", should_continue, {
        "challenger": "challenger",
        "judge": "judge",
    })
    wf.add_edge("judge", END)
    return wf.compile()


# ── Render chat ───────────────────────────────────────────────────────────────

def render_messages(messages: list[dict]):
    for m in messages:
        rnd = m.get("round", "")
        text = m["text"]
        score = m.get("score", None)
        score_str = f'<div class="score-tag">Factual score: {score}</div>' if score is not None else ""
        if m["role"] == "challenger":
            st.markdown(
                f'<div class="row-a">'
                f'<div class="label-a">🔵 Challenger — Round {rnd}</div>'
                f'<div class="bubble-a">{text}</div>'
                f'{score_str}'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="row-b">'
                f'<div class="label-b">🔴 Supporter — Round {rnd}</div>'
                f'<div class="bubble-b">{text}</div>'
                f'{score_str}'
                f'</div>',
                unsafe_allow_html=True
            )


# ── Main ──────────────────────────────────────────────────────────────────────

def simple_main():
    if not os.getenv("GROQ_API_KEY"):
        st.error("⚠️ GROQ_API_KEY not set. Add it to your .env file and restart.")
        st.stop()

    st.title("⚖️ AI Debate Arena")
    st.caption("Two AI agents debate a news article in short punchy messages. A Judge scores them on factual accuracy.")

    st.divider()

    # ── Input ─────────────────────────────────────────────────────────────────
    col_inp, col_rounds, col_btn = st.columns([5, 1, 1])
    with col_inp:
        user_article = st.text_area(
            "📋 Paste your news article or topic",
            height=120,
            placeholder="e.g. BJP vs Congress — which party has a worse corruption record?",
        )
    with col_rounds:
        num_rounds = st.number_input("Rounds", min_value=2, max_value=6, value=3, step=1)
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        start_btn = st.button("▶ Start", use_container_width=True, type="primary")

    st.divider()

    # ── Session state ─────────────────────────────────────────────────────────
    for key, default in [("messages", []), ("verdict", ""), ("a_scores", []), ("b_scores", [])]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Layout ────────────────────────────────────────────────────────────────
    col_chat, col_verdict = st.columns([3, 2])

    with col_chat:
        st.subheader("💬 Live Debate")
        chat_slot = st.empty()
        with chat_slot.container():
            if st.session_state.messages:
                render_messages(st.session_state.messages)
            else:
                st.info("Debate will appear here once you start.")

    with col_verdict:
        st.subheader("🏛️ Judge's Verdict")
        verdict_slot = st.empty()
        with verdict_slot.container():
            if st.session_state.verdict:
                st.markdown(st.session_state.verdict)
            else:
                st.info("Verdict will appear here after the debate ends.")

    # ── Run ───────────────────────────────────────────────────────────────────
    if start_btn:
        if not user_article.strip():
            st.warning("Please paste an article or topic first.")
            st.stop()

        # Reset
        st.session_state.messages = []
        st.session_state.verdict = ""
        st.session_state.a_scores = []
        st.session_state.b_scores = []

        graph = build_graph()

        initial_state: GraphState = {
            "article": user_article.strip(),
            "messages": [],
            "round": 1,
            "max_rounds": int(num_rounds),
            "a_scores": [],
            "b_scores": [],
            "verdict": "",
        }

        status = st.empty()

        try:
            for event in graph.stream(initial_state, stream_mode="updates"):

                if "challenger" in event:
                    data = event["challenger"]
                    st.session_state.messages.extend(data.get("messages", []))
                    st.session_state.a_scores.extend(data.get("a_scores", []))
                    status.info("🔵 Challenger is typing...")

                elif "supporter" in event:
                    data = event["supporter"]
                    st.session_state.messages.extend(data.get("messages", []))
                    st.session_state.b_scores.extend(data.get("b_scores", []))
                    status.info("🔴 Supporter is typing...")

                elif "judge" in event:
                    st.session_state.verdict = event["judge"].get("verdict", "")
                    status.success("🏛️ Judge has delivered the verdict!")

                # Re-render chat after every node
                with chat_slot.container():
                    render_messages(st.session_state.messages)

            status.empty()

            with verdict_slot.container():
                st.markdown(st.session_state.verdict)

            st.balloons()

        except Exception as e:
            st.error(f"Error: {e}")
            raise e


if __name__ == "__main__":
    simple_main()
