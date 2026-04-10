# Dialectic AI: Agentic News Neutralizer

Welcome to **Dialectic AI: Agentic News Neutralizer**! This is a state-of-the-art (SOTA) NLP project designed to automatically neutralize media bias, hallucination, and logical fallacies in news articles through an autonomous, multi-agent AI debate.

Built for robustness and explainability, this system leverages LangGraph, large language models (via Groq), retrieval-augmented generation (RAG), and advanced evaluation metrics to mathematically prove the neutrality and accuracy of its final synthesis.

---

## Key Features

### Multi-Agent Orchestration (LangGraph)
The core architecture loops three distinct LLM "personas" against each other:
1. **The Challenger**: Scrutinizes the article, attacking its biases and logical leaps.
2. **The Supporter**: Defends the article's core claims and provides a counter-perspective.
3. **The Mediator**: Reads the final arguments of both agents and synthesizes a perfectly neutral, balanced summary of the truth.

### Zero-Hallucination Fact Checking & RAG
Before an agent's argument is accepted, it passes through a strict Python-based Fact Checker:
*   **Named Entity Recognition (NER)**: Uses `spaCy` to extract entities from the AI's arguments.
*   **Web-Augmented Verification**: Cross-references entities against live DuckDuckGo web searches (`ddgs`). 
*   **Color-Coded UI Confidence**:
    *   <span style="color: #155724; background-color: #d4edda; padding: 2px 4px;">Verified Entity</span> (Entity existed in the original text)
    *   <span style="color: #004085; background-color: #cce5ff; padding: 2px 4px;">Verified Web</span> (Entity was proven by a live web search)
    *   <span style="color: #721c24; background-color: #f8d7da; padding: 2px 4px;">Unverified/Hallucinated</span> (AI made this up; heavily penalized)

### Reflexion & Chain-of-Thought (CoT)
The graph features an autonomous **Fallacy Checker**. If an agent's logic is weak or scores below a `0.35` threshold, the Fallacy Checker generates a detailed critique. The graph intrinsically re-routes the agents into a "Reflexion Loop," forcing them to read their critiques and rewrite their arguments until they meet journalistic standards.

### Mathematical Evaluation Metrics (SOTA)
To prevent the application from being a "black box," the Mediator calculates rigorous NLP metrics upon completion:
*   **ROUGE Scores (Info Conservation)**: Calculates the exact structural predictability and unigram overlap (ROUGE-1, ROUGE-L) between the original article and the final summary.
*   **Sentiment Analysis (Bias Reduction)**: Uses `TextBlob` to calculate the mathematical polarity shift between the heavily biased input and the synthesized output, ensuring the value approaches `0.0` (Neutral).
*   **Debate Influence Tracker**: Mathematically analyzes semantic crossover to prove exactly what percentage of the Challenger vs. Supporter arguments were retained by the Mediator.

---

## Architecture

*   **`app.py`**: The dynamic Streamlit frontend and dashboard.
*   **`graph.py`**: The LangGraph state machine orchestrating the debate and Reflexion loops.
*   **`state.py`**: The central `GraphState` TypedDict holding metrics, critiques, and logs.
*   **`config.py`**: Model loading (`sentence-transformers`, `spacy`) and silent warning interception.
*   **`agents/`**: Modular logic instances for the `challenger`, `supporter`, `fallacy_checker`, `fact_checker`, and `mediator`.
*   **`utils/tools.py`**: Web scraping and RAG utilities.

---

## Installation & Setup

1. **Clone the repository** and ensure you have Python 3.10+ installed.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_md
   ```

3. **Configure Environment Variables**:
   Create a `.env` file in the root directory. You must supply a free API key from [Groq Console](https://console.groq.com/).
   ```env
   GROQ_API_KEY=your_key_here
   GROQ_MODEL_A=qwen/qwen3-32b
   GROQ_MODEL_B=llama-3.1-8b-instant
   GROQ_MODEL_MEDIATOR=llama-3.3-70b-versatile
   ```
   *(Note: The system utilizes heterogeneous models so the agents genuinely disagree and reason differently).*

4. **Run the Application**:
   ```bash
   python -m streamlit run app.py
   ```

---

## Usage

1. Paste a raw article (or provide an HTTP link to an article) directly into the Streamlit UI.
2. Click **Start Debate**.
3. Watch the graph autonomously route the agents in real-time. If they hallucinate or use poor logic, watch the graph dynamically trigger a "Round 2" Reflexion loop!
4. Review the final generated synthesis and the SOTA mathematics validating the neutrality of the output.
