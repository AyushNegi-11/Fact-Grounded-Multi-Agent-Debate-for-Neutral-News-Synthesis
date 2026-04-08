# Requirements Document

## Introduction

This document specifies the requirements for the Fact-Grounded Multi-Agent Debate system — a LangGraph-orchestrated pipeline where two adversarial AI agents (Challenger and Supporter) debate a news article, a spaCy-based Fact-Checker scores their factual fidelity using an Authority Score formula, and a Mediator agent synthesizes a final neutral summary. The system is deployed as a Streamlit web application using Groq API (Llama-4-Scout / llama-3.1-8b-instant) for LLM inference. No external APIs beyond Groq are required.

The system addresses known issues in the existing `app.py`: broken streaming state updates, incomplete UI rendering, missing Round 2 refinement, and an incomplete Authority Score formula that omits the coherence weight term.

## Glossary

- **System**: The complete Fact-Grounded Multi-Agent Debate application
- **Pipeline**: The LangGraph StateGraph workflow that orchestrates all agents
- **Agent_A**: The Challenger agent that identifies biases, gaps, and opposing viewpoints
- **Agent_B**: The Supporter agent that highlights core claims and factual strengths
- **Fact_Checker**: The spaCy-based scoring node that computes Authority Scores without calling any LLM
- **Mediator**: The final synthesis agent that produces the neutral, authority-weighted summary
- **GraphState**: The shared TypedDict state dictionary passed between all pipeline nodes
- **Authority_Score**: The factual fidelity metric defined as `A_s = (E_cited / E_total) + ω * C_score`, where `E_cited` is the count of original article named entities present in the summary, `E_total` is the total count of named entities in the original article, `C_score` is the coherence score (ratio of summary sentences containing at least one named entity), and `ω` is a fixed coherence weight of 0.1
- **E_cited**: Count of named entities from the original article that appear in an agent's summary
- **E_total**: Total count of unique named entities extracted from the original article
- **C_score**: Coherence score — ratio of sentences in a summary that contain at least one named entity
- **ω (omega)**: Fixed coherence weight constant, value 0.1
- **NER**: Named Entity Recognition performed by spaCy `en_core_web_md`
- **Round_1**: The first generation pass where Agent_A and Agent_B independently summarize the article
- **Round_2**: The refinement pass where agents see each other's Round 1 output and their Authority Scores before rewriting
- **Rewrite_Loop**: A conditional cycle back to Agent_A triggered when either agent's Authority Score falls below the threshold
- **Transparency_Report**: A structured display showing per-agent Authority Scores, iteration count, and dominant contributor
- **URL_Input**: An optional input mode where the user provides a news article URL instead of raw text
- **Streamlit_App**: The web UI built with Streamlit that renders the 3-column debate interface

---

## Requirements

### Requirement 1: Article Input

**User Story:** As a student or researcher, I want to provide a news article either as raw text or a URL, so that I can run the debate pipeline on any news source without manual copy-pasting.

#### Acceptance Criteria

1. THE Streamlit_App SHALL provide a text area input for pasting raw article text.
2. THE Streamlit_App SHALL provide a URL input field that accepts an HTTP or HTTPS URL.
3. WHEN a URL is submitted, THE System SHALL fetch the article body text from the URL using an HTTP GET request and populate the raw text field with the extracted content.
4. IF a URL fetch fails or returns a non-200 HTTP status, THEN THE Streamlit_App SHALL display an error message stating the URL could not be fetched and SHALL NOT proceed to the pipeline.
5. IF both the URL field and the text area are empty when the user clicks the start button, THEN THE Streamlit_App SHALL display a validation error and SHALL NOT invoke the Pipeline.
6. WHEN raw text is provided directly, THE System SHALL use that text as the `original_article` value in GraphState without any modification.

---

### Requirement 2: LangGraph Pipeline State Management

**User Story:** As a developer, I want a well-defined shared state that all pipeline nodes read from and write to, so that data flows correctly between agents without loss or corruption.

#### Acceptance Criteria

1. THE Pipeline SHALL define GraphState as a TypedDict containing: `original_article` (str), `agent_a_round1` (str), `agent_b_round1` (str), `agent_a_summary` (str), `agent_b_summary` (str), `a_score` (float), `b_score` (float), `final_summary` (str), `iteration` (int), and `transparency_report` (dict).
2. THE Pipeline SHALL initialize all string fields to empty string, all float fields to 0.0, `iteration` to 0, and `transparency_report` to an empty dict before invoking the graph.
3. WHEN a node returns a partial dictionary, THE Pipeline SHALL merge that dictionary into the existing GraphState without overwriting unrelated fields.
4. THE Pipeline SHALL use `graph.stream(initial_state, stream_mode="updates")` to yield per-node state deltas so the UI can render incremental updates.
5. WHEN `graph.stream` yields an update event, THE System SHALL include the node name as the top-level key in the event dictionary so the UI can identify which node completed.

---

### Requirement 3: Round 1 — Agent A (Challenger)

**User Story:** As a reader, I want an AI agent to critically challenge the article's claims, so that I can see potential biases and gaps identified systematically.

#### Acceptance Criteria

1. WHEN the Pipeline starts, THE Agent_A SHALL be the entry point node.
2. THE Agent_A SHALL read `original_article` from GraphState and send it to the Groq LLM with a system prompt that instructs the model to act as a critical challenger identifying logical gaps, factual omissions, and ideological biases.
3. THE Agent_A SHALL instruct the LLM to produce a response of exactly 150 words.
4. THE Agent_A SHALL store the LLM response string in `agent_a_round1` in GraphState.
5. WHEN `iteration` in GraphState is greater than 0 (Round 2), THE Agent_A SHALL include `agent_b_round1`, `a_score`, and `b_score` in the prompt context and instruct the model to revise its summary to improve factual entity coverage.
6. WHEN `iteration` is greater than 0, THE Agent_A SHALL store the revised output in `agent_a_summary` instead of `agent_a_round1`.
7. IF the Groq API returns an error, THEN THE Agent_A SHALL raise the exception so the Streamlit error handler can catch and display it.

---

### Requirement 4: Round 1 — Agent B (Supporter)

**User Story:** As a reader, I want an AI agent to advocate for the article's core claims, so that I can see the strongest factual case for the reported narrative.

#### Acceptance Criteria

1. WHEN Agent_A completes, THE Pipeline SHALL route execution to Agent_B.
2. THE Agent_B SHALL read `original_article` from GraphState and send it to the Groq LLM with a system prompt that instructs the model to act as a supportive analyst highlighting verified facts, primary arguments, and credible evidence.
3. THE Agent_B SHALL instruct the LLM to produce a response of exactly 150 words.
4. THE Agent_B SHALL store the LLM response string in `agent_b_round1` in GraphState.
5. WHEN `iteration` in GraphState is greater than 0 (Round 2), THE Agent_B SHALL include `agent_a_round1`, `a_score`, and `b_score` in the prompt context and instruct the model to revise its summary to improve factual entity coverage.
6. WHEN `iteration` is greater than 0, THE Agent_B SHALL store the revised output in `agent_b_summary` instead of `agent_b_round1`.
7. IF the Groq API returns an error, THEN THE Agent_B SHALL raise the exception so the Streamlit error handler can catch and display it.

---

### Requirement 5: Fact-Checker Node and Authority Score

**User Story:** As an academic evaluator, I want a quantitative factual fidelity score for each agent's summary, so that the final synthesis can be weighted by factual accuracy rather than treating both agents equally.

#### Acceptance Criteria

1. WHEN Agent_B completes, THE Pipeline SHALL route execution to Fact_Checker.
2. THE Fact_Checker SHALL NOT invoke any LLM or external API; it SHALL use only spaCy `en_core_web_md` for all NER operations.
3. THE Fact_Checker SHALL extract all named entities from `original_article` using spaCy NER and store them as a lowercase deduplicated set (`E_total_set`).
4. THE Fact_Checker SHALL compute `E_total` as the cardinality of `E_total_set`.
5. IF `E_total` equals 0, THEN THE Fact_Checker SHALL assign `a_score = 1.0` and `b_score = 1.0` and SHALL increment `iteration` by 1.
6. THE Fact_Checker SHALL compute `E_cited` for each agent summary as the count of entities in `E_total_set` that appear (case-insensitive substring match) in the summary text.
7. THE Fact_Checker SHALL compute `C_score` for each agent summary as the ratio of sentences (split by period) that contain at least one named entity from `E_total_set`.
8. THE Fact_Checker SHALL compute `Authority_Score` for each agent as `A_s = (E_cited / E_total) + 0.1 * C_score`.
9. THE Fact_Checker SHALL clamp each `Authority_Score` to the range [0.0, 1.0].
10. THE Fact_Checker SHALL store the computed scores in `a_score` and `b_score` in GraphState.
11. THE Fact_Checker SHALL increment `iteration` by 1 in GraphState.
12. FOR ALL valid article and summary text pairs, computing the Authority Score twice on the same inputs SHALL produce the same result (idempotence).

---

### Requirement 6: Rewrite Loop Routing

**User Story:** As a system designer, I want the pipeline to automatically trigger a second refinement round when factual scores are too low, so that the final synthesis is based on adequately fact-grounded summaries.

#### Acceptance Criteria

1. WHEN Fact_Checker completes, THE Pipeline SHALL evaluate a conditional routing function.
2. THE Pipeline SHALL route to Agent_A (rewrite) WHEN `iteration` is less than or equal to 2 AND (`a_score` is less than 0.15 OR `b_score` is less than 0.15).
3. THE Pipeline SHALL route to Mediator WHEN `iteration` is greater than 2 OR (both `a_score` and `b_score` are greater than or equal to 0.15).
4. THE Pipeline SHALL allow a maximum of 2 rewrite iterations before forcing progression to Mediator regardless of scores.
5. WHEN a rewrite is triggered, THE Pipeline SHALL preserve all existing GraphState fields and only update the fields modified by Agent_A, Agent_B, and Fact_Checker in the subsequent pass.

---

### Requirement 7: Mediator Agent — Final Neutral Synthesis

**User Story:** As a reader, I want a final 200-word neutral summary that weighs each agent's contribution by their Authority Score, so that I receive a balanced, fact-grounded news synthesis.

#### Acceptance Criteria

1. WHEN the routing function directs to Mediator, THE Mediator SHALL read `original_article`, `agent_a_summary`, `agent_b_summary`, `a_score`, and `b_score` from GraphState.
2. WHEN `agent_a_summary` is empty (no Round 2 occurred), THE Mediator SHALL fall back to `agent_a_round1` and `agent_b_round1` respectively.
3. THE Mediator SHALL send a prompt to the Groq LLM instructing it to synthesize a 200-word neutral summary, weighting the higher-scoring agent's perspective proportionally more.
4. THE Mediator prompt SHALL explicitly include both agent summaries with their numeric Authority Scores so the LLM can apply authority-weighted synthesis.
5. THE Mediator SHALL instruct the LLM to remove all opinionated language, hedging phrases, and first-person references from the final output.
6. THE Mediator SHALL store the LLM response in `final_summary` in GraphState.
7. THE Mediator SHALL compute a `transparency_report` dict containing: `a_score`, `b_score`, `iteration`, `dominant_agent` (the agent with the higher score), and `score_delta` (absolute difference between scores).
8. THE Mediator SHALL store the `transparency_report` dict in GraphState.
9. IF the Groq API returns an error, THEN THE Mediator SHALL raise the exception so the Streamlit error handler can catch and display it.

---

### Requirement 8: Streamlit UI — Three-Column Layout

**User Story:** As a user, I want a clear three-column interface showing the original article, the live debate process, and the final result simultaneously, so that I can follow the entire pipeline in one view.

#### Acceptance Criteria

1. THE Streamlit_App SHALL render a three-column layout with equal width columns labeled "Input Article", "Live Debate", and "Final Report".
2. THE Streamlit_App SHALL display the original article text in the left column after the pipeline starts.
3. THE Streamlit_App SHALL display live debate progress in the center column by updating a Streamlit placeholder container after each node completes.
4. WHEN Agent_A completes, THE Streamlit_App SHALL append a labeled expander in the center column showing Agent_A's Round 1 summary text.
5. WHEN Agent_B completes, THE Streamlit_App SHALL append a labeled expander in the center column showing Agent_B's Round 1 summary text.
6. WHEN Fact_Checker completes, THE Streamlit_App SHALL display both Authority Scores and the current iteration number in the center column.
7. WHEN a rewrite loop is triggered, THE Streamlit_App SHALL display a visible warning in the center column indicating which agent(s) scored below threshold and that a refinement round is starting.
8. WHEN Mediator completes, THE Streamlit_App SHALL display the final neutral summary in the right column.
9. THE Streamlit_App SHALL display the Transparency Report in the right column below the final summary, showing `a_score`, `b_score`, `dominant_agent`, `score_delta`, and total `iteration` count.
10. IF an exception is raised during pipeline execution, THEN THE Streamlit_App SHALL display the exception message in the center column and SHALL NOT leave the UI in a loading state.

---

### Requirement 9: Transparency Report Display

**User Story:** As an academic reviewer, I want a transparency report showing which agent contributed more and their respective scores, so that I can evaluate the objectivity and reproducibility of the synthesis.

#### Acceptance Criteria

1. THE Streamlit_App SHALL render the Transparency Report as a structured section in the right column after the final summary.
2. THE Transparency Report SHALL display Agent A's Authority Score as a decimal rounded to 3 places.
3. THE Transparency Report SHALL display Agent B's Authority Score as a decimal rounded to 3 places.
4. THE Transparency Report SHALL display the dominant agent label ("Challenger" or "Supporter") based on which agent has the higher Authority Score.
5. WHEN both agents have equal Authority Scores, THE Transparency Report SHALL display "Equal Contribution" as the dominant agent label.
6. THE Transparency Report SHALL display the total number of rewrite iterations completed.
7. THE Transparency Report SHALL display the score delta (|a_score - b_score|) rounded to 3 places.

---

### Requirement 10: Model Configuration and Environment

**User Story:** As a developer, I want the LLM model and API key to be configurable via environment variables, so that I can switch between Groq-hosted models without modifying source code.

#### Acceptance Criteria

1. THE System SHALL load the Groq API key exclusively from the `GROQ_API_KEY` environment variable via `python-dotenv`.
2. THE System SHALL read the model name from a `GROQ_MODEL` environment variable, defaulting to `llama-3.1-8b-instant` if the variable is not set.
3. IF `GROQ_API_KEY` is not set or is an empty string, THEN THE Streamlit_App SHALL display a configuration error on startup and SHALL NOT render the input form.
4. THE System SHALL support `llama-3.1-8b-instant` and `meta-llama/llama-4-scout-17b-16e-instruct` as valid values for `GROQ_MODEL`.
5. THE System SHALL NOT require any API key or SDK other than Groq (no Gemini, OpenAI, or Anthropic dependencies).

---

### Requirement 11: spaCy Model Initialization

**User Story:** As a developer, I want the spaCy model to load once and be reused across all pipeline runs, so that the application does not incur repeated model loading overhead.

#### Acceptance Criteria

1. THE System SHALL load `en_core_web_md` exactly once per Streamlit session using `@st.cache_resource`.
2. IF `en_core_web_md` is not installed, THEN THE System SHALL automatically download it using `spacy.cli.download` before loading.
3. THE Fact_Checker SHALL receive the cached spaCy model instance as a parameter or module-level reference rather than reloading it on each invocation.

---

### Requirement 12: Round-Trip Consistency of Authority Score

**User Story:** As a developer, I want the Authority Score computation to be deterministic and consistent, so that the same article and summary always produce the same score regardless of execution order.

#### Acceptance Criteria

1. FOR ALL article texts and summary texts, THE Fact_Checker SHALL produce the same `Authority_Score` when called with the same inputs in any order (confluence property).
2. FOR ALL article texts and summary texts, calling THE Fact_Checker twice with the same inputs SHALL return identical `a_score` and `b_score` values (idempotence property).
3. THE Fact_Checker SHALL treat entity matching as case-insensitive so that "Biden" and "biden" are considered the same entity.
