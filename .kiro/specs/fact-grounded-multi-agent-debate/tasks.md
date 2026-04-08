# Implementation Plan: Fact-Grounded Multi-Agent Debate

## Overview

Rewrite/fix `app.py` as a single-file Python Streamlit application. Tasks follow the design document top-to-bottom: state definition → scoring formula → agent round logic → mediator → UI fixes → URL input → env config → transparency report display → property-based tests → dependencies.

## Tasks

- [x] 1. Fix `GraphState` TypedDict — add missing fields
  - Add `agent_a_round1: str`, `agent_b_round1: str`, and `transparency_report: dict` to the `GraphState` TypedDict definition in `app.py`
  - Update `initial_state` dict in `simple_main` to initialize `agent_a_round1=""`, `agent_b_round1=""`, `transparency_report={}`
  - _Requirements: 2.1, 2.2_

- [x] 2. Fix `fact_checker_node` — complete Authority Score formula
  - [x] 2.1 Implement `C_score` computation inside `calculate_score`: split summary by `"."`, filter non-empty segments, compute ratio of segments containing at least one entity from `E_total_set`
    - _Requirements: 5.7_
  - [x] 2.2 Apply full formula: `A_s = (E_cited / E_total) + 0.1 * C_score`, then clamp to `[0.0, 1.0]` using `min(1.0, max(0.0, raw))`
    - _Requirements: 5.8, 5.9_
  - [x]* 2.3 Write property test — Property 1: Authority Score always in [0, 1]
  - [x]* 2.4 Write property test — Property 2: Authority Score idempotence
  - [x]* 2.5 Write property test — Property 3: E_cited never exceeds E_total
  - [x]* 2.6 Write property test — Property 4: C_score always in [0, 1]
  - [x]* 2.7 Write property test — Property 13: Case-insensitive entity matching

- [x] 3. Fix `agent_a_node` and `agent_b_node` — Round 1 vs Round 2 behavior
  - [x] 3.1 Update `agent_a_node`: Round 1 → `agent_a_round1`, Round 2 → `agent_a_summary`
    - _Requirements: 3.2, 3.3, 3.4, 3.5, 3.6_
  - [x] 3.2 Update `agent_b_node`: Round 1 → `agent_b_round1`, Round 2 → `agent_b_summary`
    - _Requirements: 4.2, 4.3, 4.4, 4.5, 4.6_
  - [x]* 3.3 Write property test — Property 7: Agent A Round 1 stores to `agent_a_round1`
  - [x]* 3.4 Write property test — Property 8: Agent A Round 2 stores to `agent_a_summary`

- [x] 4. Fix `mediator_node` — transparency report and round1 fallback
  - [x] 4.1 Add round1 fallback
    - _Requirements: 7.1, 7.2_
  - [x] 4.2 Compute `transparency_report` dict
    - _Requirements: 7.7, 9.4, 9.5, 9.7_
  - [x] 4.3 Return `{"final_summary": ..., "transparency_report": ...}`
    - _Requirements: 7.6, 7.8_
  - [x]* 4.4 Write property test — Property 9: Mediator fallback uses round1 when summary is empty
  - [x]* 4.5 Write property test — Property 10: Transparency report `dominant_agent` correctness
  - [x]* 4.6 Write property test — Property 11: Transparency report `score_delta` correctness

- [x] 5. Fix routing — `mediation_router` correctness
  - _Requirements: 6.2, 6.3, 6.4_
  - [x]* 5.1 Write property test — Property 5: Routing correctness — rewrite condition
  - [x]* 5.2 Write property test — Property 6: Routing correctness — mediator condition
  - [x]* 5.3 Write property test — Property 14: Iteration increments by exactly 1 per Fact-Checker pass

- [x] 6. Fix Streamlit UI — accumulate debate log via `session_state`
  - _Requirements: 8.3, 8.4, 8.5, 8.6, 8.7_

- [x] 7. Add URL input and `fetch_article()` function
  - [x] 7.1 Implement `fetch_article(url)`
    - _Requirements: 1.3, 1.4_
  - [x] 7.2 Add URL text input field in left column
    - _Requirements: 1.2, 1.4, 1.5_
  - [x]* 7.3 Write property test — Property 12: URL fetch failure does not invoke pipeline

- [x] 8. Add `GROQ_API_KEY` validation and `GROQ_MODEL` env var support
  - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [x] 9. Add Transparency Report display in right column
  - _Requirements: 8.9, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7_

- [x] 10. Checkpoint — verify graph wiring and full pipeline
  - _Requirements: 2.4, 6.1_

- [x] 11. Update `requirements.txt` with all dependencies
  - _Requirements: 10.5, 11.1_

- [x] 12. Final checkpoint — all tests pass

## Notes

- Tasks marked with `*` are optional property-based tests (all implemented)
- All property tests use Hypothesis with `@settings(max_examples=100)`
- LLM calls in agent and mediator tests are mocked via `unittest.mock.patch`
- spaCy is used directly (no mock) in Fact-Checker tests — it is deterministic
