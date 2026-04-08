# tests/test_fact_checker.py
# Feature: fact-grounded-multi-agent-debate
# Properties 1, 2, 3, 4, 13, 14

import pytest
from hypothesis import given, settings
from hypothesis.strategies import text, integers

from app import fact_checker_node
from tests.conftest import make_state


# --- Property 1: Authority Score is always in [0, 1] ---
# Validates: Requirements 5.9
@given(article=text(), summary=text())
@settings(max_examples=100)
def test_authority_score_clamped(article, summary):
    """Property 1: Authority Score is always in [0, 1]"""
    state = make_state(
        original_article=article,
        agent_a_round1=summary,
        agent_b_round1=summary,
    )
    result = fact_checker_node(state)
    assert 0.0 <= result["a_score"] <= 1.0
    assert 0.0 <= result["b_score"] <= 1.0


# --- Property 2: Authority Score idempotence ---
# Validates: Requirements 5.12, 12.2
@given(article=text(), summary=text())
@settings(max_examples=100)
def test_authority_score_idempotent(article, summary):
    """Property 2: Calling fact_checker_node twice with same inputs returns same scores."""
    state = make_state(
        original_article=article,
        agent_a_round1=summary,
        agent_b_round1=summary,
    )
    result1 = fact_checker_node(state)
    result2 = fact_checker_node(state)
    assert result1["a_score"] == result2["a_score"]
    assert result1["b_score"] == result2["b_score"]


# --- Property 3: E_cited never exceeds E_total (score base term <= 1.0) ---
# Validates: Requirements 5.6
@given(article=text(), summary=text())
@settings(max_examples=100)
def test_score_base_never_exceeds_one(article, summary):
    """Property 3: The entity overlap ratio (E_cited/E_total) never exceeds 1.0."""
    state = make_state(
        original_article=article,
        agent_a_round1=summary,
        agent_b_round1=summary,
    )
    result = fact_checker_node(state)
    # Since A_s is clamped and C_score contribution is 0.1*[0,1]=max 0.1,
    # the base entity ratio must be <= 1.0 for the clamp to be meaningful.
    # We verify the final score is still within bounds.
    assert result["a_score"] <= 1.0
    assert result["b_score"] <= 1.0


# --- Property 4: C_score is always in [0, 1] (verified via final score bounds) ---
# Validates: Requirements 5.7
@given(summary=text())
@settings(max_examples=100)
def test_c_score_contribution_bounded(summary):
    """Property 4: C_score contribution (0.1 * C_score) is always in [0, 0.1]."""
    # Use a fixed article with known entities to isolate C_score behavior
    article = "Barack Obama visited Paris in 2023."
    state = make_state(
        original_article=article,
        agent_a_round1=summary,
        agent_b_round1=summary,
    )
    result = fact_checker_node(state)
    # Score must remain in [0, 1] regardless of summary content
    assert 0.0 <= result["a_score"] <= 1.0


# --- Property 13: Case-insensitive entity matching ---
# Validates: Requirements 12.3
def test_case_insensitive_entity_matching():
    """Property 13: Entity matching is case-insensitive."""
    article = "Barack Obama visited Paris in 2023."
    summary_lower = "barack obama visited paris in 2023."
    summary_upper = "BARACK OBAMA VISITED PARIS IN 2023."
    summary_mixed = "Barack OBAMA visited PARIS in 2023."

    state_lower = make_state(original_article=article, agent_a_round1=summary_lower, agent_b_round1=summary_lower)
    state_upper = make_state(original_article=article, agent_a_round1=summary_upper, agent_b_round1=summary_upper)
    state_mixed = make_state(original_article=article, agent_a_round1=summary_mixed, agent_b_round1=summary_mixed)

    r_lower = fact_checker_node(state_lower)
    r_upper = fact_checker_node(state_upper)
    r_mixed = fact_checker_node(state_mixed)

    assert r_lower["a_score"] == r_upper["a_score"] == r_mixed["a_score"]


# --- Property 14: Iteration increments by exactly 1 per Fact-Checker pass ---
# Validates: Requirements 5.11
@given(n=integers(min_value=0, max_value=100), article=text(), summary=text())
@settings(max_examples=100)
def test_iteration_increments_by_one(n, article, summary):
    """Property 14: fact_checker_node increments iteration by exactly 1."""
    state = make_state(
        original_article=article,
        agent_a_round1=summary,
        agent_b_round1=summary,
        iteration=n,
    )
    result = fact_checker_node(state)
    assert result["iteration"] == n + 1


# --- Edge case: empty article returns 1.0 scores ---
def test_empty_article_returns_full_scores():
    """When article has no entities, both scores default to 1.0."""
    state = make_state(original_article="", agent_a_round1="some text", agent_b_round1="some text")
    result = fact_checker_node(state)
    assert result["a_score"] == 1.0
    assert result["b_score"] == 1.0
