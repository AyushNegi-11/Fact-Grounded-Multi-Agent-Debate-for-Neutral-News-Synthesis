# tests/test_routing.py
# Feature: fact-grounded-multi-agent-debate
# Properties 5, 6

from hypothesis import given, settings
from hypothesis.strategies import integers, floats

from app import mediation_router
from tests.conftest import make_state


# --- Property 5: Routing correctness — rewrite condition ---
# Validates: Requirements 6.2
@given(
    iteration=integers(min_value=0, max_value=2),
    a_score=floats(min_value=0.0, max_value=0.1499),
    b_score=floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=100)
def test_router_rewrite_when_a_low(iteration, a_score, b_score):
    """Property 5: Routes to rewrite when iteration<=2 and a_score < 0.15."""
    state = make_state(iteration=iteration, a_score=a_score, b_score=b_score)
    assert mediation_router(state) == "rewrite"


@given(
    iteration=integers(min_value=0, max_value=2),
    a_score=floats(min_value=0.0, max_value=1.0),
    b_score=floats(min_value=0.0, max_value=0.1499),
)
@settings(max_examples=100)
def test_router_rewrite_when_b_low(iteration, a_score, b_score):
    """Property 5: Routes to rewrite when iteration<=2 and b_score < 0.15."""
    state = make_state(iteration=iteration, a_score=a_score, b_score=b_score)
    assert mediation_router(state) == "rewrite"


# --- Property 6: Routing correctness — mediator condition ---
# Validates: Requirements 6.3

@given(
    iteration=integers(min_value=3, max_value=10),
    a_score=floats(min_value=0.0, max_value=1.0),
    b_score=floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=100)
def test_router_mediator_when_iteration_exceeded(iteration, a_score, b_score):
    """Property 6: Routes to mediator when iteration > 2 regardless of scores."""
    state = make_state(iteration=iteration, a_score=a_score, b_score=b_score)
    assert mediation_router(state) == "mediator"


@given(
    iteration=integers(min_value=0, max_value=2),
    a_score=floats(min_value=0.15, max_value=1.0),
    b_score=floats(min_value=0.15, max_value=1.0),
)
@settings(max_examples=100)
def test_router_mediator_when_both_scores_pass(iteration, a_score, b_score):
    """Property 6: Routes to mediator when both scores >= 0.15."""
    state = make_state(iteration=iteration, a_score=a_score, b_score=b_score)
    assert mediation_router(state) == "mediator"
