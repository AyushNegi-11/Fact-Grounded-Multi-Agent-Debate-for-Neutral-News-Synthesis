# tests/test_agents.py
# Feature: fact-grounded-multi-agent-debate
# Properties 7, 8

from unittest.mock import MagicMock, patch
from hypothesis import given, settings
from hypothesis.strategies import text, integers

from app import agent_a_node, agent_b_node
from tests.conftest import make_state


def _mock_llm(response_text="mocked response"):
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = response_text
    mock_llm.invoke.return_value = mock_response
    return mock_llm


# --- Property 7: Agent A Round 1 stores to agent_a_round1 ---
# Validates: Requirements 3.4
@given(article=text())
@settings(max_examples=100)
def test_agent_a_round1_stores_to_correct_field(article):
    """Property 7: When iteration==0, agent_a_node returns agent_a_round1 key only."""
    state = make_state(original_article=article, iteration=0)
    with patch("app.get_llm", return_value=_mock_llm("challenger output")):
        result = agent_a_node(state)
    assert "agent_a_round1" in result
    assert "agent_a_summary" not in result
    assert result["agent_a_round1"] == "challenger output"


# --- Property 8: Agent A Round 2 stores to agent_a_summary ---
# Validates: Requirements 3.6
@given(article=text(), iteration=integers(min_value=1, max_value=5))
@settings(max_examples=100)
def test_agent_a_round2_stores_to_summary_field(article, iteration):
    """Property 8: When iteration>0, agent_a_node returns agent_a_summary key only."""
    state = make_state(original_article=article, iteration=iteration)
    with patch("app.get_llm", return_value=_mock_llm("revised challenger output")):
        result = agent_a_node(state)
    assert "agent_a_summary" in result
    assert "agent_a_round1" not in result
    assert result["agent_a_summary"] == "revised challenger output"


# --- Agent B: Round 1 stores to agent_b_round1 ---
@given(article=text())
@settings(max_examples=100)
def test_agent_b_round1_stores_to_correct_field(article):
    """Agent B Round 1 stores to agent_b_round1."""
    state = make_state(original_article=article, iteration=0)
    with patch("app.get_llm", return_value=_mock_llm("supporter output")):
        result = agent_b_node(state)
    assert "agent_b_round1" in result
    assert "agent_b_summary" not in result


# --- Agent B: Round 2 stores to agent_b_summary ---
@given(article=text(), iteration=integers(min_value=1, max_value=5))
@settings(max_examples=100)
def test_agent_b_round2_stores_to_summary_field(article, iteration):
    """Agent B Round 2 stores to agent_b_summary."""
    state = make_state(original_article=article, iteration=iteration)
    with patch("app.get_llm", return_value=_mock_llm("revised supporter output")):
        result = agent_b_node(state)
    assert "agent_b_summary" in result
    assert "agent_b_round1" not in result
