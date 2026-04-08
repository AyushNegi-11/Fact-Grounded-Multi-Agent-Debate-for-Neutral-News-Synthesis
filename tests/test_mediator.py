# tests/test_mediator.py
# Feature: fact-grounded-multi-agent-debate
# Properties 9, 10, 11

from unittest.mock import MagicMock, patch
from hypothesis import given, settings, assume
from hypothesis.strategies import text, floats

from app import mediator_node
from tests.conftest import make_state


def _mock_llm(response_text="final neutral summary"):
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = response_text
    mock_llm.invoke.return_value = mock_response
    return mock_llm


# --- Property 9: Mediator fallback uses round1 when summary is empty ---
# Validates: Requirements 7.2
@given(article=text(), round1_a=text(min_size=1), round1_b=text(min_size=1))
@settings(max_examples=100)
def test_mediator_fallback_to_round1(article, round1_a, round1_b):
    """Property 9: When agent_a_summary is empty, mediator uses agent_a_round1."""
    state = make_state(
        original_article=article,
        agent_a_round1=round1_a,
        agent_b_round1=round1_b,
        agent_a_summary="",   # empty — should fall back
        agent_b_summary="",   # empty — should fall back
        a_score=0.5,
        b_score=0.5,
    )
    captured_prompt = []

    def fake_invoke(messages):
        captured_prompt.append(messages[0].content)
        mock_resp = MagicMock()
        mock_resp.content = "final summary"
        return mock_resp

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = fake_invoke

    with patch("app.get_llm", return_value=mock_llm):
        mediator_node(state)

    assert len(captured_prompt) == 1
    assert round1_a in captured_prompt[0]
    assert round1_b in captured_prompt[0]


# --- Property 10: Transparency report dominant_agent correctness ---
# Validates: Requirements 7.7, 9.4, 9.5
@given(a=floats(min_value=0.0, max_value=1.0), b=floats(min_value=0.0, max_value=1.0))
@settings(max_examples=100)
def test_dominant_agent_correctness(a, b):
    """Property 10: dominant_agent reflects which score is higher."""
    assume(not (a != a) and not (b != b))  # exclude NaN
    state = make_state(
        original_article="test article",
        agent_a_round1="summary a",
        agent_b_round1="summary b",
        a_score=a,
        b_score=b,
    )
    with patch("app.get_llm", return_value=_mock_llm()):
        result = mediator_node(state)

    tr = result["transparency_report"]
    if a > b:
        assert tr["dominant_agent"] == "Challenger"
    elif b > a:
        assert tr["dominant_agent"] == "Supporter"
    else:
        assert tr["dominant_agent"] == "Equal Contribution"


# --- Property 11: Transparency report score_delta correctness ---
# Validates: Requirements 7.7, 9.7
@given(a=floats(min_value=0.0, max_value=1.0), b=floats(min_value=0.0, max_value=1.0))
@settings(max_examples=100)
def test_score_delta_correctness(a, b):
    """Property 11: score_delta == abs(a_score - b_score)."""
    assume(not (a != a) and not (b != b))  # exclude NaN
    state = make_state(
        original_article="test article",
        agent_a_round1="summary a",
        agent_b_round1="summary b",
        a_score=a,
        b_score=b,
    )
    with patch("app.get_llm", return_value=_mock_llm()):
        result = mediator_node(state)

    tr = result["transparency_report"]
    assert abs(tr["score_delta"] - abs(a - b)) < 1e-9


# --- Transparency report structure ---
def test_transparency_report_has_all_fields():
    """Mediator returns transparency_report with all required fields."""
    state = make_state(
        original_article="article",
        agent_a_round1="summary a",
        agent_b_round1="summary b",
        a_score=0.6,
        b_score=0.4,
        iteration=1,
    )
    with patch("app.get_llm", return_value=_mock_llm()):
        result = mediator_node(state)

    tr = result["transparency_report"]
    assert "a_score" in tr
    assert "b_score" in tr
    assert "iteration" in tr
    assert "dominant_agent" in tr
    assert "score_delta" in tr
    assert "final_summary" in result
