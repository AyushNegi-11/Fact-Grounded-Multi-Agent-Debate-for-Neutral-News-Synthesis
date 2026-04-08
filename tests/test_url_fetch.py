# tests/test_url_fetch.py
# Feature: fact-grounded-multi-agent-debate
# Property 12: URL fetch failure does not invoke pipeline

from unittest.mock import MagicMock, patch
import pytest

from app import fetch_article


# --- Property 12: URL fetch failure does not invoke pipeline ---
# Validates: Requirements 1.4
@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 500, 503])
def test_non_200_returns_error(status_code):
    """Property 12: Non-200 HTTP status returns empty text and non-empty error."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    with patch("app.requests.get", return_value=mock_resp):
        text, err = fetch_article("https://example.com/article")
    assert text == ""
    assert err is not None
    assert len(err) > 0


def test_200_returns_text():
    """200 OK with HTML body returns extracted text and no error."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "<html><body><p>Hello world news article.</p></body></html>"
    with patch("app.requests.get", return_value=mock_resp):
        text, err = fetch_article("https://example.com/article")
    assert err is None
    assert "Hello world" in text


def test_request_exception_returns_error():
    """Network exception returns empty text and non-empty error."""
    import requests as req
    with patch("app.requests.get", side_effect=req.exceptions.ConnectionError("timeout")):
        text, err = fetch_article("https://example.com/article")
    assert text == ""
    assert err is not None


def test_empty_body_returns_error():
    """200 OK but empty/whitespace body returns error."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "<html><head><title>x</title></head><body>   </body></html>"
    with patch("app.requests.get", return_value=mock_resp):
        text, err = fetch_article("https://example.com/article")
    assert text == ""
    assert err is not None
