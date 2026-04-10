"""
utils/tools.py
Helper functions to execute DuckDuckGo search or parse URLs.
"""
import re
import requests
import streamlit as st
from ddgs import DDGS

def auto_fetch_text(input_text: str) -> str:
    """If the input is a URL, strip the HTML and fetch the text. Otherwise return as is."""
    text = input_text.strip()
    if (text.startswith("http://") or text.startswith("https://")) and " " not in text:
        try:
            resp = requests.get(text, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            resp.raise_for_status()
            html = resp.text
            html = re.sub(r'<script.*?</script>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<style.*?</style>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
            clean_text = re.sub(r'<[^>]+>', ' ', html)
            return re.sub(r'\s+', ' ', clean_text).strip()
        except Exception as e:
            st.error(f"Failed to fetch article from URL: {e}")
            return input_text
    return input_text

def perform_web_search(query: str) -> str:
    """Uses DuckDuckGo to perform a web search and returns a text summary."""
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = DDGS().text(query, max_results=3)
        return " | ".join([res["body"] for res in results]) if results else "No external context found."
    except Exception as e:
        return f"Search failed: {str(e)}"
