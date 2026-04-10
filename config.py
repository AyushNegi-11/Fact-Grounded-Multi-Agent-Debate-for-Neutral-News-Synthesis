"""
config.py
Handles loading of ML models and environment variables.
"""
import os
import logging
import warnings
import functools
import spacy
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Suppress noise
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message=r".*duckduckgo_search.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*torch\.jit\.script.*")
warnings.filterwarnings("ignore", message=r".*HF_TOKEN.*")

load_dotenv()

@functools.lru_cache(maxsize=1)
def load_spacy_model():
    try:
        return spacy.load("en_core_web_md")
    except OSError:
        from spacy.cli import download
        download("en_core_web_md")
        return spacy.load("en_core_web_md")

@functools.lru_cache(maxsize=1)
def load_nli_model():
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from sentence_transformers import CrossEncoder
            return CrossEncoder("cross-encoder/nli-deberta-v3-small")
    except Exception:
        return None

def get_llm(model_alias: str = "A", max_tokens: int = None) -> ChatGroq:
    """Returns a ChatGroq LLM based on environment alias."""
    model_env_key = f"GROQ_MODEL_{model_alias}"
    model_name = os.getenv(model_env_key, "llama-3.3-70b-versatile")
    kwargs = {"model": model_name, "temperature": 0.5}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    return ChatGroq(**kwargs)
