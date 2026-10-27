"""
Small helpers shared by the prompt engineering and structured output examples.

All LLM access goes through `get_llm()` from chapter c01 (local Ollama by default, OpenAI when the
environment variable LLM_TO_USE=openai). No .env file is needed: the Ollama URL is read from
OLLAMA_BASE_URL (default http://localhost:07011) and the model from OLLAMA_MODEL (default qwen3:8b).
"""

import os
import re
from typing import Any, Union

import requests
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

from p07_llms.c01_running_llms.s02_langchain.utils.helpers import get_llm

DEFAULT_OLLAMA_BASE_URL = "http://localhost:07011"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"


def get_chat_model(
    temperature: Union[float, None] = None,
    think: bool = False,
    model: Union[str, None] = None,
    **ollama_options: Any,
) -> BaseChatModel:
    """
    Return a chat model from `get_llm()`, optionally with a different temperature.

    Args:
        temperature: Sampling temperature. None keeps the default of get_llm() (0, i.e. greedy-ish).
        think: For Ollama "thinking" models such as qwen3: True lets the model produce its hidden reasoning
            first (slower), False switches the thinking mode off so we get a direct answer.
        model: Ollama model name; default is OLLAMA_MODEL or qwen3:8b. Ignored for OpenAI (get_llm's default).
        **ollama_options: Extra ChatOllama fields (e.g. num_ctx=8192, num_predict=256). Ignored for OpenAI.

    Returns:
        A LangChain chat model (ChatOllama or ChatOpenAI).
    """
    use = os.getenv("LLM_TO_USE", "ollama").lower()
    model = (
        (model or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL))
        if use == "ollama"
        else None
    )
    if use == "ollama" and not is_ollama_available():
        raise RuntimeError(
            f"No Ollama models reachable at {os.getenv('OLLAMA_BASE_URL', DEFAULT_OLLAMA_BASE_URL)}. "
            "Start p07_llms/c01_running_llms/s01_ollama/docker-compose.yml and pull a model "
            "(e.g. qwen3:8b), or set LLM_TO_USE=openai and OPENAI_API_KEY."
        )
    llm = get_llm(
        model=model,
        use=use,
        base_url_ollama=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
    )

    updates: dict[str, Any] = {}
    if temperature is not None:
        updates["temperature"] = temperature
    if isinstance(llm, ChatOllama):
        updates["reasoning"] = think
        updates.update(ollama_options)
    # pydantic's model_copy gives us a configured copy without touching get_llm()
    return llm.model_copy(update=updates) if updates else llm


def is_ollama_available() -> bool:
    """
    Return True if the Ollama server answers and has at least one model installed.
    """
    base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=3)
        response.raise_for_status()
        return len(response.json().get("models", [])) > 0
    except (requests.exceptions.RequestException, ValueError):
        return False


def is_llm_available() -> bool:
    """
    True if the selected backend can be used: OpenAI with an API key set, or a reachable Ollama server.
    """
    if os.getenv("LLM_TO_USE", "ollama").lower() == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    return is_ollama_available()


def strip_think(text: str) -> str:
    """
    Remove <think>...</think> blocks that some reasoning models (e.g. qwen3) put in the answer text.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
