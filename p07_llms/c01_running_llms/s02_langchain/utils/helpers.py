import os
from typing import Literal, Union
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from p07_llms.c01_running_llms.s01_ollama.utils.helpers import (
    ollama_get_available_models,
    normalise_ollama_model,
)


def get_llm(
    model: Union[str, None] = None,
    use: Literal["openai", "ollama"] = "ollama",
    base_url_ollama: str = "http://localhost:07011",
):
    """
    Initialise and return the selected LLM. If environment variables are set,
    they will be used to override the model and base URL.
    """
    use = os.getenv("LLM_TO_USE", use).lower()
    if use not in ["openai", "ollama"]:
        raise ValueError(f"Invalid LLM selection: {use}. Must be 'openai' or 'ollama'")

    if use == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        model = model or "gpt-4o-mini"
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return ChatOpenAI(model=model, temperature=0)

    elif use == "ollama":
        # Allow environment overrides without having to plumb parameters through the graph nodes.
        base_url_ollama = os.getenv("OLLAMA_BASE_URL", base_url_ollama)
        model = model or os.getenv("OLLAMA_MODEL")
        available_models = ollama_get_available_models(base_url=base_url_ollama)

        if model:
            resolved = normalise_ollama_model(model, available_models)
            if resolved is None:
                fallback = available_models[0]
                print(
                    f"Warning: Requested Ollama model '{model}' is not installed/available. "
                    f"Falling back to '{fallback}'. Available models: {available_models}"
                )
                model = fallback
            else:
                model = resolved
        else:
            model = available_models[0]

        return ChatOllama(model=model, base_url=base_url_ollama, temperature=0)
    else:
        raise ValueError(f"Invalid LLM selection: {use}")
