import requests
import json
from p07_llms.c01_running_llms.s01_ollama.utils.helpers import (
    ollama_get_available_models,
    normalise_ollama_model,
)


def query_ollama(
    model: str, prompt: str, stream: bool = False, url="http://localhost:07011"
) -> dict:
    """
    Send a query to the Ollama API and return the response.

    Args:
        model: The model name to use (e.g., 'llama2', 'mistral'), must be installed
        prompt: The prompt/query to send
        stream: Whether to stream the response
        url: The base URL of the Ollama API

    Returns:
        dict: The API response
    """
    url = f"{url}/api/generate"

    payload = {"model": model, "prompt": prompt, "stream": stream}

    response = requests.post(url, json=payload)
    response.raise_for_status()

    if stream:
        return response
    else:
        return response.json()


if __name__ == "__main__":
    # Example usage
    base_url = "http://localhost:07011"
    model = "qwen3"
    prompt = "What is Python?"

    models = ollama_get_available_models(base_url=base_url)
    model = normalise_ollama_model(model, models)
    print(f"Using model: {model}")

    print("\nSending query to Ollama...")
    result = query_ollama(
        model=model,
        prompt=prompt,
        stream=False,
    )

    print("\nResponse:")
    print(result.get("response", "No response"))
