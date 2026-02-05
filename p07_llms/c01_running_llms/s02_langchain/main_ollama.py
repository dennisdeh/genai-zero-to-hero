import requests
import json


OLLAMA_API_URL = "http://localhost:07011"


def query_ollama(model: str, prompt: str, stream: bool = False) -> dict:
    """
    Send a query to the Ollama API and return the response.

    Args:
        model: The model name to use (e.g., 'llama2', 'mistral'), must be installed
        prompt: The prompt/query to send
        stream: Whether to stream the response

    Returns:
        dict: The API response
    """
    url = f"{OLLAMA_API_URL}/api/generate"

    payload = {"model": model, "prompt": prompt, "stream": stream}

    response = requests.post(url, json=payload)
    response.raise_for_status()

    if stream:
        return response
    else:
        return response.json()


def list_models() -> dict:
    """List all available models in Ollama."""
    url = f"{OLLAMA_API_URL}/api/tags"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    # Example usage
    print("Available models:")
    models = list_models()
    print(json.dumps(models, indent=2))

    print("\nSending query to Ollama...")
    result = query_ollama(
        model="qwen3:latest",  # Change to your installed model
        prompt="What is Python?",
        stream=False,
    )

    print("\nResponse:")
    print(result.get("response", "No response"))
