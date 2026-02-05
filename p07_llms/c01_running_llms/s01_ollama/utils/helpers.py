import requests
from typing import List


def normalise_ollama_model(requested: str, available: list[str]) -> str | None:
    """
    Try to resolve common Ollama model naming mismatches.
    Examples:
      - llama3.2 -> llama3.2:latest (if installed)
      - llama3.2:latest -> llama3.2 (if installed)
    Returns the resolved model name if found, otherwise None.
    """
    if not requested:
        return None

    if requested in available:
        return requested

    # Try adding ':latest'
    if ":" not in requested:
        candidate = f"{requested}:latest"
        if candidate in available:
            return candidate

    # Try stripping any tag
    base = requested.split(":", 1)[0]
    if base in available:
        return base
    candidate = f"{base}:latest"
    if candidate in available:
        return candidate

    return None


def ollama_get_available_models(base_url: str = "http://localhost:07011") -> List[str]:
    """
    Get available models from Ollama API.

    Args:
        base_url: The base URL of the Ollama API (default: http://localhost:07011)

    Returns:
        A list of available model names. Returns an empty list if no models are
        installed or if the connection fails.
    """
    try:
        response = requests.get(f"{base_url}/api/tags")
        response.raise_for_status()

        data = response.json()
        models = data.get("models", [])

        if not models:
            print("No models are currently installed in Ollama.")
            return []
        available_models = [model.get("name") for model in models if model.get("name")]
        available_models = [m for m in available_models if m]
        available_models_sorted = sorted(available_models)
        return available_models_sorted

    except requests.exceptions.ConnectionError:
        print(
            f"Error: Could not connect to Ollama at {base_url}. Verify the URL and that Ollama is running and try again."
        )
        return []
    except requests.exceptions.RequestException as e:
        print(f"Error fetching models: {e}")
        return []


if __name__ == "__main__":
    print(ollama_get_available_models())
