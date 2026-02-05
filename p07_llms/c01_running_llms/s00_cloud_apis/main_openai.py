import os
from openai import OpenAI

# Load API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)


def chat_completion(prompt: str, model: str = "gpt-5.2") -> str:
    """
    Send a chat completion request to OpenAI API.

    Args:
        prompt: The user prompt to send
        model: The model to use (default: gpt-5.2)

    Returns:
        The response content from the API
    """
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # Example usage
    prompt = "How do I invest wisely in real estate?"
    response = chat_completion(prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
