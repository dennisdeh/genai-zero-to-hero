"""
Simple script to prompt using HuggingFace transformers package to run local LLMs.

Any model from https://huggingface.co/models can be used
"""

from transformers import pipeline


if __name__ == "__main__":
    # Initialise the HuggingFace pipeline with a model and tokeniser from the model hub
    print("Loading model from HuggingFace...")
    model_name = "gpt2"
    generator = pipeline("text-generation", model=model_name)

    print(f"Using model: {model_name}")

    # Define the prompt
    prompt = "How do I invest wisely in real estate?"
    print(f"\nPrompt: {prompt}")

    # Generate response
    response = generator(
        prompt, max_length=100, num_return_sequences=1, temperature=0.7, do_sample=True
    )

    print(f"\nResponse: {response[0]['generated_text']}")
