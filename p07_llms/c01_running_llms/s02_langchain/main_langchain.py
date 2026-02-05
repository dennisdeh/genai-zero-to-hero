"""
Simple script to connect to either OpenAI or Ollama using LangChain.
"""

import os
from p07_llms.c01_running_llms.s02_langchain.utils.helpers import get_llm


if __name__ == "__main__":
    # Set which LLM to use
    os.environ["LLM_TO_USE"] = "ollama"
    print(f"Using LLM: {os.getenv('LLM_TO_USE')}")

    # Initialize the LLM
    llm = get_llm()

    # Send a prompt
    prompt = "What is LangChain?"
    print(f"\nPrompt: {prompt}")

    # Invoke the LLM and get response
    response = llm.invoke(prompt)

    print(f"\nResponse: {response.content}")
