"""
This main file contains compiles the graph for the agent with long-term memory
and runs it.
"""

import os
import dotenv
from p07_llms.c01_running_llms.s02_langchain.utils.helpers import get_llm
from p08_agents.c00_langchain_langgraph.s05_agents_with_memory.utils.graph import (
    compile_graph,
    send_message,
)

# 1. Set up Ollama LLM and embedding objects
path_env = os.path.join("p08_agents/c00_langchain_langgraph", ".env")
dotenv.load_dotenv(path_env)
OLLAMA_BASE_URL = f"http://localhost:{os.getenv('OLLAMA_PORT_HOST')}"

llm = get_llm(
    model="qwen3:8b",
    use="ollama",
    base_url_ollama=OLLAMA_BASE_URL,
)


if __name__ == "__main__":
    # set global variables
    os.environ["LLM_TO_USE"] = "ollama"
    print("Agent app with long-term memory is initialised!")
    print(f"Using LLM backend: {os.getenv('LLM_TO_USE')}")
    debug = False

    # compile the graph
    graph = compile_graph(llm=llm)

    # visualise
    with open(
        "p08_agents/c00_langchain_langgraph/s05_agents_with_memory/graph_1.png", "wb"
    ) as f:
        f.write(graph.get_graph().draw_mermaid_png())

    # invoke the agent for a chain of interactions
    send_message(
        "Hello, I am Carl and like musicals, in particular the Lion King",
        graph=graph,
        user_id="carl42",
        debug=debug,
    )
    send_message(
        "What do I like?",
        graph=graph,
        user_id="carl42",
        debug=debug,
    )
    send_message(
        "When did I mention that?",
        graph=graph,
        user_id="carl42",
        debug=debug,
    )
    # give the agent some more information
    send_message(
        "Hi Agent, I also like to read the news",
        graph=graph,
        user_id="carl42",
        debug=debug,
    )
    send_message(
        "What do I like?",
        graph=graph,
        user_id="carl42",
        debug=debug,
    )
    send_message(
        "When did I mention that?",
        graph=graph,
        user_id="carl42",
        debug=debug,
    )
    send_message(
        "What else do you know about me?",
        graph=graph,
        user_id="carl42",
        debug=debug,
    )
    # Send a message as a different user now, check that you do not get information from the previous user
    send_message(
        "I am Bernhard, what do you know about me?",
        graph=graph,
        user_id="bernhard42",
        debug=debug,
    )
    send_message(
        "I am Bernhard, what do you know about Carl?",
        graph=graph,
        user_id="bernhard42",
        debug=debug,
    )
