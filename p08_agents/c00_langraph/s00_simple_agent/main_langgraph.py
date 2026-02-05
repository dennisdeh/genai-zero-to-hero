"""
Demonstrate a simple stock analysis agent using LangGraph.
"""

import os
from typing import Literal, Union

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from p07_llms.c01_running_llms.s01_ollama.utils.helpers import (
    ollama_get_available_models,
    normalise_ollama_model,
)


@tool
def get_stock_analysis(ticker: str) -> str:
    """
    Perform financial analysis on a given stock ticker.

    Args:
        ticker: The stock ticker symbol (e.g. 'AAPL', 'GOOGL')

    Returns:
        A string containing financial analysis data
    """
    # Mock financial data - in production, this would call a real API
    mock_data = {
        "AAPL": "Apple Inc. - Current Price: $175.23, P/E Ratio: 28.5, Market Cap: $2.8T, 52-week range: $164-198",
        "GOOGL": "Alphabet Inc. - Current Price: $140.15, P/E Ratio: 25.3, Market Cap: $1.7T, 52-week range: $120-155",
        "MSFT": "Microsoft Corp. - Current Price: $380.50, P/E Ratio: 32.1, Market Cap: $2.9T, 52-week range: $350-405",
    }

    ticker_upper = ticker.upper()
    if ticker_upper in mock_data:
        return f"Financial Analysis for {ticker_upper}:\n{mock_data[ticker_upper]}\nNote: Strong fundamentals, consider long-term investment."
    else:
        return (
            f"Unable to find financial data for ticker: {ticker}. "
            f"Please try AAPL, GOOGL, or MSFT."
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


def chatbot(state: MessagesState):
    """
    Process messages and interact with the LLM.
    """
    llm = get_llm()
    tools = [get_stock_analysis]
    llm_with_tools = llm.bind_tools(tools)
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def compile_graph():
    # Build the graph
    graph_builder = StateGraph(MessagesState)

    # Add nodes
    graph_builder.add_node("chatbot", chatbot)
    tool_node = ToolNode(tools=[get_stock_analysis])
    graph_builder.add_node("tools", tool_node)

    # Add edges
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges(
        "chatbot",
        lambda state: "tools" if state["messages"][-1].tool_calls else END,
    )
    graph_builder.add_edge("tools", "chatbot")

    # Compile the graph
    return graph_builder.compile()


if __name__ == "__main__":
    # set global variables
    os.environ["LLM_TO_USE"] = "ollama"

    print("Stock Analysis Agent initialised!")
    print(f"Using LLM: {os.getenv('LLM_TO_USE')}")
    str_query = "Can you analyze Apple stock (AAPL) for me?"
    print(f"Asking: {str_query}")

    messages = [HumanMessage(content=str_query)]

    # instantiate the language graph
    graph = compile_graph()

    for event in graph.stream({"messages": messages}):
        for value in event.values():
            if isinstance(value["messages"][-1], AIMessage):
                print(f"Agent: {value['messages'][-1].content}")
