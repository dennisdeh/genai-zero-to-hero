"""
Demonstrate a simple stock analysis ReAct agent using LangGraph with a chat interface in the terminal.
"""

import os
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from p07_llms.c01_running_llms.s02_langchain.utils.helpers import get_llm


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


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
        return f"Financial Analysis for {ticker_upper}:\n{mock_data[ticker_upper]}"
    else:
        return (
            f"Unable to find financial data for ticker: {ticker}. "
            f"Please try AAPL, GOOGL, or MSFT."
        )


def chatbot(state: AgentState) -> AgentState:
    """
    Process messages and interact with the LLM.
    """
    llm = get_llm()
    tools = [get_stock_analysis]
    llm_with_tools = llm.bind_tools(tools)
    print(state["messages"])
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def compile_graph():
    # Build the graph
    graph_builder = StateGraph(AgentState)

    # Add nodes
    graph_builder.add_node("chatbot", chatbot)
    tool_node = ToolNode(tools=[get_stock_analysis])
    graph_builder.add_node("tools", tool_node)

    # Add edges
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges(
        "chatbot",
        lambda state: "tools" if state["messages"][-1].tool_calls else END,
        {"chatbot": "chatbot", "tools": "tools", END: END},
    )
    graph_builder.add_edge("tools", "chatbot")

    # Add memory
    memory = InMemorySaver()

    # Compile the graph
    return graph_builder.compile(checkpointer=memory)


if __name__ == "__main__":
    # set global variables
    os.environ["LLM_TO_USE"] = "ollama"
    print("Stock Analysis Agent chat initialised!")
    print(f"Using LLM backend: {os.getenv('LLM_TO_USE')}")

    # instantiate the language graph
    graph = compile_graph()

    # visualise
    with open(
        "p08_agents/c00_langchain_langgraph/s01_simple_agents/graph_5.png", "wb"
    ) as f:
        f.write(graph.get_graph().draw_mermaid_png())

    while True:
        """
        Try e.g.: How is the performance of AAPL compared to GOOGL

        Then (to illustrate the memory) try: Which stocks have you analysed?

        Then (to illustrate limitations of the tool): How about Tesla?
        """  #
        prompt = input("Message for the Agent: ")
        result = graph.invoke(
            {"messages": [HumanMessage(content=prompt)]},
            config={"configurable": {"thread_id": 42}},
        )
        print(result["messages"][-1].content)
