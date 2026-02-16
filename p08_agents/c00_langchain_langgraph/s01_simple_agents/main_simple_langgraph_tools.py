"""
Demonstrate a simple chat agent using LangGraph with a simple set of tools for basic tasks.
"""

import os
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from p07_llms.c01_running_llms.s02_langchain.utils.helpers import get_llm


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# define tools
@tool
def tool_sum(values: list[float]) -> AgentState:
    """
    Sums all the numbers in the list 'values' and returns the result.
    """
    vals = [float(val) for val in values]
    result = sum(vals)
    return result


@tool
def tool_multiplication(values: list[float]) -> AgentState:
    """
    Multiplies all the numbers in the list 'values' and returns the result.
    """
    vals = [float(val) for val in values]
    result = 1.0
    for val in vals:
        result *= val
    return result


def determine_loop(state: AgentState) -> str:
    """
    Determines if more tools are needed based on the operation in the state.
    """
    if state["messages"][-1].tool_calls:
        return "loop"
    else:
        return "exit"


def chatbot_with_tools(state: AgentState) -> AgentState:
    """
    Process messages and interact with the LLM bound with the tools.
    """
    # get llm and invoke
    llm = get_llm()
    # bind tools to LLM
    llm_with_tools = llm.bind_tools([tool_sum, tool_multiplication])
    response = llm_with_tools.invoke(state["messages"])
    # appends the response to the state as we are using Annotated
    return {"messages": [response]}


if __name__ == "__main__2":
    # set global variables
    os.environ["LLM_TO_USE"] = "ollama"
    print("Simple agent with tools initialised!")
    print(f"Using LLM backend: {os.getenv('LLM_TO_USE')}")

    # initialise building the language graph
    graph = StateGraph(AgentState)
    # add nodes
    graph.add_node("chatbot", chatbot_with_tools)
    tools_node = ToolNode(tools=[tool_sum, tool_multiplication])
    graph.add_node("tools", tools_node)
    # add edges
    graph.add_edge(START, "chatbot")
    graph.add_conditional_edges(
        "chatbot", determine_loop, {"loop": "tools", "exit": END}
    )
    # compile the graph
    agent = graph.compile()

    # invoke the agent for a given initial state
    inputs = [
        HumanMessage(
            content="Multiply these numbers: 2, 3, 4. Then add 1 to the result. Multiply this number by 4."
        )
    ]
    out = agent.invoke({"messages": inputs})
    for output in agent.stream({"messages": inputs}):
        print(output)
