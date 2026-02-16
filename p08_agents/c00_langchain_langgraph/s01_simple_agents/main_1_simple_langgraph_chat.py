"""
Demonstrate a simple chat agent using LangGraph - without memory and tools
"""

import os
from typing import TypedDict, List
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from p07_llms.c01_running_llms.s02_langchain.utils.helpers import get_llm


class AgentState(TypedDict):
    messages: List[HumanMessage]


def chatbot(state: AgentState) -> AgentState:
    """
    Process messages and interact with the LLM.
    """
    # get llm and invoke
    llm = get_llm()
    response = llm.invoke(state["messages"])
    # append just the core reply from the LLM, not all the meta-data
    state["messages"].append(AIMessage(content=response.content))
    return state


if __name__ == "__main__":
    # set global variables
    os.environ["LLM_TO_USE"] = "ollama"

    print("Simple agent without memory or tools initialised!")
    print(f"Using LLM backend: {os.getenv('LLM_TO_USE')}")

    # initialise building the language graph
    graph = StateGraph(AgentState)
    # add nodes
    graph.add_node("chatbot", chatbot)
    # add edges
    graph.add_edge(START, "chatbot")
    graph.add_edge("chatbot", END)
    # compile the graph
    agent = graph.compile()

    # visualise
    with open(
        "p08_agents/c00_langchain_langgraph/s01_simple_agents/graph_1.png", "wb"
    ) as f:
        f.write(agent.get_graph().draw_mermaid_png())

    # chat loop (the state is overwritten each time, hence no memory)
    while True:
        str_input = input("Enter a message: ")
        messages = [HumanMessage(content=str_input)]
        print(f"current state: {messages}")
        output = agent.invoke({"messages": messages})
        print(f"Agent: {output['messages'][-1].content}")
        print(f"current state: {messages}")
