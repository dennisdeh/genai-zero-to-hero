"""
Demonstrate a simple agent graph with nodes, edges, tools, and no AI using LangGraph.
"""

from typing import TypedDict, Literal
from langgraph.graph import END, START, StateGraph


# We define a custom state type for this dummy agent with just the allowed operations
class AgentState(TypedDict):
    message: str
    operation: Literal["sum", "multiplication"]
    values: list[float]
    result: float


# Initial and final nodes
def initial_node(state: AgentState) -> AgentState:
    state["message"] = "Enter a list of numbers separated by spaces"
    return state


def final_node(state: AgentState) -> AgentState:
    state["message"] = f"The {state['operation']} of the numbers is {state['result']}"
    return state


# Tool nodes
def tool_sum(state: AgentState) -> AgentState:
    """
    Sums the values in the "values" key in the current state and returns
    a new state with the result in the "result" key.
    """
    vals = [float(val) for val in state["values"]]
    result = sum(vals)

    state["result"] = result

    return state


def tool_multiplication(state: AgentState) -> AgentState:
    """
    Multiplies the values in the "values" key in the current state and returns
    a new state with the result in the "result" key.
    """
    vals = [float(val) for val in state["values"]]
    result = 1.0
    for val in vals:
        result *= val

    state["result"] = result

    return state


def determine_branch(state: AgentState) -> str:
    """
    Determines the branch to take based on the operation in the state.
    """
    ret = ""
    if state["operation"] == "sum":
        ret = "tool_sum"
    elif state["operation"] == "multiplication":
        ret = "tool_multiplication"
    else:
        raise ValueError(f"Invalid operation: {state['operation']}")

    return ret


# Define graph object
graph = StateGraph(AgentState)
# Add nodes
graph.add_node("initial", initial_node)
graph.add_node("tool_sum", tool_sum)
graph.add_node("tool_multiplication", tool_multiplication)
graph.add_node("final", final_node)
# Add edges
graph.add_edge(START, "initial")
graph.add_conditional_edges("initial", determine_branch)
graph.add_edge("tool_sum", "final")
graph.add_edge("tool_multiplication", "final")
graph.add_edge("final", END)

agent = graph.compile()
with open(
    "p08_agents/c00_langchain_langraph/s00_very_simple_graph/graph_very_simple.png",
    "wb",
) as f:
    f.write(agent.get_graph().draw_mermaid_png())

# run the graph (get final state)
d = {"operation": "sum", "values": [1, 2, 3, 4]}
print(f"Input state: {d}")
output = agent.invoke(d)
print(f"Output state: {output}")
