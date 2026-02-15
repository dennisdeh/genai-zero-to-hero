"""
Demonstrate a simple agent graph with nodes, edges, tools, loops, and no AI using LangGraph.
"""

from typing import TypedDict, Literal
from langgraph.graph import END, START, StateGraph
import random


# We define a custom state type for this dummy agent with just the allowed operations
class AgentState(TypedDict):
    message: str
    operation: Literal["sum", "multiplication"]
    values: list[float]
    result: float
    counter: int
    route: str
    threshold: int


# Initial and final nodes
def initial_node(state: AgentState) -> AgentState:
    state["message"] = "Enter a list of numbers separated by commas"
    state["counter"] = 0
    assert state["threshold"] is not None
    assert state["threshold"] > 0
    return state


def final_node(state: AgentState) -> AgentState:
    state["message"] = f"The {state['operation']} of the numbers is {state['result']}"
    return state


def append_random_number_node(state: AgentState) -> AgentState:
    """
    Add a random number between 0 and 10 to the values and increment the counter.
    """
    state["values"].append(random.randint(0, 10))
    state["counter"] += 1
    print(f"Added {state['values'][-1]} to values and incremented counter.")
    return state


def identity_node(state: AgentState) -> AgentState:
    """
    Identity function for testing purposes.
    """
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


def determine_loop(state: AgentState) -> str:
    assert state["counter"] is not None
    assert state["values"] is not None
    assert state["counter"] >= 0

    if state["counter"] > state["threshold"]:
        return "exit"
    else:
        return "loop"


def determine_branch(state: AgentState) -> str:
    """
    Determines the branch to take based on the operation in the state.
    """
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
graph.add_node("add_random_number", append_random_number_node)
graph.add_node(
    "identity", identity_node
)  # does not change the state, but is needed to link the conditional edges
graph.add_node("final", final_node)
# Add edges
graph.add_edge(START, "initial")
graph.add_edge("initial", "add_random_number")
graph.add_conditional_edges(
    "add_random_number",
    determine_loop,
    {"loop": "add_random_number", "exit": "identity"},
)
graph.add_conditional_edges(
    "identity",
    determine_branch,
    {"tool_sum": "tool_sum", "tool_multiplication": "tool_multiplication"},
)
graph.add_edge("tool_sum", "final")
graph.add_edge("tool_multiplication", "final")
graph.add_edge("final", END)

agent = graph.compile()
with open(
    "p08_agents/c00_langchain_langgraph/s00_very_simple_graphs/graph_very_simple_looping.png",
    "wb",
) as f:
    f.write(agent.get_graph().draw_mermaid_png())

# run the graph (get final state)
d = {"operation": "sum", "values": [], "threshold": 5}
print(f"Input state: {d}")
output = agent.invoke(d)
print(f"Output state: {output}")
