"""
Demonstrate a single agent setup with a human in the loop that approves
the multiplication calculations using the HumanInTheLoopMiddleware in LangGraph.

We define a math agent
- The use of tool_multiplication needs to be approved by the human in the loop:  It can
    be approved, edited, or rejected.
- Memory checkpointing is used to save the state of the math agent

Run the script in the terminal.
"""

import os
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain_core.tools import tool
import dotenv
from p07_llms.c01_running_llms.s02_langchain.utils.helpers import get_llm

# ----------------------------
# Config
# ----------------------------
path_env = os.path.join("p08_agents/c00_langchain_langgraph", ".env")
dotenv.load_dotenv(path_env)
# URLs
OLLAMA_BASE_URL = f"http://localhost:{os.getenv('OLLAMA_PORT_HOST')}"

# 1. Set up Ollama LLM and Agent state dict
llm = get_llm(model="qwen3:8b", use="ollama", base_url_ollama=OLLAMA_BASE_URL)


@tool
def tool_sum(
    values: list[float], add_description: bool = False, prefix_message: str = ""
) -> str:
    """
    Computes the sum of a list of numerical values and optionally formats the result
    with a description and prefix message.

    This function takes a list of numerical values, calculates their sum, and returns
    the result as a string. If desired, it can include a description of the operation
    and a custom prefix message as part of the output.

    :param values: A list of numerical values whose sum is to be calculated.
    :type values: list[float]
    :param add_description: Specifies whether to include a description of the result
        in the returned string. Default is False.
    :type add_description: bool
    :param prefix_message: An optional prefix message to include in the output.
        Only used when ``add_description`` is True. Default is an empty string.
    :type prefix_message: str
    :return: The computed sum as a string. If ``add_description`` is True, the result
        is formatted with the description and prefix message.
    :rtype: str
    """
    vals = [float(val) for val in values]
    result = sum(vals)
    if add_description:
        return (
            f"{prefix_message} The sum of all values in the list {values} is: {result}"
        )
    else:
        return str(result)


@tool
def tool_multiplication(
    values: list[float], add_description: bool = False, prefix_message: str = ""
) -> str:
    """
    Calculates the multiplication of all numerical values in a given list and returns the result as a
    string. Optionally, appends a descriptive message including the result, prefixed by a custom message.

    :param values: A list of numerical values (floats) for which the multiplication is to be calculated.
    :param add_description: A boolean indicating whether a descriptive message should be included in the
        output.
    :param prefix_message: A custom prefix message to be included as part of the descriptive output if
        add_description is set to True.
    :return: A string representing either the result of the multiplication or a descriptive message
        containing the result prefixed with the provided custom message.
    """
    vals = [float(val) for val in values]
    result = 1.0
    for val in vals:
        result *= val
    if add_description:
        return f"{prefix_message} The multiplication of all values in the list {values} is: {result}"
    else:
        return str(result)


# Define math agent
agent_math = create_agent(
    model=llm,
    name="Math Agent",
    tools=[tool_sum, tool_multiplication],
    system_prompt=(
        "You are a math agent."
        "You can only answer questions about math and must ALWAYS use the tools provided. "
        "Never calculate a sum or multiply without using the tools. "
        "Format all outputs in markdown docstrings"
    ),
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "tool_multiplication": {
                    "allowed_decisions": ["approve", "edit", "reject"]
                }
            },
        ),
    ],
)

if __name__ == "__main__":
    # set global variables
    os.environ["LLM_TO_USE"] = "ollama"
    print("Math agent app with HITL is initialised!")
    print(f"Using LLM backend: {os.getenv('LLM_TO_USE')}")
    config = {"configurable": {"thread_id": "1"}}
    debug = False

    # draw the graph
    with open(
        "p08_agents/c00_langchain_langgraph/s01_simple_agents/graph_6_math_agent_hitl.png",
        "wb",
    ) as f:
        f.write(agent_math.get_graph().draw_mermaid_png())

    # Submit a query to the agent
    str_query = "Sum the numbers [1,2,3] and multiply the numbers [2,3,4,5]"

    interrupts = []
    for step in agent_math.stream(
        {"messages": [HumanMessage(content=str_query)]}, config=config, debug=debug
    ):
        for update in step.values():
            if isinstance(update, dict):
                for message in update.get("messages", []):
                    message.pretty_print()
            else:
                interrupt_ = update[0]
                interrupts.append(interrupt_)
                print(f"\nINTERRUPTED: {interrupt_.id}")

    resume = {}
    if len(interrupts) > 0:
        resume["decisions"] = list()
        for interrupt_ in interrupts:
            print(interrupt_.value["action_requests"][0]["description"])
            approval = input("Approve? (y/n/edit): ")
            if approval == "y":
                resume["decisions"].append(
                    {"type": "approve", "message": "Approved by user."}
                )
            elif approval == "n":
                resume["decisions"].append(
                    {
                        "type": "reject",
                        "message": "The previous multiplication request has been cancelled by a human intervention. "
                        "Do not use the original multiplication instruction or its original values.",
                    }
                )
            elif approval == "edit":
                edited_action = {}
                str_new_values = input("Enter new values to multiply: ")
                ls_new = str_new_values.split(",")
                edited_values = [float(x.strip()) for x in ls_new]

                # set up the structure of the edited action
                edited_action["name"] = "tool_multiplication"
                edited_action["args"] = {}
                edited_action["args"]["values"] = edited_values
                edited_action["args"]["add_description"] = True
                edited_action["args"]["prefix_message"] = (
                    f"The first multiplication request has been replaced by a human edit. "
                    f"Do not use the original multiplication instruction or its original values. "
                    f"Use only these updated values for tool_multiplication: {edited_values}. "
                    "Inform the user that the original multiplication request has been replaced by a human edit. "
                )

                resume["decisions"].append(
                    {
                        "type": "edit",
                        "edited_action": edited_action,
                    }
                )
            else:
                raise ValueError("Invalid input. Please enter 'y', 'n', or 'edit'.")

    interrupts = []
    for step in agent_math.stream(Command(resume=resume), config=config, debug=debug):
        for update in step.values():
            if isinstance(update, dict):
                for message in update.get("messages", []):
                    message.pretty_print()
            elif update is not None:
                print(update)
                interrupt_ = update[0]
                interrupts.append(interrupt_)
                print(f"\nINTERRUPTED: {interrupt_.id}")
