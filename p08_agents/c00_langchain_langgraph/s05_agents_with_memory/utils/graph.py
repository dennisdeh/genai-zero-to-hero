"""
This file contains the graph for the memory agent, using an InMemoryStore to save the
memories (non-persistent, will be reset on restart) and an InMemorySaver to save the checkpoints.

Portions of this file are derived from:
https://medium.com/@anil.jain.baba/long-term-agentic-memory-with-langgraph-824050b09852
Original author: Anil Jain Baba
Changes: Adaption to use Ollama, compatability with the newest version of LangGraph,
no use of sqllite, in-memory store and checkpointer.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from p08_agents.c00_langchain_langgraph.s05_agents_with_memory.utils.agent_state import (
    AgentState,
)
from p08_agents.c00_langchain_langgraph.s05_agents_with_memory.utils.memory import (
    upsert_memory,
    store_memory,
    retrieve_memories,
)


def llm_chain(llm: BaseChatModel):
    """Create a LangChain LLM chain with memory support for a given LLM model object.

    This function sets up a LangChain LLM chain that includes memory capabilities
    and integrates with Ollama for model execution. It configures the LLM with
    the specified model and tools.
    """
    # bind tools to LLM
    llm = llm.bind_tools([upsert_memory])
    # define the system prompt
    str_system_prompt = """
    You are an AI assistant with access to long-term memory.
    Use stored memories to personalize responses when relevant.
    Identify durable information about the user (preferences, goals, projects, habits).
    Store useful long-term facts using the `upsert_memory` tool.
    Write memories as short, atomic, third-person statements.
    Avoid saving temporary context or sensitive information.
    Update existing memories instead of duplicating them.
    Never invent memories.
    Respect user requests to view, modify, or delete stored memories.
    """
    # define the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt}"),
            ("placeholder", "{messages}"),
        ]
    )
    # create a RunnablePassthrough with the system prompt that includes the memory context when available
    chain = (
        RunnablePassthrough.assign(
            system_prompt=lambda x: (
                str_system_prompt
                + "\n\nMemory Context:\n"
                + "\n".join(
                    m.content if hasattr(m, "content") else str(m)
                    for m in x.get("memory_context", [])
                )
            )
        )
        | prompt
        | llm
    )
    return chain


def compile_graph(llm: BaseChatModel):
    """
    Creates and returns a compiled state graph for managing memory processes.

    The function initialises an in-memory store and a checkpointer, then builds
    a state graph with defined nodes and transitions for retrieving, processing,
    and storing memory-related states. Conditional edges are used to route
    state transitions based on specific logic derived from the current state.
    The resulting graph is compiled with the defined checkpointer.

    :param llm: The base chat model used to generate the response. It serves as
    the core language processing system in this method.
    :type llm: BaseChatModel
    :return: A compiled state graph that manages memory processes, transitions,
        and actions executed at various states.
    :rtype: StateGraph
    """
    # Initialise store
    memory_store = InMemoryStore()
    # Initialise checkpointer
    checkpointer = InMemorySaver()
    # Build graph
    builder = StateGraph(AgentState)

    # Define special node functions
    def retrieve_memories_node(state: AgentState):
        return retrieve_memories(state, store=memory_store)

    def store_memory_node(state: AgentState):
        return store_memory(state, store=memory_store)

    def call_model(state: AgentState):
        llmc = llm_chain(llm=llm)
        response = llmc.invoke(state)
        return {"messages": [response]}

    # add nodes
    builder.add_node("retrieve_memories", retrieve_memories_node)
    builder.add_node("call_model", call_model)
    builder.add_node("store_memory", store_memory_node)

    # define edges
    builder.add_edge(START, "retrieve_memories")
    builder.add_edge("retrieve_memories", "call_model")

    # define conditional edges for handling tool calls
    def route_after_model(state: AgentState):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "store_memory"
        return END

    builder.add_conditional_edges(
        "call_model",
        route_after_model,
        {
            "store_memory": "store_memory",
            END: END,
        },
    )
    builder.add_edge("store_memory", "call_model")

    # compile graph and return
    return builder.compile(checkpointer=checkpointer)


def send_message(
    user_input, graph, user_id="default_user", thread_id=None, debug=False
):
    """
    Processes a user input message, interacts with a given graph object, and returns the
    final response message. The function assigns default or optional thread identifiers
    and ensures proper configuration for invoking the graph object.

    :param user_input: The message content provided by the user, typically a string.
    :param graph: An object that allows invocation for interaction based on the provided
        user input and configuration.
    :param user_id: An optional identifier for the user. Defaults to "default_user".
    :param thread_id: An optional identifier for the thread. If not provided, a default
        thread ID is generated based on the user ID.
    :param debug: Whether to enable LangGraph debugging output.
    :return: The final response message from the interaction with the graph object.
    """
    thread_id = thread_id or f"thread_{user_id}"
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}

    print(f"{33*'='} User Message {33*'='}\n{user_input}\n")
    messages = [HumanMessage(content=user_input)]
    response = graph.invoke(
        {"messages": messages, "user_id": user_id},
        config=config,
        debug=debug,
    )

    final_message = response["messages"][-1]
    final_message.pretty_print()
    return final_message
