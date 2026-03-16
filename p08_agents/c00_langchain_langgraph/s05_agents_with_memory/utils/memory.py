"""
Portions of this file are derived from:
https://medium.com/@anil.jain.baba/long-term-agentic-memory-with-langgraph-824050b09852
Original author: Anil Jain Baba
Changes: Adaption to use Ollama, compatability with the newest version of LangGraph
and the existing structure, new implementation of the upsert_memory tool, update of the
functions to store and retrieve memories.
"""

from typing import Optional
from langgraph.store.memory import BaseStore
from langchain_core.messages import ToolMessage, SystemMessage
from langchain_core.tools import tool
import uuid
from p08_agents.c00_langchain_langgraph.s05_agents_with_memory.utils.agent_state import (
    AgentState,
)


@tool
def upsert_memory(
    content: str,
    context: str,
    memory_id: Optional[str] = None,
):
    """
    Inserts or updates memory content within a specified context. If a memory ID is
    provided, the function updates the corresponding memory entry; otherwise, it
    creates a new memory entry with the given content and context.

    :param content: The content to be stored or updated in memory.
    :type content: str
    :param context: The contextual information related to the content.
    :type context: str
    :param memory_id: The unique identifier of the memory to be updated. If not
        provided, a new memory entry will be created.
    :type memory_id: Optional[str]
    :return: A dictionary containing the memory ID, context, and content after
        insertion or update.
    :rtype: dict
    """
    return {
        "memory_id": memory_id,
        "context": context,
        "content": content,
    }


def _find_existing_memory_id(
    store: BaseStore,
    namespace: tuple[str, str],
    content: str,
    context: str,
) -> Optional[str]:
    """
    Searches for an existing memory ID from the given store that matches the provided
    namespace, content, and context. If a match is found, the key of the matched memory
    is returned.

    :param store: The store object used to search for existing memories.
    :type store: BaseStore
    :param namespace: The namespace tuple identifying the group of stored memories.
    :type namespace: tuple[str, str]
    :param content: The specific content to search for within the existing memories.
    :type content: str
    :param context: The context paired with the content to search for in the memories.
    :type context: str
    :return: The key of the matching memory if found, otherwise None.
    :rtype: Optional[str]
    """
    existing_memories = store.search(namespace)
    for memory in existing_memories:
        value = memory.value or {}
        if value.get("context") == context and value.get("content") == content:
            return memory.key
    return None


def store_memory(state: AgentState, *, store: BaseStore):
    """
    Stores memory entries based on the provided tool calls in the last message of the given state.
    Processes `upsert_memory` tool calls, identifies or generates memory IDs, and stores the
    defined contexts and content in the specified store. Updates the state with any results
    produced by the storage operations.

    :param state: The current message state object containing user messages and metadata.
    :type state: AgentState

    :param store: A storage system that implements the required methods to interact with
                  memory entries.
    :type store: BaseStore

    :return: The updated state after processing and storing memory-related tool calls.
    :rtype: AgentState
    """
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", []) or []
    results = []
    user_id = state["user_id"]
    namespace = ("memories", user_id)

    for tc in tool_calls:
        if tc["name"] == "upsert_memory":
            context = tc["args"]["context"]
            content = tc["args"]["content"]
            memory_id = tc["args"].get("memory_id")
            # check if memory already exists, in this case it should be updated
            if memory_id is None:
                memory_id = _find_existing_memory_id(
                    store=store,
                    namespace=namespace,
                    content=content,
                    context=context,
                ) or str(uuid.uuid4())
            # put the memory in the store
            store.put(
                namespace,
                key=memory_id,
                value={"context": context, "content": content},
            )
            results.append(f"Stored memory: {context} - {content}")

    updated_state = dict(state)
    updated_messages = list(state["messages"])

    if tool_calls and results:
        updated_messages.append(
            ToolMessage(content="\n".join(results), tool_call_id=tool_calls[0]["id"])
        )

    updated_state["messages"] = updated_messages
    return updated_state


def retrieve_memories(state: AgentState, *, store: BaseStore):
    """
    Retrieve and integrate relevant memories from the store based on the user's last message.

    This function searches for relevant memories in the provided `store` using the user's
    conversation history and integrates these memories into the current state. If relevant
    memories are found, the function updates the state with a memory context containing the
    retrieved memory details.

    :param state: The current message state containing conversation details, including
        the user ID and message history.
    :type state: AgentState
    :param store: The storage backend used to search for relevant memories.
    :type store: BaseStore
    :return: The updated message state containing the integrated memory context if any
        relevant memories are found.
    :rtype: dict
    """
    user_id = state["user_id"]
    namespace = ("memories", user_id)
    query = state["messages"][-1].content

    memories = store.search(namespace, query=query)
    memory_text = "\n".join(
        [f"{mem.value['context']} - {mem.value['content']}" for mem in memories]
    )

    updated_state = dict(state)
    if memory_text:
        updated_state["memory_context"] = [
            SystemMessage(content=f"Relevant memories:\n{memory_text}")
        ]

    return updated_state
