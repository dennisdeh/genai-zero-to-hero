"""
Demonstrate a multi-agent setup with a supervisor using LangGraph.

We define two agents:
- a math agent
- a RAG Agent (vector search against a local VectorDB server)
"""

import os
from typing import TypedDict, Annotated, Sequence
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langgraph.types import Command
from langchain_core.tools import tool
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import dotenv
from p07_llms.c01_running_llms.s02_langchain.utils.helpers import get_llm
from p07_llms.c04_rag_systems.s01_finma_rag_system.utils.document_loaders import (
    load_documents_from_folder,
)

# ----------------------------
# Config
# ----------------------------
path_env = os.path.join("p08_agents/c00_langchain_langgraph", ".env")
dotenv.load_dotenv(path_env)
# URLs
OLLAMA_BASE_URL = f"http://localhost:{os.getenv('OLLAMA_PORT_HOST')}"
SQLALCHEMY_URL = f"mysql+pymysql://{os.getenv('MARIADB_USER')}:{os.getenv('MARIADB_PASSWORD')}@localhost:{os.getenv('MARIADB_PORT_HOST')}"
QDRANT_URL = f"http://localhost:{os.getenv('QDRANT_PORT_HOST')}"
# VectorStore settings
k = 4  # For selection of relevant documents
QDRANT_COLLECTION = "finma_docs"
DOCUMENTS_DIR = os.path.join(
    "p07_llms/c04_rag_systems/s01_finma_rag_system", "documents"
)


# 1. Set up Ollama LLM and embedding objects
llm = get_llm(model="qwen3:8b", use="ollama", base_url_ollama=OLLAMA_BASE_URL)
embedding = OllamaEmbeddings(
    model="qwen3-embedding",
    base_url=OLLAMA_BASE_URL,
)
qdrant_client = QdrantClient(
    url="http://localhost",
    port=os.getenv("QDRANT_PORT_HOST"),
)

# 2: check if the collection exists, if not, create it
qdrant = None
if not qdrant_client.collection_exists(QDRANT_COLLECTION):
    # 2A.1. Load documents from ./documents and split into chunks
    raw_docs = load_documents_from_folder(DOCUMENTS_DIR)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )
    docs = splitter.split_documents(raw_docs)
    # 2A.2. Build or use a Qdrant collection from these documents
    qdrant = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embedding,
        url="http://localhost",
        port=os.getenv("QDRANT_PORT_HOST"),
        prefer_grpc=False,
        collection_name=QDRANT_COLLECTION,
    )
else:
    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embedding,
        url="http://localhost",
        port=os.getenv("QDRANT_PORT_HOST"),
        prefer_grpc=False,
        collection_name=QDRANT_COLLECTION,
    )


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


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


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """
    Retrieve relevant context from the Qdrant collection.

    :param query: Query string for context retrieval.
    :return: Serialized context and retrieved documents.
    """
    retrieved_docs = qdrant.similarity_search(query, k=k)
    serialised = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}" for doc in retrieved_docs
    )
    return serialised, retrieved_docs


# Define sub-agents
agent_math = create_agent(
    model=llm,
    tools=[tool_sum, tool_multiplication],
    system_prompt=(
        "You are a math agent."
        "You can only answer questions about math."
        "You can use any of the following tools: "
        "sum (to add all numbers in a list), multiplication (to multiply all numbers in a list)."
    ),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"tool_multiplication": True},
            description_prefix="Validate the inputs",
        ),
    ],
)
agent_rag = create_agent(
    model=llm,
    tools=[retrieve_context],
    system_prompt=(
        "You are an agent that answers questions about financial markets regulations "
        "from FINMA (Switzerland’s independent financial-markets regulator). "
        "Answer using ONLY the provided context (ALL are FINMA documents). "
        "ALWAYS quote the source of the information. "
        "If the answer is not in the context, say you don't know."
    ),
)


# wrap agents as tools (best practice)
@tool
def math(request: str) -> str:
    """
    Get answers to questions about math.

    Use this to ask questions about math.

    Input: natual language question.
    :param request:
    :return:
    """
    result = agent_math.invoke({"messages": [HumanMessage(content=request)]})
    return result["messages"][-1].content


@tool
def finma_rag(request: str) -> str:
    """
    Get the answer to the query from the RAG agent.

    Use this to ask questions about FINMA documents.

    Input: natual language question.
    """
    result = agent_rag.invoke({"messages": [HumanMessage(content=request)]})
    return result["messages"][-1].content


# Define supervisor agent
agent_supervisor = create_agent(
    model=llm,
    tools=[math, finma_rag],
    system_prompt=(
        "You are a helpful agent that answers questions about financial "
        "markets regulations and solves math problems. "
    ),
    checkpointer=InMemorySaver(),
)


if __name__ == "__main__2":
    # set global variables
    os.environ["LLM_TO_USE"] = "ollama"
    print(
        "Multi-Agent app with a supervisor agent and Human-in-the-Loop is initialised!"
    )
    print(f"Using LLM backend: {os.getenv('LLM_TO_USE')}")

    # trace the output of the graph
    str_query = "Tell me how many times mentions climate risks in their recent circular, multiply the numbers [2,3,4,5]"
    config = {"configurable": {"thread_id": "6"}}

    interrupts = []
    for step in agent_supervisor.stream(
        {"messages": [HumanMessage(content=str_query)]},
        config,
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
    for interrupt_ in interrupts:
        if interrupt_.id == "3da87be07718d50beff9123a8b485b97":
            # Edit multiplication
            edited_action = interrupt_.value["action_requests"][0].copy()
            edited_action["args"]["values"] = "multiply the numbers [1,2,3]"
            resume[interrupt_.id] = {
                "decisions": [{"type": "edit", "edited_action": edited_action}]
            }
        else:
            resume[interrupt_.id] = {"decisions": [{"type": "approve"}]}

    interrupts = []
    for step in agent_supervisor.stream(
        Command(resume=resume),
        config,
    ):
        for update in step.values():
            if isinstance(update, dict):
                for message in update.get("messages", []):
                    message.pretty_print()
            else:
                interrupt_ = update[0]
                interrupts.append(interrupt_)
                print(f"\nINTERRUPTED: {interrupt_.id}")
