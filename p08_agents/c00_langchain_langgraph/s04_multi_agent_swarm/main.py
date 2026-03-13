"""
This is a multi-agent swarm example using LangChain and LangGraph.
There is not a supervisor agent, instead there is a swarm of two agents that work
together to answer queries by handing off the work to the other agent when relevant.

We define two agents:
- a math agent
- a RAG Agent (vector search against a local VectorDB server)
"""

import os
from langgraph.checkpoint.memory import InMemorySaver
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langgraph_swarm import create_handoff_tool, create_swarm
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


# ----------------------------
# Agents and tools
# ----------------------------


@tool
def tool_sum(values: list[float]) -> float:
    """
    Sums all the numbers in the list 'values' and returns the result.
    """
    vals = [float(val) for val in values]
    result = sum(vals)
    return result


@tool
def tool_multiplication(values: list[float]) -> float:
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


# Handoff tools
transfer_to_rag_agent = create_handoff_tool(
    agent_name="agent_rag",
    description="Transfer user to the rag agent that can retrieve information from FINMA documents.",
)
transfer_to_math_agent = create_handoff_tool(
    agent_name="agent_math",
    description="Transfer user to the math agent that can answer math questions.",
)


# Define sub-agents
agent_math = create_agent(
    name="agent_math",
    model=llm,
    tools=[tool_sum, tool_multiplication, transfer_to_rag_agent],
    system_prompt=(
        "You are a math agent."
        "You can only answer questions about math."
        "You can use any of the following tools: "
        "sum (to add all numbers in a list), multiplication (to multiply all numbers in a list). "
        "If the user asks a question about FINMA financial markets regulations, hand off to agent_rag."
    ),
)
agent_rag = create_agent(
    name="agent_rag",
    model=llm,
    tools=[retrieve_context, transfer_to_math_agent],
    system_prompt=(
        "You are an agent that answers questions about financial markets regulations "
        "from FINMA (Switzerland’s independent financial-markets regulator). "
        "For every FINMA or regulatory question, call retrieve_context exactly once. "
        "Do not answer from memory. "
        "If the user asks a math question, hand off to agent_math."
    ),
)


if __name__ == "__main__":
    # set global variables
    os.environ["LLM_TO_USE"] = "ollama"
    print("Multi-Agent app with a supervisor agent is initialised!")
    print(f"Using LLM backend: {os.getenv('LLM_TO_USE')}")
    debug = True

    # trace the output of the graph
    str_query = "Tell me how many times FINMA mentions climate risks in their recent circular, add the numbers [2,3,4,5]"
    messages = [
        HumanMessage(content=str_query),
    ]

    # Compile and run the swarm
    checkpointer = InMemorySaver()
    builder = create_swarm([agent_math, agent_rag], default_active_agent="agent_math")
    config = {"configurable": {"thread_id": "1"}}

    # Important: compile the swarm with a checkpointer to remember
    # previous interactions and last active agent
    app = builder.compile(checkpointer=checkpointer)

    # invoke the swarm
    for event in app.stream(
        {"messages": messages}, config=config, stream_mode="values", debug=debug
    ):
        event["messages"][-1].pretty_print()
