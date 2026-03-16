"""
Demonstrate a simple RAG agent that implements adapts the simple RAG over local files (PDF + Word .docx)
in p07_llms/c04_rag_systems/s01_finma_rag_system using the following components:
- Ollama (LLM + embeddings)
- Qdrant (vector DB)
- LangGraph Agent

The documents folder should contain PDFs and Word .docx files. In the example, circulars
from FINMA (https://www.finma.ch/en/documents/) are used.
"""

import os
from typing import TypedDict, Annotated, Sequence
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
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
def get_stock_analysis(ticker: str) -> str:
    """
    Perform financial analysis on a given stock ticker.

    Args:
        ticker: The stock ticker symbol (e.g. 'AAPL', 'GOOGL')

    Returns:
        A string containing financial analysis data
    """
    # mock financial data - in production, this would call a real API
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


def chatbot(state: AgentState) -> AgentState:
    """
    Process messages and interact with the LLM.
    """
    tools = [get_stock_analysis, retrieve_context]
    llm_with_tools = llm.bind_tools(tools)
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def compile_graph():
    # initialise building the language graph
    graph_builder = StateGraph(AgentState)

    # add nodes
    graph_builder.add_node("chatbot", chatbot)
    tool_node = ToolNode(tools=[get_stock_analysis, retrieve_context])
    graph_builder.add_node("tools", tool_node)

    # add edges
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges(
        "chatbot",
        lambda state: "tools" if state["messages"][-1].tool_calls else END,
        {"chatbot": "chatbot", "tools": "tools", END: END},
    )
    graph_builder.add_edge("tools", "chatbot")

    # compile the graph
    return graph_builder.compile()


if __name__ == "__main__":
    # set global variables
    os.environ["LLM_TO_USE"] = "ollama"
    print("RAG Agent initialised!")
    print(f"Using LLM backend: {os.getenv('LLM_TO_USE')}")

    # instantiate the language graph
    graph = compile_graph()

    # visualise
    with open(
        "p08_agents/c00_langchain_langgraph/s02_rag_agent/graph_rag_agent.png", "wb"
    ) as f:
        f.write(graph.get_graph().draw_mermaid_png())
    # trace the output of the graph
    str_query = (
        "Can you analyse the Apple stock (AAPL) for me and tell me what "
        "FINMA says about risks associated with stock trading?"
    )
    messages = [
        SystemMessage(
            content="You are a helpful assistant that answers questions about financial markets regulations "
            "from FINMA (Switzerland’s independent financial-markets regulator). "
            "Answer using ONLY the provided context (ALL are FINMA documents). "
            "ALWAYS quote the source of the information. "
            "If the answer is not in the context, say you don't know."
        ),
        HumanMessage(content=str_query),
    ]
    for event in graph.stream(
        {"messages": messages},
        stream_mode="values",
    ):
        event["messages"][-1].pretty_print()
