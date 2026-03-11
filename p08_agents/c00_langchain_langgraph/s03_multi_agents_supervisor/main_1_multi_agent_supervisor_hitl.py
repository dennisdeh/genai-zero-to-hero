"""
Demonstrate a multi-agent setup with a supervisor and a human in the loop using
the HumanInTheLoopMiddleware in LangGraph.

We define two agents:
- a math agent
    - The use of tool_multiplication needs to be approved by the human in the loop
    - Memory checkpointing is used to save the state of the math agent
- a RAG Agent (vector search against a local VectorDB server)

The workflow for the interrupt is:
    - user asks the original query
    - query_rewrite is interrupted
    - you edit it or approve it
    - the graph resumes, tied to the exact interrupt IDs
    - nested agent/tool execution restarts from the stored state
"""

import os
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
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
k = 5  # For selection of relevant documents
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


def retrieve_context_data(query: str) -> tuple[str, list]:
    """
    Plain Python helper for retrieval.

    This separates retrieval business logic from the LangChain tool wrapper so the
    function can be safely reused from normal Python code and from agents/tools.
    """
    retrieved_docs = qdrant.similarity_search(query, k=k)
    serialised = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}" for doc in retrieved_docs
    )
    return serialised, retrieved_docs


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """
    Retrieve relevant context from the Qdrant collection.

    :param query: Query string for context retrieval.
    :return: Serialized context and retrieved documents.
    """
    return retrieve_context_data(query)


agent_math = create_agent(
    model=llm,
    tools=[tool_sum, tool_multiplication],
    system_prompt=(
        "You are a math agent."
        "You can only answer questions about math."
        "You can use any of the following tools: "
        "sum (to add all numbers in a list), "
        "multiplication (to multiply all numbers in a list)."
    ),
)


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


agent_query_rewrite = create_agent(
    model=llm,
    tools=None,
    system_prompt=(
        "You improve the quality of a query for retrieval. "
        "Return only the rewritten query as plain text. "
        "Do not explain your reasoning. "
        "Do not answer the question."
    ),
)


@tool
def query_rewrite(request: str) -> str:
    """
    Rewrites the given query string to improve the quality of the query.

    Use this tool to rewrite the query before sending it to the RAG agent.

    :param request: The input query string to be rewritten.
    :type request: str
    :return: The rewritten query string.
    :rtype: str
    """
    result = agent_query_rewrite.invoke({"messages": [HumanMessage(content=request)]})
    return result["messages"][-1].content


agent_rag_rewriter = create_agent(
    model=llm,
    tools=[query_rewrite],
    system_prompt=(
        "You improve user queries before retrieval. "
        "You MUST call the query_rewrite tool exactly once for the ORIGINAL user query. "
        "After the tool returns, do not call any tool again. "
        "Return the rewritten query and do not answer the user question yourself."
    ),
    checkpointer=InMemorySaver(),  # lets the interrupted agent_rag_rewriter resume from its own saved state
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"query_rewrite": {"allowed_decisions": ["approve", "edit"]}},
        ),
    ],
)


agent_rag_answer = create_agent(
    model=llm,
    tools=[retrieve_context],
    system_prompt=(
        "You are an agent that answers questions about financial markets regulations "
        "from FINMA (Switzerland’s independent financial-markets regulator). "
        "Use retrieve_context to fetch relevant FINMA documents for the query you receive. "
        "Answer using ONLY the provided context (ALL are FINMA documents). "
        "ALWAYS quote the source of the information. "
        "If the answer is not in the context, say you don't know."
    ),
    # checkpointer=InMemorySaver(),
)


@tool
def finma_rag(request: str) -> str:
    """
    Get the answer to the query from the RAG agent.

    Use this to ask questions about FINMA documents.

    Input: natual language question.
    """
    # rewrite the query using the rag rewriter
    rewritten = agent_rag_rewriter.invoke({"messages": [HumanMessage(content=request)]})

    # extract the latest rewritten query from the messages
    rewritten_query = None
    for message in reversed(rewritten["messages"]):
        if isinstance(message, ToolMessage) and message.name == "query_rewrite":
            rewritten_query = message.content
            break

    if rewritten_query is None:
        raise ValueError(
            "query_rewrite did not return a rewritten query when it was expected."
        )

    # get context for the rewritten query
    retrieved_context, _ = retrieve_context_data(rewritten_query)

    # answer the rewritten query using the retrieved context
    answer = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are an agent that answers questions about financial markets regulations "
                    "from FINMA (Switzerland’s independent financial-markets regulator). "
                    "Answer using ONLY the provided context (ALL are FINMA documents). "
                    "ALWAYS quote the source of the information. "
                    "If the answer is not in the context, say you don't know."
                )
            ),
            HumanMessage(
                content=(
                    f"User request: {rewritten_query}\n\n"
                    f"Retrieved context:\n{retrieved_context}"
                )
            ),
        ]
    )
    return answer.content


# Define supervisor agent
agent_supervisor = create_agent(
    model=llm,
    tools=[math, finma_rag],
    system_prompt=(
        "You are a helpful agent that answers questions about financial "
        "markets regulations and solves math problems. "
        "Math problems are handed to the math tool."
        "Questions about FINMA documents are handed to the finma_rag tool."
        "Only call the finma_rag tool once"
        "Format the final output in markdown docstrings."
    ),
    checkpointer=InMemorySaver(),
)


def handle_interrupt(interrupts: list):
    """
    Handle all interrupts and returns a resume dictionary
    where the keys are the interrupt ids and the values are the decisions.

    :param interrupts:
    :return: resume dictionary
    """
    resume = {}
    for interrupt_ in interrupts:
        print(interrupt_.value["action_requests"][0]["description"])
        approval = input("Approve or edit? (y/edit): ")

        if approval == "y":
            resume[interrupt_.id] = {
                "decisions": [{"type": "approve", "message": "Approved by user."}]
            }
        elif approval == "edit":
            str_new_query = input("Enter revised query: ")

            action_request = interrupt_.value["action_requests"][0]
            edited_action = action_request.copy()
            edited_action["args"] = action_request["args"].copy()
            edited_action["args"]["request"] = str_new_query

            resume[interrupt_.id] = {
                "decisions": [
                    {
                        "type": "edit",
                        "edited_action": edited_action,
                    }
                ]
            }
        else:
            raise ValueError("Invalid input. Please enter 'y' or 'edit'.")
    return resume


if __name__ == "__main__":
    # set global variables
    os.environ["LLM_TO_USE"] = "ollama"
    print("Multi-Agent app with a supervisor agent and HITL is initialised!")
    print(f"Using LLM backend: {os.getenv('LLM_TO_USE')}")
    debug = False

    # draw the graphs
    with open(
        "p08_agents/c00_langchain_langgraph/s03_multi_agents_supervisor/graph_1_multi_agent_supervisor_rag_answer.png",
        "wb",
    ) as f:
        f.write(agent_rag_answer.get_graph().draw_mermaid_png())
    with open(
        "p08_agents/c00_langchain_langgraph/s03_multi_agents_supervisor/graph_1_multi_agent_supervisor_math.png",
        "wb",
    ) as f:
        f.write(agent_math.get_graph().draw_mermaid_png())
    with open(
        "p08_agents/c00_langchain_langgraph/s03_multi_agents_supervisor/graph_1_multi_agent_supervisor_supervisor.png",
        "wb",
    ) as f:
        f.write(agent_supervisor.get_graph().draw_mermaid_png())

    # trace the output of the graph
    str_query = "Tell me how many times FINMA mentions financial risks in their recent circular, multiply the numbers [2,3,4,5]."
    config = {"configurable": {"thread_id": "1"}}

    interrupts = []
    for step in agent_supervisor.stream(
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

    while len(interrupts) > 0:
        resume = handle_interrupt(interrupts)

        interrupts = []
        for step in agent_supervisor.stream(
            Command(resume=resume), config=config, debug=debug
        ):
            for update in step.values():
                if isinstance(update, dict):
                    for message in update.get("messages", []):
                        message.pretty_print()
                elif update is not None:
                    interrupt_ = update[0]
                    interrupts.append(interrupt_)
                    print(f"\nINTERRUPTED: {interrupt_.id}")
                else:
                    print("No update")
                    break
