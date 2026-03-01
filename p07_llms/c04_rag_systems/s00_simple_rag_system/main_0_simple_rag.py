"""
This is a simple RAG system using Qdrant and Ollama. The "documents" (simple text
pieces about Langchain) are loaded into Qdrant, and the RAG chain is run against them.
"""

from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from p07_llms.c01_running_llms.s02_langchain.utils.helpers import get_llm
import dotenv
import os

# ----------------------------
# Config
# ----------------------------
# https://docs.langchain.com/oss/python/langchain/rag#rag-chains
path_env = os.path.join("p07_llms/c04_rag_systems", ".env")
dotenv.load_dotenv(path_env)
# URLs
OLLAMA_BASE_URL = f"http://localhost:{os.getenv('OLLAMA_PORT_HOST')}"
QDRANT_URL = f"http://localhost:{os.getenv('QDRANT_PORT_HOST')}"
# Other parameters
QDRANT_COLLECTION = "docs"

# 1. Set up Ollama LLM and embedding objects
llm = get_llm(model="qwen3:8b", use="ollama", base_url_ollama=OLLAMA_BASE_URL)
embedding = OllamaEmbeddings(
    model="qwen3-embedding",
    base_url=OLLAMA_BASE_URL,
)

# 2. Load and split documents (some text examples from LangChain's website)
docs = [
    Document(
        page_content="LangChain is a composable framework to build with LLMs. LangGraph is the orchestration framework for controllable agentic workflows."
    ),
    Document(page_content="The largest community building the future of LLM apps"),
    Document(
        page_content="LangChain’s flexible abstractions and AI-first toolkit make it the #1 choice for developers when building with GenAI. Join 1M+ builders standardizing their LLM app development in LangChain's Python and JavaScript frameworks."
    ),
]

# 3. Connect to Qdrant and upload documents
qdrant = QdrantVectorStore.from_documents(
    documents=docs,
    embedding=embedding,
    url=QDRANT_URL,
    prefer_grpc=False,
    collection_name=QDRANT_COLLECTION,
)

# 4. Create retriever and a simple RAG chain (Retriever -> Prompt+LLM)
retriever = qdrant.as_retriever(search_kwargs={"k": 3})
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer using ONLY the provided context. "
            "If the answer is not in the context, say you don't know.",
        ),
        ("human", "Question: {input}\n\nContext:\n{context}"),
    ]
)
document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
rag_chain = create_retrieval_chain(
    retriever=retriever, combine_docs_chain=document_chain
)

# 5. Run a query asking about the documents
query = "What is LangChain? Give a short summary."
result = rag_chain.invoke({"input": query})

print("\nAnswer:", result["answer"])
print("\nSources:")
for doc in result["context"]:
    print("-", doc.page_content)
