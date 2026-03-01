"""
Simple RAG over local files (PDF + Word .docx) using:
- Ollama (LLM + embeddings)
- Qdrant (vector DB)
- LangChain retrieval chain

The documents folder should contain PDFs and Word .docx files. In the example, circulars
from FINMA (https://www.finma.ch/en/documents/) are used.
"""

import os
import dotenv
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from p07_llms.c01_running_llms.s02_langchain.utils.helpers import get_llm
from p07_llms.c04_rag_systems.s01_finma_rag_system.utils.document_loaders import (
    load_documents_from_folder,
)

# ----------------------------
# Config
# ----------------------------
path_env = os.path.join("p07_llms/c04_rag_systems", ".env")
dotenv.load_dotenv(path_env)

OLLAMA_BASE_URL = f"http://localhost:{os.getenv('OLLAMA_PORT_HOST')}"
QDRANT_COLLECTION = "finma_docs"
DOCUMENTS_DIR = os.path.join(
    "p07_llms/c04_rag_systems/s01_finma_rag_system", "documents"
)


if __name__ == "__main__":
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

    # 4. Create the retriever object and a simple RAG chain
    retriever = qdrant.as_retriever(search_kwargs={"k": 5})
    # define prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that answers questions about financial markets regulations "
                "from FINMA (Switzerland’s independent financial-markets regulator). "
                "Answer using ONLY the provided context (ALL are FINMA documents). "
                "If the answer is not in the context, say you don't know.",
            ),
            ("human", "Question: {input}\n\nContext:\n{context}"),
        ]
    )
    # create a simple RAG chain which passes a list of documents to the LLM after the relevant context is extracted
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    rag_chain = create_retrieval_chain(
        retriever=retriever, combine_docs_chain=document_chain
    )

    # 5. Ask a question, which will be answered by the RAG chain
    query = "What does FINMA say about credit risk developments?"
    result = rag_chain.invoke({"input": query})

    print("\nAnswer:\n", result["answer"])
    print("\nSources (top matches):")
    for d in result["context"]:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        page_txt = f" (page {page})" if page is not None else ""
        print(f"- {src}{page_txt}")
