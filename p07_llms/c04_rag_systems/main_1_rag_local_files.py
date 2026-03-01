"""
Simple RAG over local files (PDF + Word .docx) using:
- Ollama (LLM + embeddings)
- Qdrant (vector DB)
- LangChain retrieval chain

The documents folder should contain PDFs and Word .docx files. In the example, circulars
from FINMA (https://www.finma.ch/en/documents/) are used.
"""

from pathlib import Path
import os
import dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from p07_llms.c01_running_llms.s02_langchain.utils.helpers import get_llm


# ----------------------------
# Config
# ----------------------------
path_env = os.path.join("p07_llms/c04_rag_systems", ".env")
dotenv.load_dotenv(path_env)

OLLAMA_BASE_URL = f"http://localhost:{os.getenv('OLLAMA_PORT_HOST')}"
QDRANT_COLLECTION = "docs"
DOCUMENTS_DIR = os.path.join("p07_llms/c04_rag_systems", "documents")


def load_documents_from_folder(folder: str):
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Documents folder not found: {folder}. Create it and add .pdf/.docx files."
        )
    pdf_loader = DirectoryLoader(
        str(folder),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
    )
    docx_loader = DirectoryLoader(
        str(folder),
        glob="**/*.docx",
        loader_cls=Docx2txtLoader,
        show_progress=True,
        use_multithreading=True,
    )

    docs = []
    docs.extend(pdf_loader.load())
    docs.extend(docx_loader.load())

    if not docs:
        raise ValueError(f"No .pdf or .docx files found under: {folder}")

    return docs


if __name__ == "__main__":
    # 1. Set up Ollama LLM and embedding objects
    llm = get_llm(model="qwen3:8b", use="ollama", base_url_ollama=OLLAMA_BASE_URL)
    embedding = OllamaEmbeddings(
        model="qwen3-embedding",
        base_url=OLLAMA_BASE_URL,
    )

    # 2. Load documents from ./documents and split into chunks
    raw_docs = load_documents_from_folder(DOCUMENTS_DIR)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = splitter.split_documents(raw_docs)

    # 3. Build / overwrite a Qdrant collection from these documents
    # NOTE: from_documents will (re)index the passed docs into the collection.
    qdrant = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embedding,
        url="http://localhost",
        port=os.getenv("QDRANT_PORT_HOST"),
        prefer_grpc=False,
        collection_name=QDRANT_COLLECTION,
    )

    # 4. Create retriever and a simple RAG chain
    retriever = qdrant.as_retriever(search_kwargs={"k": 4})

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

    # 5. Ask a question
    query = "What do the circulars say about cryptocurrencies and their treatment?"
    result = rag_chain.invoke({"input": query})

    print("\nAnswer:\n", result["answer"])
    print("\nSources (top matches):")
    for d in result["context"]:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        page_txt = f" (page {page})" if page is not None else ""
        print(f"- {src}{page_txt}")
