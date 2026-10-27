## Environment
- Python conda environment to use: `conda_env_gzh_llm4.yml`
- Docker compose setup: `docker-compose.yml` (needed for all sections)
- Environment variables: set in the `.env` file

The `.env` file in this folder defines the port mapping used by `docker-compose.yml` and read by the scripts:
`OLLAMA_PORT_HOST`, `OLLAMA_PORT_CONT` (Ollama listens on 11434 inside the container), `QDRANT_PORT_HOST` and
`QDRANT_PORT_CONT` (Qdrant's REST API listens on 6333 inside the container). The scripts connect to
`http://localhost:<OLLAMA_PORT_HOST>` and `http://localhost:<QDRANT_PORT_HOST>`.

## How to run
Start Ollama and Qdrant and pull the chat and embedding models (once). Run the commands from this chapter folder
so that Docker Compose picks up the `.env` file:

```bash
cd p07_llms/c04_rag_systems
docker compose up -d
docker exec -it gzh-ollama_rag ollama pull qwen3:8b
docker exec -it gzh-ollama_rag ollama pull qwen3-embedding
cd ../..
```

Without an NVIDIA GPU (and the NVIDIA Container Toolkit), remove the `deploy:` block of the `ollama` service.

For `s01_finma_rag_system`, create the folder `p07_llms/c04_rag_systems/s01_finma_rag_system/documents/` (it is
gitignored) and put the PDF and Word (`.docx`) documents to be queried there, e.g. circulars and the Risk Monitor
from https://www.finma.ch/en/documents/. The documents are indexed into the Qdrant collection `finma_docs` on the
first run only; delete the collection (Qdrant dashboard at `http://localhost:<QDRANT_PORT_HOST>/dashboard`) to
re-index after changing the documents. The same collection is used by the agents in `p08_agents`.

Run the examples from the repository root (the scripts load `p07_llms/c04_rag_systems/.env` relative to it):

```bash
python -m p07_llms.c04_rag_systems.s00_simple_rag_system.main_simple_rag
python -m p07_llms.c04_rag_systems.s01_finma_rag_system.main_rag_local_files
```

| Script | What it shows |
|---|---|
| `s00_simple_rag_system/main_simple_rag.py` | minimal RAG: 3 in-memory documents -> Qdrant collection `docs` -> retriever (k=3) -> stuff-documents chain with a grounded prompt |
| `s01_finma_rag_system/main_rag_local_files.py` | RAG over local PDF/Word files: loading with page metadata, recursive chunking (1000/150 characters), index-once logic, similarity scores, score threshold, answers with source file and page |
