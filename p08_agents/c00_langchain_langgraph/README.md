## Environment
- Python conda environment to use: `conda_env_gzh_agents0.yml`
- Docker compose setup:
  - s00: none (no LLM is used)
  - s01 (`main_1` to `main_5`): an Ollama server reachable via `get_llm()`, i.e. by default the one from
    `p07_llms/c01_running_llms/s01_ollama/docker-compose.yml` on `http://localhost:07011` (or set
    `OLLAMA_BASE_URL` to the Ollama service of this chapter's compose file); these scripts pass no model name,
    so they use `OLLAMA_MODEL` if set, otherwise the first installed model (use a tool-calling model such as
    `qwen3` for `main_3` to `main_5`)
  - s01 (`main_6`), s02 - s05: `p08_agents/c00_langchain_langgraph/docker-compose.yml`
- Environment variables: set in the `.env` file in this folder, which is read both by Docker Compose and by the
  scripts (`dotenv.load_dotenv("p08_agents/c00_langchain_langgraph/.env")`, relative to the repository root):
  - `OLLAMA_PORT_HOST`, `OLLAMA_PORT_CONT`: port mapping of the Ollama service (11434 inside the container)
  - `QDRANT_PORT_HOST`, `QDRANT_PORT_CONT`: port mapping of the Qdrant service (6333 inside the container)
  - `MARIADB_ROOT_PASSWORD`, `MARIADB_USER`, `MARIADB_PASSWORD`, `MARIADB_PORT_HOST`, `MARIADB_PORT_CONT`,
    `ADMINER_PORT_HOST`, `ADMINER_PORT_CONT`: MariaDB and its web UI Adminer
  - optional for `get_llm()`: `LLM_TO_USE`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OPENAI_API_KEY` (note that the
    scripts set `LLM_TO_USE=ollama` themselves)

### Services in `docker-compose.yml` and which sections need them

| Service | Container | Used by |
|---|---|---|
| `ollama` (LLM + embeddings) | `gzh-ollama_agents_rag` | s01 `main_6`, s02, s03, s04, s05 (models `qwen3:8b` and `qwen3-embedding`) |
| `qdrant` (vector database) | `gzh-qdrant_agents_rag` | s02, s03, s04 (collection `finma_docs`) |
| `mariadb` + `adminer` | `gzh-mariadb_agents_rag`, `gzh-adminer_agents_rag` | not used by the current scripts (s02 - s04 only define a `SQLALCHEMY_URL`); prepared for SQL tools |

Notes:
- The `mariadb` service bind-mounts `./init` to `/docker-entrypoint-initdb.d` for SQL initialisation scripts.
  This folder is not part of the repository; Docker creates it (empty) on the first start, which is fine because
  no initial tables are required. Put `*.sql` files there if you want the database to be initialised with data
  (they are only executed when the data directory is empty, i.e. on the very first start).
- Adminer is preconfigured with `ADMINER_DEFAULT_SERVER: mariadb` (the compose service name of the database), so the
  server field of the Adminer login form is already filled in.
- Without an NVIDIA GPU (and the NVIDIA Container Toolkit), remove the `deploy:` block of the `ollama` service.
- s02 - s04 index the FINMA documents from `p07_llms/c04_rag_systems/s01_finma_rag_system/documents/` (gitignored;
  put the PDF/Word files there, see the README of `p07_llms/c04_rag_systems`) into Qdrant on the first run, if the
  collection `finma_docs` does not exist yet.

## How to run
Start the services from this folder (so that Docker Compose reads the `.env` file) and pull the models once:

```bash
cd p08_agents/c00_langchain_langgraph
docker compose up -d                 # or: docker compose up -d ollama qdrant
docker exec -it gzh-ollama_agents_rag ollama pull qwen3:8b
docker exec -it gzh-ollama_agents_rag ollama pull qwen3-embedding
cd ../..
```

Run the scripts from the repository root:

```bash
# s00: graphs without an LLM (run anywhere, no services needed)
python -m p08_agents.c00_langchain_langgraph.s00_very_simple_graphs.main_very_simple_langgraph
python -m p08_agents.c00_langchain_langgraph.s00_very_simple_graphs.main_very_simple_langgraph_looping
# s01: simple agents
python -m p08_agents.c00_langchain_langgraph.s01_simple_agents.main_1_simple_chat
python -m p08_agents.c00_langchain_langgraph.s01_simple_agents.main_2_simple_chat_memory
python -m p08_agents.c00_langchain_langgraph.s01_simple_agents.main_3_simple_tools
python -m p08_agents.c00_langchain_langgraph.s01_simple_agents.main_4_simple_stock_analyser
python -m p08_agents.c00_langchain_langgraph.s01_simple_agents.main_5_simple_stock_analyser_chat
python -m p08_agents.c00_langchain_langgraph.s01_simple_agents.main_6_math_agent_hitl
# s02: RAG agent
python -m p08_agents.c00_langchain_langgraph.s02_rag_agent.main_rag_agent
# s03: multi-agent system with a supervisor
python -m p08_agents.c00_langchain_langgraph.s03_multi_agents_supervisor.main_0_multi_agent_supervisor
python -m p08_agents.c00_langchain_langgraph.s03_multi_agents_supervisor.main_1_multi_agent_supervisor_hitl
# s04: multi-agent swarm
python -m p08_agents.c00_langchain_langgraph.s04_multi_agent_swarm.main
# s05: agent with long-term memory
python -m p08_agents.c00_langchain_langgraph.s05_agents_with_memory.main_0_longterm_in_memory
```

| Script | What it shows | Interactive |
|---|---|---|
| `s00/main_very_simple_langgraph.py` | `StateGraph` with a typed state, nodes, a conditional edge choosing between a sum and a multiplication "tool" | no |
| `s00/main_very_simple_langgraph_looping.py` | the same with a loop: a node appends random numbers until a counter exceeds a threshold | no |
| `s01/main_1_simple_chat.py` | one-node chat graph without memory | chat loop (Ctrl+C to quit) |
| `s01/main_2_simple_chat_memory.py` | chat graph that keeps the conversation history | chat loop |
| `s01/main_3_simple_tools.py` | ReAct loop built by hand: LLM with bound tools, a conditional edge to a `ToolNode` and an edge back to the model, which calls tools until it can answer | no |
| `s01/main_4_simple_stock_analyser.py` | the same ReAct loop with a (mock) stock analysis tool | no |
| `s01/main_5_simple_stock_analyser_chat.py` | the same as a chat with `InMemorySaver` checkpointing | chat loop |
| `s01/main_6_math_agent_hitl.py` | `create_agent` with `HumanInTheLoopMiddleware`: approve, edit or reject a tool call, resume with `Command(resume=...)` | approval prompt |
| `s02/main_rag_agent.py` | agentic RAG: the agent decides when to call a Qdrant retrieval tool over the FINMA documents (and a mock stock analysis tool) | no |
| `s03/main_0_multi_agent_supervisor.py` | supervisor agent that calls a math agent and a RAG agent wrapped as tools | no |
| `s03/main_1_multi_agent_supervisor_hitl.py` | supervisor with an LLM query rewriter whose tool call needs human approval | approval prompt |
| `s04/main.py` | swarm of a math agent and a RAG agent handing over control with `langgraph-swarm` handoff tools | no |
| `s05/main_0_longterm_in_memory.py` | long-term memory: an `InMemoryStore` per user with retrieve/store-memory nodes and an upsert tool | no |

The s00 scripts save their graph diagram to `s00_very_simple_graphs/outputs/` (as PNG when the mermaid.ink
service is reachable, otherwise as Mermaid text, using `s00_very_simple_graphs/utils/graphs.py`). The other
scripts write their `graph_*.png` files next to the script with `draw_mermaid_png()`, which needs internet access
(mermaid.ink).
