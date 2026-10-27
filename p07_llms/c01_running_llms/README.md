## Environment
- Python conda environment to use: `conda_env_gzh_llm1.yml`
- Docker compose setup: `s01_ollama/docker-compose.yml` (for local ollama)
- Environment variables: API keys for OpenAI must be set in the environment variables. 
For local ollama the env variables are set in the docker compose file (i.e. port mapping).

## How to run
Start the local Ollama server and pull a model (once):

```bash
docker compose -f p07_llms/c01_running_llms/s01_ollama/docker-compose.yml up -d
docker exec -it gzh_ollama ollama pull qwen3:4b    # or any model from https://ollama.com/search
curl http://localhost:07011/api/tags               # lists the installed models
```

Without an NVIDIA GPU (and the NVIDIA Container Toolkit), remove the `deploy:` block from the compose file;
Ollama then runs on the CPU.

Run the examples from the repository root:

```bash
export OPENAI_API_KEY=...   # only for s00 and for LLM_TO_USE=openai
python -m p07_llms.c01_running_llms.s00_cloud_apis.main_openai
python -m p07_llms.c01_running_llms.s01_ollama.main_ollama
python -m p07_llms.c01_running_llms.s01_ollama.utils.helpers      # prints the installed Ollama models
python -m p07_llms.c01_running_llms.s02_langchain.main_langchain
python -m p07_llms.c01_running_llms.s03_hugging_face.main_huggingface  # downloads gpt2 from the HF Hub
```

| Script | Backend | Needs |
|---|---|---|
| `s00_cloud_apis/main_openai.py` | OpenAI Chat Completions API | `OPENAI_API_KEY`, internet |
| `s01_ollama/main_ollama.py` | Ollama REST API (`/api/generate`) via `requests` | Ollama container, model `qwen3` (i.e. `qwen3:latest`) pulled |
| `s02_langchain/main_langchain.py` | `ChatOllama` / `ChatOpenAI` via `get_llm()` | Ollama container (the script sets `LLM_TO_USE=ollama`) |
| `s03_hugging_face/main_huggingface.py` | `transformers` pipeline, in-process | internet for the first download; CPU is enough for `gpt2` |

Environment variables read by `get_llm()` (used by most later chapters): `LLM_TO_USE` (`ollama` or `openai`),
`OLLAMA_BASE_URL` (default `http://localhost:07011`), `OLLAMA_MODEL` (default: first installed model) and
`OPENAI_API_KEY`.
