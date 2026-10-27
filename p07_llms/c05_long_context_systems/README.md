## Environment
- Python conda environment to use: `conda_env_gzh_llm5.yml`
- Docker compose setup: `p07_llms/c01_running_llms/s01_ollama/docker-compose.yml` (local Ollama on
  `http://localhost:07011`), with a model pulled, e.g. `qwen3:8b`.
- Environment variables (all optional, no `.env` file needed):
  - `OLLAMA_BASE_URL` (default `http://localhost:07011`), `OLLAMA_MODEL` (default `qwen3:8b`)
  - `LLM_TO_USE=openai` and `OPENAI_API_KEY` to use OpenAI instead of Ollama
  - `NIAH_CONTEXT_LENGTHS` for the needle test, e.g. `1000,4000,8000` (default `1000,4000,8000,16000`)
- `tiktoken` downloads its vocabulary files on first use (internet needed once); without them the scripts
  fall back to an approximate count of 4 characters per token.

## How to run
Run from the repository root:

```bash
python -m p07_llms.c05_long_context_systems.main_1_token_budgeting          # offline: tokens, budgets, KV cache
python -m p07_llms.c05_long_context_systems.main_2_needle_in_haystack       # Ollama: needle test -> outputs/
python -m p07_llms.c05_long_context_systems.main_3_long_document_strategies # Ollama: truncation, map-reduce, refine
python -m p07_llms.c05_long_context_systems.main_4_chat_history_trimming    # trim_messages offline, LLM part optional
```

| Script | Shows |
|---|---|
| `main_1_token_budgeting.py` | token counts per text type (tiktoken), context budget planning, KV cache memory and attention cost |
| `main_2_needle_in_haystack.py` | needle-in-a-haystack grid (context length x depth) with `num_ctx`, CSV + heatmap in `outputs/`, truncation pitfall |
| `main_3_long_document_strategies.py` | truncation, map-reduce, refine and stuffing compared by fact recall, LLM calls and time |
| `main_4_chat_history_trimming.py` | `trim_messages` strategies vs summary memory for long chat histories |
