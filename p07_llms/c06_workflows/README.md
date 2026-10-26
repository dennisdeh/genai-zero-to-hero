## Environment
- Python conda environment to use: `conda_env_gzh_llm6.yml`
- Docker compose setup: `p07_llms/c01_running_llms/s01_ollama/docker-compose.yml` (local Ollama on
  `http://localhost:07011`), with a model pulled, e.g. `qwen3:8b`.
- Environment variables (all optional, no `.env` file needed):
  - `OLLAMA_BASE_URL` (default `http://localhost:07011`), `OLLAMA_MODEL` (default `qwen3:8b`)
  - `LLM_TO_USE=openai` and `OPENAI_API_KEY` to use OpenAI instead of Ollama
- The workflow diagrams are saved as Mermaid text in `outputs/`; PNG rendering uses the mermaid.ink web
  service and is skipped without internet access.

## How to run
Run from the repository root:

```bash
python -m p07_llms.c06_workflows.main_1_prompt_chaining
python -m p07_llms.c06_workflows.main_2_routing
python -m p07_llms.c06_workflows.main_3_parallelisation
python -m p07_llms.c06_workflows.main_4_orchestrator_workers
python -m p07_llms.c06_workflows.main_5_evaluator_optimiser
```

| Script | Pattern | LangGraph features |
|---|---|---|
| `main_1_prompt_chaining.py` | outline -> gate -> draft -> translate | sequential edges, conditional edge to `END`, `stream_mode="updates"` |
| `main_2_routing.py` | structured-output classifier -> specialised handlers | `with_structured_output`, conditional edges |
| `main_3_parallelisation.py` | sectioning (3 analysts) and voting (phishing detection) | parallel edges, fan-in with `add_edge([...], node)`, `Send`, `operator.add` reducer |
| `main_4_orchestrator_workers.py` | planner -> parallel section writers -> synthesiser | `Send` API for dynamic fan-out |
| `main_5_evaluator_optimiser.py` | generator <-> evaluator loop (max iterations) | loops with conditional edges, rule checks + LLM judge |
