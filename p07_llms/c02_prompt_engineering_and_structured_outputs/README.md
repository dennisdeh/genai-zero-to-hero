## Environment
- Python conda environment to use: `conda_env_gzh_llm2.yml`
- Docker compose setup: `p07_llms/c01_running_llms/s01_ollama/docker-compose.yml` (local Ollama on
  `http://localhost:07011`). Pull a model once, e.g. `docker exec -it <ollama container> ollama pull qwen3:8b`.
- Environment variables (all optional, no `.env` file needed):
  - `OLLAMA_BASE_URL` (default `http://localhost:07011`)
  - `OLLAMA_MODEL` (default `qwen3:8b`; falls back to the first installed model)
  - `LLM_TO_USE=openai` and `OPENAI_API_KEY` to use OpenAI instead of Ollama

## How to run
Run from the repository root:

```bash
# s00 - prompt engineering
python -m p07_llms.c02_prompt_engineering_and_structured_outputs.s00_prompt_engineering.main_1_zero_vs_few_shot
python -m p07_llms.c02_prompt_engineering_and_structured_outputs.s00_prompt_engineering.main_2_roles_and_chain_of_thought
python -m p07_llms.c02_prompt_engineering_and_structured_outputs.s00_prompt_engineering.main_3_templates_and_temperature

# s01 - structured outputs
python -m p07_llms.c02_prompt_engineering_and_structured_outputs.s01_structured_outputs.main_1_with_structured_output
python -m p07_llms.c02_prompt_engineering_and_structured_outputs.s01_structured_outputs.main_2_invoice_extraction_with_retry
python -m p07_llms.c02_prompt_engineering_and_structured_outputs.s01_structured_outputs.main_3_compare_output_methods
```

| Script | Shows |
|---|---|
| `s00/main_1_zero_vs_few_shot.py` | zero-shot vs role prompt vs few-shot (`FewShotChatMessagePromptTemplate`) on headline sentiment, with accuracy |
| `s00/main_2_roles_and_chain_of_thought.py` | system prompts / personas; direct vs chain-of-thought vs built-in thinking on finance word problems |
| `s00/main_3_templates_and_temperature.py` | templates with variables, `partial`, `MessagesPlaceholder`, LCEL `batch`; temperature sweep with diversity metrics |
| `s01/main_1_with_structured_output.py` | `with_structured_output` with enums, nested models and lists; `include_raw=True` |
| `s01/main_2_invoice_extraction_with_retry.py` | pydantic validators + retry loop that feeds validation errors back to the model |
| `s01/main_3_compare_output_methods.py` | manual parsing, `PydanticOutputParser`, JSON mode, JSON schema and tool calling compared |
