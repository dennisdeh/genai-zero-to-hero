## Environment
- Python conda environment to use: `conda_env_gzh_llm3.yml`
- Docker compose setup: `s03_more_unsloth/docker-compose.yml` (for more unsloth fine-tuning ntebooks)
- Environment variables: For the unsloth notebook the env variables are set in the docker compose file (i.e. port mapping).
- Hardware: the Unsloth scripts (`s00_peft_unsloth`, `s01_rl_unsloth`) require an NVIDIA GPU with CUDA.
  `s01_rl_unsloth` with the default `unsloth/Qwen2.5-3B-Instruct` needs roughly 12-16 GB of VRAM
  (use `unsloth/Qwen2.5-1.5B-Instruct` for smaller GPUs). Models and GSM8K are downloaded from HuggingFace on first run.

## How to run
All commands are run from the repository root with the `gzh_llm3` environment activated
(`conda env create -f p07_llms/c03_more_fine_tuning/conda_env_gzh_llm3.yml && conda activate gzh_llm3`).

### s00: supervised PEFT (LoRA) of GPT OSS with Unsloth
- Set `training = True` in `s00_peft_unsloth/main_gpt_oss_peft_unsloth.py` to fine-tune, or `False` (and
  `load_ft_model`) to run inference with saved adapters from `s00_peft_unsloth/trained_models/`.
- `python -m p07_llms.c03_more_fine_tuning.s00_peft_unsloth.main_gpt_oss_peft_unsloth`

### s01: reinforcement learning (GRPO) on GSM8K with TRL + Unsloth
- `python -m p07_llms.c03_more_fine_tuning.s01_rl_unsloth.main_grpo_unsloth`
  - `training = True` (default): evaluates the base model on 20 GSM8K test questions, trains LoRA adapters with
    GRPO and rule-based rewards (`max_steps = 250`), evaluates again and saves the adapters to
    `s01_rl_unsloth/outputs/grpo_lora_<timestamp>/` (gitignored).
  - `training = False`: set `load_ft_model` to the saved folder name to reload the adapters and evaluate.
  - Reduce `max_steps` (e.g. 50) for a quick smoke test; the logged `reward` and `rewards/<function>/mean` columns
    show the progress.
- The reward functions are pure Python and can be tested without a GPU (`pytest` is part of the conda env):
  `python -m pytest p07_llms/c03_more_fine_tuning/s01_rl_unsloth/test_rewards.py`
- Acknowledgement: the reward functions, prompt format and most GRPO hyperparameters are adapted from the Unsloth
  notebook "Qwen2.5 (3B) GRPO" by Unsloth AI (https://github.com/unslothai/notebooks, LGPL-3.0), which builds on the
  GRPO demo by Will Brown (https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb). The adapted portions
  remain under the LGPL-3.0; see the file headers and `sources.md`.

### s03: more Unsloth notebooks (docker)
- `docker compose -f p07_llms/c03_more_fine_tuning/s03_more_unsloth/docker-compose.yml up -d`, then open JupyterLab at http://localhost:07012 (needs the NVIDIA container toolkit).
