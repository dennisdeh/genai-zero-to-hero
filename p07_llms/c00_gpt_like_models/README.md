## Environment
- Python conda environment to use: `conda_env_gzh_llm0.yml`
- Docker compose setup: None
- Environment variables: None

## How to run
`s00_tinygpt` runs on a CPU in about a minute. Its `main.py` imports `transformer_blocks` as a top-level module,
so run it from inside the section folder:

```bash
cd p07_llms/c00_gpt_like_models/s00_tinygpt
python main.py
```

`s01_minigpt` is run from the repository root. Pretraining a GPT-2-sized model needs a (preferably several)
CUDA GPU(s); the datasets (Simple English Wikipedia, finance-alpaca) are downloaded from the Hugging Face Hub
on first use, and the scripts additionally need `datasets`, `nltk`, `tokenizers`, `tiktoken` and `mlflow`.
Training runs are logged to an MLflow tracking server expected at `http://localhost:5000`
(e.g. `mlflow server --port 5000`).

```bash
# 1. clean the Wikipedia data and train the BPE tokeniser (saved to s01_minigpt/trained_tokenisers/)
python -m p07_llms.c00_gpt_like_models.s01_minigpt.main_step1_data_preparation
# 2. pretraining: set `training = True` at the top of the script (False = inference with `load_model`)
python -m p07_llms.c00_gpt_like_models.s01_minigpt.main_step2_pretraining
# 3. full fine-tuning on finance Q&A: set `training = True` and `load_pt_model` to the model saved in step 2
python -m p07_llms.c00_gpt_like_models.s01_minigpt.main_step3_finetuning_full
# unit test: minGPT reproduces Hugging Face GPT-2 (needs `transformers`, downloads the GPT-2 weights)
python -m unittest p07_llms.c00_gpt_like_models.s01_minigpt.mingpt.tests.test_huggingface_import
```

Models are saved to `s01_minigpt/trained_models/`; both `trained_models/` and `trained_tokenisers/` are
gitignored. The original minGPT demos in `s01_minigpt/projects/` (`adder`, `chargpt`) use the import style
`from mingpt.model import GPT`, so both the repository root and `s01_minigpt` must be on the Python path. They
train until interrupted (Ctrl+C) and write checkpoints to `./out/`:

```bash
PYTHONPATH=.:p07_llms/c00_gpt_like_models/s01_minigpt python p07_llms/c00_gpt_like_models/s01_minigpt/projects/adder/adder.py
```
