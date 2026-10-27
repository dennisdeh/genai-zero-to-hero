# Unstructured data: text cleaning and tokenisation

## Environment
- Python conda environment to use: `p03_data_processing/conda_env_gzh_proc0.yml`
  (`conda env create -f p03_data_processing/conda_env_gzh_proc0.yml && conda activate gzh_proc0`)
- No docker setup or environment variables are needed.
- Internet access is only needed for `tokenisation/main.py` (downloads the `gbharti/finance-alpaca` dataset
  from Hugging Face and the `cl100k_base` vocabulary of tiktoken on the first run; both are cached).
  `main_bpe_tokeniser.py` uses the bundled corpus `tokenisation/data/corpus.txt` and runs offline; it only
  adds `cl100k_base` to its comparison if the vocabulary can be downloaded.

## How to run
From the repository root:

```bash
# cleaning and normalising raw text (HTML, Unicode, PII masking, deduplication)
python -m p03_data_processing.c01_unstructured_data.s00_text.text_cleaning.main_text_cleaning

# byte-pair encoding from scratch vs. HF tokenizers vs. tiktoken (offline)
python -m p03_data_processing.c01_unstructured_data.s00_text.tokenisation.main_bpe_tokeniser

# tokenise the finance-alpaca instructions with tiktoken cl100k_base (needs internet on first run)
python -m p03_data_processing.c01_unstructured_data.s00_text.tokenisation.main
```

The trained from-scratch tokeniser is saved to `s00_text/tokenisation/outputs/scratch_bpe.json`.

## Structure
- `s00_text/text_cleaning/main_text_cleaning.py`: text cleaning pipeline (standard library only)
- `s00_text/tokenisation/utils/bpe.py`: the from-scratch byte-level BPE tokeniser (`BPETokeniser`)
- `s00_text/tokenisation/main_bpe_tokeniser.py`: training, round trip, export to tiktoken, comparisons
- `s00_text/tokenisation/main.py`: tokenising a Hugging Face dataset with tiktoken
- `s00_text/tokenisation/data/corpus.txt`: small original corpus for offline training
