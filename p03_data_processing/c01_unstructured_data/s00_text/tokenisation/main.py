"""
Tokenising a real dataset with OpenAI's tiktoken (cl100k_base, the GPT-4 tokeniser).

The instructions of the finance-alpaca dataset (Hugging Face: gbharti/finance-alpaca) are tokenised and
concatenated into one long token stream, separated by the end-of-text token - exactly how text is
prepared for pre-training a GPT-like model. Some statistics about the token counts are printed.

Requires internet access on the first run (the dataset and the tiktoken vocabulary are downloaded and
cached). A from-scratch BPE tokeniser that runs fully offline is in main_bpe_tokeniser.py.

Run from the repository root:
    python -m p03_data_processing.c01_unstructured_data.s00_text.tokenisation.main
"""

import numpy as np
import tiktoken
from datasets import load_dataset

N_SAMPLES = 5000  # use a subset so the example runs quickly; set to None for all ~69k rows


def tokenise_documents(texts: list[str], encoding: tiktoken.Encoding) -> tuple[list[int], list[int]]:
    """
    Tokenise documents and concatenate them, separated by the end-of-text token.

    :param texts: documents to tokenise
    :param encoding: tiktoken encoding
    :return: (token stream, number of tokens per document)
    """
    eot_token = encoding.eot_token  # id of <|endoftext|> (100257 for cl100k_base)
    all_tokens, lengths = [], []
    for tokens in encoding.encode_ordinary_batch(texts):  # batch encoding uses several threads
        all_tokens.extend(tokens)
        all_tokens.append(eot_token)  # add EOT token after each document
        lengths.append(len(tokens))
    return all_tokens, lengths


if __name__ == "__main__":
    # 1. Load the finance-alpaca dataset (only the first N_SAMPLES rows of the training split)
    split = "train" if N_SAMPLES is None else f"train[:{N_SAMPLES}]"
    train_data = load_dataset("gbharti/finance-alpaca", split=split)
    instructions = train_data["instruction"]

    # 2. Initialise the encoding and tokenise all instructions
    encoding = tiktoken.get_encoding("cl100k_base")
    all_tokens, lengths = tokenise_documents(instructions, encoding)

    # 3. Statistics
    n_chars = sum(len(t) for t in instructions)
    print(f"Number of instructions tokenised: {len(instructions)}")
    print(f"Total number of tokens (incl. one EOT per instruction): {len(all_tokens)}")
    print(f"Unique tokens: {len(set(all_tokens))} of a vocabulary of {encoding.n_vocab}")
    print(f"Average tokens per instruction: {np.mean(lengths):.2f} (median {np.median(lengths):.0f}, "
          f"95th percentile {np.percentile(lengths, 95):.0f}, max {max(lengths)})")
    print(f"Characters per token: {n_chars / sum(lengths):.2f}")

    # 4. Look at one example
    example = instructions[0]
    ids = encoding.encode_ordinary(example)
    print(f"\nexample: {example!r}")
    print(f"ids:     {ids}")
    print(f"tokens:  {[encoding.decode([i]) for i in ids]}")
