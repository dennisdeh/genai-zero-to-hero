"""
In this script we will take a look at the data collection and preparation for training an LLM

We do not do any further data preprocessing or analysis here to check the quality of the data
beyond a simple sanity check and statistics.
"""

import random
from typing import Any
import tiktoken
import tokenizers
from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from p07_llms.c00_gpt_like_models.s01_minigpt.step1_data_collection import (
    data_collection as dc,
)


# Define a custom dataset class appropriate for our task
class TextDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def get_block_size(self):
        return self.block_size

    def __getitem__(self, idx):
        # We grab a chunk of tokens of length block_size + 1
        # x is the sequence, y is the sequence shifted by 1 (the targets)
        chunk = self.tokens[idx : idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def data_preparation_wiki(
    d_articles: dict,
    tokeniser: Any,
    str_eot: str = "<|endoftext|>",
    val_pct: float = 0.1,
    sampling_strategy: str = "random sentences",
    block_size: int = 512,
):
    """
    The function collects an example dataset, training and validation splits, tokenises
    them, and creates custom dataset classes, taking the input data in the format
    provided by dc_wiki in the data collection module.

    Parameters:
        d_articles: The dictionary of articles returned by dc_wiki.
        tokeniser: The tokeniser object used for tokenisation.
        str_eot: The end-of-text token.
        val_pct: The percentage of data to use for validation.
        sampling_strategy: The strategy for sampling sentences; must be "random sentences"
            or "train first validate last"
        block_size: The maximum number of tokens in a context window.

    Returns:
        dataset_train: Ready-to-use PyTorch dataset for training.
        dataset_val: Ready-to-use PyTorch dataset for validation.
        encoding: The tokeniser object
    """
    # 0: Initialisation and data collection
    print(" *** 1. Data preparation *** ")
    # assertions
    assert isinstance(d_articles, dict), "d_articles must be a dictionary of articles"
    assert all(
        isinstance(v, list) for v in d_articles.values()
    ), "All values in d_articles must be lists of sentences"
    assert isinstance(
        tokeniser, (tiktoken.core.Encoding, tokenizers.Tokenizer)
    ), "tokeniser must be from a supported class"
    assert (
        isinstance(block_size, int) and block_size > 0
    ), "block_size must be a positive integer"
    assert isinstance(str_eot, str), "str_eot must be a string"
    assert (
        isinstance(val_pct, float) and 0 <= val_pct <= 1
    ), "val_pct must be a float between 0 and 1"
    assert isinstance(sampling_strategy, str), "sampling_strategy must be a string"
    # useful messages
    print("Settings:")
    print(f"   Number of articles: {len(d_articles)}")
    print(f"   Number of sentences: {sum(len(v) for v in d_articles.values())}")
    print(f"   Tokeniser: {tokeniser.__str__}")
    print(f"   Block size: {block_size} tokens")
    print(f"   Data collection strategy: {sampling_strategy}")
    print(f"   Validation percentage: {val_pct * 100:.1f}%")

    # 1: Split into train and validation sets
    d_val = {}
    d_train = {}
    n_val = 0
    n_train = 0
    if sampling_strategy == "random sentences":
        # Sample sentences from each article val_pct of the sentences for the
        # validation and training sets. Like this, all topics should be represented in both sets,
        # except if the article is very short.
        print(f"Sampling {val_pct * 100:.1f}% of sentences randomly in each article")
        for t, v in d_articles.items():
            d_val[t] = random.sample(v, k=int(len(v) * val_pct))
            d_train[t] = [s for s in v if s not in d_val[t]]
            # join the sentences back together for each article
            d_train[t] = " ".join(d_train[t])
            d_val[t] = " ".join(d_val[t])
            # get statistics for the current article
            n_val += len(d_val[t])
            n_train += len(d_train[t])
    elif sampling_strategy == "train first validate last":
        # Sample the first val_pct% of sentences from each article for the training set,
        # and the rest for the validation set.
        print(
            f"Sampling {(1-val_pct) * 100:.1f}% of sentences from the beginning in each article for "
            f"training; the rest for validation"
        )
        for t, v in d_articles.items():
            d_train[t] = " ".join(v[: int(len(v) * (1 - val_pct))])
            d_val[t] = " ".join(v[int(len(v) * (1 - val_pct)) :])
            # get statistics for the current article
            n_val += len(d_val[t])
            n_train += len(d_train[t])
    else:
        raise ValueError("Invalid sample type")

    # create lists of articles for training and validation sets
    ls_train = list(d_train.values())
    ls_val = list(d_val.values())

    print(
        f"Length of text in the samples created:\n"
        f"    Training: {n_train}\n"
        f"    Validation: {n_val}"
    )

    # 2: Tokenisation
    # Initialize encoding and get EOT token
    eot_token = tokeniser.encode_ordinary(str_eot)[0]  # End of text token

    # Tokenise all articles in each data set and concatenate with EOT token
    data_train = []
    max_tokens = 0
    for article in ls_train:
        tokens = tokeniser.encode(article)
        max_tokens = max(max_tokens, len(tokens))
        data_train.extend(tokens)
        data_train.append(eot_token)

    data_val = []
    for article in ls_val:
        tokens = tokeniser.encode(article)
        max_tokens = max(max_tokens, len(tokens))
        data_val.extend(tokens)
        data_val.append(eot_token)

    # Determine the maximum number of tokens in the data and other statistics
    print(
        "Number of tokens in the datasets created:\n"
        f"   Training: {len(data_train)}\n"
        f"   Validation: {len(data_val)}\n"
        f"   Maximum number of tokens per article: {max_tokens}"
    )

    # 3: Instantiate a custom dataset classes
    dataset_train = TextDataset(data_train, block_size)
    dataset_val = TextDataset(data_val, block_size)

    # Print final statistics and summary
    print(
        f"Statistics of training and validation data:\n"
        f"   Total number of tokens: {len(data_train) + len(data_val)}\n"
        f"   Block size: {block_size} tokens\n"
        f"   Unique tokens: {len(set(data_train + data_val))}\n"
        f"   Vocabulary size: {tokeniser.n_vocab}\n"
        f"   Average number of tokens per sentence (training): {len(data_train)/len(ls_train):.2f}\n"
        f"   Average number of tokens per sentence (validation): {len(data_val)/len(ls_val):.2f}"
    )

    return dataset_train, dataset_val


def data_preparation_finance(block_size: int = 256, qa_pairs=None):
    """
    The function collects the finance-alpaca dataset, training and validation splits, tokenises
    them, and creates custom dataset classes.

    The block size is checked to ensure that it is larger than the maximum number of tokens in the data.

    Returns:
        dataset_train: Ready-to-use PyTorch dataset for training.
        dataset_val: Ready-to-use PyTorch dataset for validation.
        encoding: The tokeniser object
    """
    print(" *** 1. Data collection and preparation *** ")
    # 1: Data collection
    # Use the finance-alpaca dataset (this contains only data labelled "train"; we split it manually into
    # train and validation sets), and we take the "instruction" column
    ds = load_dataset("gbharti/finance-alpaca", split="train")
    ls_instructions = list(ds["instruction"])
    print(f"Text examples: {ls_instructions[:2]}")

    # Longest instruction in the dataset
    max_len = max(len(instruction) for instruction in ls_instructions)
    print(f"Longest example text: {max_len} characters")

    # 2: Split into train and validation sets
    random.shuffle(ls_instructions)
    split_idx = int(len(ls_instructions) * 0.9)
    ls_train = ls_instructions[:split_idx]
    ls_val = ls_instructions[split_idx:]
    print(f"Training samples: {len(ls_train)}, Validation samples: {len(ls_val)}")

    # 3: Tokenisation
    # Initialize encoding and get EOT token
    encoding = tiktoken.get_encoding("cl100k_base")
    eot_token = encoding.encode_ordinary("<|endoftext|>")[0]  # End of text token

    # Tokenise all data and concatenate with EOT token
    data_train = []
    max_tokens = 0
    for instruction in ls_train:
        tokens = encoding.encode(instruction)
        max_tokens = max(max_tokens, len(tokens))
        data_train.extend(tokens)
        data_train.append(eot_token)  # Add EOT token after each instruction

    data_val = []
    for instruction in ls_val:
        tokens = encoding.encode(instruction)
        max_tokens = max(max_tokens, len(tokens))
        data_val.extend(tokens)
        data_val.append(eot_token)
    # Determine the block size (number of tokens in a context)

    assert (
        max_tokens <= block_size
    ), "The block size must be larger than the maximum number of tokens in the data"
    print(f"Block size: {block_size} tokens > {max_tokens} maximum tokens in data")

    # instantiate a custom dataset classes
    dataset_train = TextDataset(data_train, block_size)
    dataset_val = TextDataset(data_val, block_size)

    # Print final statistics and summary
    print(
        f"Statistics of training and validation data:\n"
        f"  Total number of tokens: {len(data_train) + len(data_val)}\n"
        f"  Unique tokens: {len(set(data_train + data_val))}\n"
        f"  Vocabulary size: {encoding.n_vocab}\n"
        f"  Average number of tokens per sentence (training): {len(data_train)/len(ls_train):.2f}\n"
        f"  Average number of tokens per sentence (validation): {len(data_val)/len(ls_val):.2f}"
    )

    return dataset_train, dataset_val, encoding


if __name__ == "__main__2":
    # Data collection:
    # Initialisation and data collection
    d, _ = dc.dc_wiki(n_samples=1000, list_sentences=True)
    # Select tokeniser
    tknsr = tiktoken.get_encoding("cl100k_base")
    # Prepare data
    dataset_train, dataset_val = data_preparation_wiki(
        d_articles=d, tokeniser=tknsr, block_size=256
    )

    n_limit = 10
    n0 = 0
    for x, y in dataset_train:
        print(f"input {n0}:")
        print(f"   {tknsr.decode(x.tolist())}")
        print(f"target {n0}:")
        print(f"   {tknsr.decode(y.tolist())}")
        print(" ---------------- ")
        if n0 >= n_limit:
            break
        else:
            n0 += 1
