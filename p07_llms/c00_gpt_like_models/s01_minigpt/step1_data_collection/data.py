"""
In this script we will take a look at the data collection and preparation for training an LLM

We do not do any further data preprocessing or analysis here to check the quality of the data
beyond a simple sanity check and statistics.
"""

import random
import tiktoken
from datasets import load_dataset
import torch
from torch.utils.data import Dataset


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


def data_preparation(block_size: int = 256):
    """
    The function collects an example dataset, training and validation splits, tokenises
    them, and creates custom dataset classes.

    The block size is checked to ensure that it is larger than the maximum number of tokens in the data.

    Returns:
        dataset_train: Ready-to-use PyTorch dataset for training.
        dataset_val: Ready-to-use PyTorch dataset for validation.
        encoding: The tokeniser object
    """
    print(" *** 1. Data collection and preparation *** ")
    # 1: Data collection
    # Use the finance-alpaca dataset as an example (this contains only data labelled "train";
    # we split it manually into train and validation sets), and we take the "instruction" column
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
