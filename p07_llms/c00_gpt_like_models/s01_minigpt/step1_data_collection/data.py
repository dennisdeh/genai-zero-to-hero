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


class TextDataset(Dataset):
    """
    Custom dataset class for text data with token-based operations.
    """

    def __init__(self, tokens, block_size, pad_token: int = 0):
        self.tokens = tokens
        self.block_size = block_size
        self.pad_token = pad_token

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

        # Pad if the chunk is shorter than block_size
        if len(x) < self.block_size:
            padding_length = self.block_size - len(x)
            x = torch.cat(
                [x, torch.full((padding_length,), self.pad_token, dtype=torch.long)]
            )
            y = torch.cat(
                [y, torch.full((padding_length,), self.pad_token, dtype=torch.long)]
            )

        return x, y


class QnADataset(Dataset):
    def __init__(self, tokens, pad_token: int = 0):
        """
        Custom dataset class for discrete QnA samples

        Args:
            tokens: List of token lists, where each list represents one QnA pair
            pad_token: The padding token ID
        """
        # Filter out None values and empty samples
        self.tokens = [s for s in tokens if s is not None and len(s) > 0]

        if len(self.tokens) == 0:
            raise ValueError("Cannot create QnADataset with empty samples list")

        self.pad_token = pad_token
        self.block_size = len(self.tokens[0])

        # Validate all samples have the same length
        for i, token in enumerate(self.tokens):
            if len(token) != self.block_size:
                raise ValueError(
                    f"Sample {i} has length {len(token)}, expected {self.block_size}. "
                    f"All samples must have the same length."
                )

    def __len__(self):
        return len(self.tokens)

    def get_block_size(self):
        return self.block_size

    def __getitem__(self, idx):
        # Get the sample (already padded to consistent length)
        tokens = self.tokens[idx]

        # x is the sequence, y is the sequence shifted by 1 (the targets)
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)

        return x, y


def data_preparation_wiki(
    d_articles: dict,
    tokeniser: Any,
    str_sot: str = "<|startoftext|>",
    str_eot: str = "<|endoftext|>",
    str_pad: str = "<|pad|>",
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
        str_sot: The start-of-text token.
        str_eot: The end-of-text token.
        str_pad: The padding token.
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
    assert isinstance(str_sot, str), "str_sot must be a string"
    assert isinstance(str_eot, str), "str_eot must be a string"
    assert isinstance(str_pad, str), "str_pad must be a string"
    assert (
        isinstance(val_pct, float) and 0 <= val_pct <= 1
    ), "val_pct must be a float between 0 and 1"
    assert isinstance(sampling_strategy, str), "sampling_strategy must be a string"
    # useful messages
    print("Settings:")
    print(f"   Number of articles: {len(d_articles)}")
    print(f"   Number of sentences: {sum(len(v) for v in d_articles.values())}")
    print(f"   Tokeniser: {tokeniser.__class__.__name__}")
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
    # remove empty strings:
    ls_train = [x for x in ls_train if len(x.strip()) > 0]
    ls_val = [x for x in ls_val if len(x.strip()) > 0]

    print(
        f"Length of text in the samples created:\n"
        f"    Training: {n_train}\n"
        f"    Validation: {n_val}"
    )

    # 2: Tokenisation
    # Initialise encoding and special tokens
    if hasattr(tokeniser, "encode_ordinary"):
        # Tiktoken
        sot_token = tokeniser.encode_ordinary(str_sot)[0]
        eot_token = tokeniser.encode_ordinary(str_eot)[0]
        pad_token = tokeniser.encode_ordinary(str_pad)[0]
    else:
        # Hugging Face tokenisers style
        sot_token = tokeniser.token_to_id(str_sot)
        eot_token = tokeniser.token_to_id(str_eot)
        pad_token = tokeniser.token_to_id(str_pad)

    # Tokenise all sentences of all articles in each data set: add start of text token, end of
    # text token before each sentence.
    data_train = []
    max_tokens = 0
    for article in ls_train:
        if hasattr(tokeniser, "encode_ordinary"):
            tokens = tokeniser.encode(article)
        else:
            tokens = tokeniser.encode(article).ids

        max_tokens = max(max_tokens, len(tokens))
        data_train.extend([sot_token] + tokens + [eot_token])

    data_val = []
    for article in ls_val:
        if hasattr(tokeniser, "encode_ordinary"):
            tokens = tokeniser.encode(article)
        else:
            tokens = tokeniser.encode(article).ids

        max_tokens = max(max_tokens, len(tokens))
        data_val.extend([sot_token] + tokens + [eot_token])

    # Determine the maximum number of tokens in the data and other statistics
    print(
        "Number of tokens in the datasets created:\n"
        f"   Training: {len(data_train)}\n"
        f"   Validation: {len(data_val)}\n"
        f"   Maximum number of tokens per article: {max_tokens}"
    )

    # 3: Instantiate a custom dataset classes
    dataset_train = TextDataset(data_train, block_size, pad_token=pad_token)
    dataset_val = TextDataset(data_val, block_size, pad_token=pad_token)

    # Print final statistics and summary
    print(
        f"Statistics of training and validation data:\n"
        f"   Total number of tokens: {len(data_train) + len(data_val)}\n"
        f"   Block size: {block_size} tokens\n"
        f"   Unique tokens: {len(set(data_train + data_val))}\n"
        f"   Vocabulary size: {tokeniser.n_vocab if hasattr(tokeniser, 'n_vocab') else tokeniser.get_vocab_size()}\n"
        f"   Average number of tokens per sentence (training): {len(data_train)/len(ls_train):.2f}\n"
        f"   Average number of tokens per sentence (validation): {len(data_val)/len(ls_val):.2f}"
    )

    return dataset_train, dataset_val


def data_preparation_finance(
    d_qna: dict,
    tokeniser: Any,
    str_sot: str = "<|startoftext|>",
    str_eot: str = "<|endoftext|>",
    str_sep: str = "<|separation|>",
    str_pad: str = "<|pad|>",
    val_pct: float = 0.1,
    block_size: int = 512,
):
    """
    The function collects an example QnA dataset, training and validation splits, tokenises
    them, and creates custom dataset classes, taking the input data in the format
    provided by dc_wiki in the data collection module.

    Parameters:
        d_qna: The dictionary of articles returned by dc_finance_qna or similar functions
            where each key is the question and the value is the answer.
        tokeniser: The tokeniser object used for tokenisation.
        str_sot: The start-of-text token.
        str_eot: The end-of-text token.
        str_sep: The separation token.
        str_pad: The padding token.
        val_pct: The percentage of data to use for validation.
        block_size: The maximum number of tokens in a context window.

    Returns:
        dataset_train: Ready-to-use PyTorch dataset for training.
        dataset_val: Ready-to-use PyTorch dataset for validation.
        encoding: The tokeniser object
    """
    # 0: Initialisation and data collection
    print(" *** 1. Data preparation *** ")
    # assertions
    assert isinstance(d_qna, dict), "d_qna must be a dictionary of articles"
    assert all(
        isinstance(v, str) for v in d_qna.values()
    ), "All values in d_qna must be lists of sentences"
    assert isinstance(
        tokeniser, (tiktoken.core.Encoding, tokenizers.Tokenizer)
    ), "tokeniser must be from a supported class"
    assert (
        isinstance(block_size, int) and block_size > 0
    ), "block_size must be a positive integer"
    assert isinstance(str_sot, str), "str_sot must be a string"
    assert isinstance(str_eot, str), "str_eot must be a string"
    assert isinstance(str_pad, str), "str_pad must be a string"
    assert (
        isinstance(val_pct, float) and 0 <= val_pct <= 1
    ), "val_pct must be a float between 0 and 1"
    # useful messages
    print("Settings:")
    print(f"   Number of questions: {len(d_qna)}")
    print(f"   Tokeniser: {tokeniser.__class__.__name__}")
    print(f"   Block size: {block_size} tokens")
    print(f"   Validation percentage: {val_pct * 100:.1f}%")

    # 1: Split into train and validation sets
    qna_pairs = list(d_qna.items())  # [(question, answer), ...]
    random.shuffle(qna_pairs)
    split_idx = int(len(qna_pairs) * (1 - val_pct))
    ls_train = qna_pairs[:split_idx]
    ls_val = qna_pairs[split_idx:]

    # 2: Tokenisation
    # Initialise encoding and special tokens
    if hasattr(tokeniser, "encode_ordinary"):
        # Tiktoken
        sot_token = tokeniser.encode_ordinary(str_sot)[0]
        eot_token = tokeniser.encode_ordinary(str_eot)[0]
        sep_token = tokeniser.encode_ordinary(str_sep)[0]
        pad_token = tokeniser.encode_ordinary(str_pad)[0]
    else:
        # Hugging Face tokenisers style
        sot_token = tokeniser.token_to_id(str_sot)
        eot_token = tokeniser.token_to_id(str_eot)
        sep_token = tokeniser.token_to_id(str_sep)
        pad_token = tokeniser.token_to_id(str_pad)

    # First pass: calculate max_tokens and filter/truncate samples
    max_tokens = 0
    n_too_long = 0
    n_truncated = 0
    filtered_train = []
    filtered_val = []

    for q, a in ls_train:
        if hasattr(tokeniser, "encode_ordinary"):
            q_tokens = tokeniser.encode(q)
            a_tokens = tokeniser.encode(a)
        else:
            q_tokens = tokeniser.encode(q).ids
            a_tokens = tokeniser.encode(a).ids

        total_len = len(q_tokens) + len(a_tokens) + 3  # +3 for sot, sep, eot
        if total_len > block_size:
            # Truncate answer to fit within block_size instead of discarding
            available_for_answer = block_size - len(q_tokens) - 3
            if (
                available_for_answer > 20
            ):  # Keep only if we have reasonable space for answer
                a_tokens = a_tokens[:available_for_answer]
                total_len = len(q_tokens) + len(a_tokens) + 3
                n_truncated += 1
                filtered_train.append((q, q_tokens, a, a_tokens))
                max_tokens = max(max_tokens, total_len)
            else:
                n_too_long += 1
        else:
            filtered_train.append((q, q_tokens, a, a_tokens))
            max_tokens = max(max_tokens, total_len)

    for q, a in ls_val:
        if hasattr(tokeniser, "encode_ordinary"):
            q_tokens = tokeniser.encode(q)
            a_tokens = tokeniser.encode(a)
        else:
            q_tokens = tokeniser.encode(q).ids
            a_tokens = tokeniser.encode(a).ids

        total_len = len(q_tokens) + len(a_tokens) + 3
        if total_len > block_size:
            # Truncate answer to fit within block_size instead of discarding
            available_for_answer = block_size - len(q_tokens) - 3
            if (
                available_for_answer > 20
            ):  # Keep only if we have reasonable space for answer
                a_tokens = a_tokens[:available_for_answer]
                total_len = len(q_tokens) + len(a_tokens) + 3
                n_truncated += 1
                filtered_train.append((q, q_tokens, a, a_tokens))
                max_tokens = max(max_tokens, total_len)
            else:
                n_too_long += 1
        else:
            filtered_val.append((q, q_tokens, a, a_tokens))
            max_tokens = max(max_tokens, total_len)

    print(
        f"Training samples: {len(filtered_train)}, Validation samples: {len(filtered_val)}"
    )
    if n_truncated > 0:
        print(f"Note: Truncated {n_truncated} QnA pairs to fit within block_size")
    if n_too_long > 0:
        print(
            f"Warning: Discarded {n_too_long} QnA pairs (question too long or insufficient space for answer)"
        )

    # Second pass: create discrete samples with consistent padding
    data_train = []
    for q, q_tokens, a, a_tokens in filtered_train:
        sample_len = len(q_tokens) + len(a_tokens) + 3
        pad_tokens = block_size - sample_len
        data_train.append(
            [sot_token]
            + q_tokens
            + [sep_token]
            + a_tokens
            + [eot_token]
            + [pad_token] * pad_tokens
        )

    data_val = []
    for q, q_tokens, a, a_tokens in filtered_val:
        sample_len = len(q_tokens) + len(a_tokens) + 3
        pad_tokens = block_size - sample_len
        data_val.append(
            [sot_token]
            + q_tokens
            + [sep_token]
            + a_tokens
            + [eot_token]
            + [pad_token] * pad_tokens
        )

    # Determine the maximum number of tokens in the data and other statistics
    total_train_tokens = sum(len(sample) for sample in data_train)
    total_val_tokens = sum(len(sample) for sample in data_val)
    all_tokens = [token for sample in data_train + data_val for token in sample]

    print(
        "Number of tokens in the datasets created:\n"
        f"   Training: {total_train_tokens}\n"
        f"   Validation: {total_val_tokens}\n"
        f"   Sample length (including special tokens and padding): {max_tokens}"
    )
    # assertions
    assert total_train_tokens > 0, "No training samples generated"
    assert total_val_tokens > 0, "No validation samples generated"
    assert (
        total_train_tokens / len(data_train) == block_size
    ), "Inconsistent tokenisation"

    # 3: Instantiate a custom dataset classes
    dataset_train = QnADataset(data_train, pad_token=pad_token)
    dataset_val = QnADataset(data_val, pad_token=pad_token)

    # Print final statistics and summary
    print(
        f"Statistics of training and validation data:\n"
        f"   Total number of samples: {len(data_train) + len(data_val)}\n"
        f"   Training samples: {len(data_train)}\n"
        f"   Validation samples: {len(data_val)}\n"
        f"   Sample length: {max_tokens} tokens\n"
        f"   Block size: {block_size} tokens\n"
        f"   Unique tokens: {len(set(all_tokens))}\n"
        f"   Vocabulary size: {tokeniser.n_vocab if hasattr(tokeniser, 'n_vocab') else tokeniser.get_vocab_size()}\n"
    )

    return dataset_train, dataset_val


if __name__ == "__main__2":
    # Wiki dataset
    # Data collection:
    # Initialisation and data collection
    d, _ = dc.dc_wiki(n_samples=1000, list_sentences=True)
    # Select tokeniser
    tokeniser = tiktoken.get_encoding("cl100k_base")
    # Prepare data
    dataset_train, dataset_val = data_preparation_wiki(
        d_articles=d, tokeniser=tokeniser, block_size=256
    )

    n_limit = 10
    n0 = 0
    for x, y in dataset_train:
        print(f"input {n0}:")
        print(f"   {tokeniser.decode(x.tolist())}")
        print(f"target {n0}:")
        print(f"   {tokeniser.decode(y.tolist())}")
        print(" ---------------- ")
        if n0 >= n_limit:
            break
        else:
            n0 += 1

    # Finance QnA dataset
    # Initialisation and data collection
    d = dc.dc_finance_qna()
    # Select tokeniser
    tokeniser = tiktoken.get_encoding("cl100k_base")
    # Prepare data
    dataset_train, dataset_val = data_preparation_finance(
        d_qna=d, tokeniser=tokeniser, block_size=256
    )
    n_limit = 10
    n0 = 0
    for x, y in dataset_val:
        print(f"input {n0}:")
        print(f"   {tokeniser.decode(x.tolist())}")
        print(f"target {n0}:")
        print(f"   {tokeniser.decode(y.tolist())}")
        print(" ---------------- ")
        if n0 >= n_limit:
            break
        else:
            n0 += 1
