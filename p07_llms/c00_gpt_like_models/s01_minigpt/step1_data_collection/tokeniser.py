"""
This module provides a tokeniser for the MinGPT model using Byte Pair Encoding (BPE) trained
on the Wikipedia data prepared in `data_collection.py`.

The pre-tokeniser is designed to handle byte-level tokenisation, which is robust to weird
characters and can be useful for text processing tasks.

It uses the Huggingface `tokenizers` library: https://github.com/huggingface/tokenizers
"""

import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing
import p07_llms.c00_gpt_like_models.s01_minigpt.step1_data_collection.data_collection as dc


def train_custom_tokeniser(
    d_articles: dict,
    vocab_size: int = 32000,
    unknown_token: str = "<|unknown|>",
    special_tokens: list = None,
    path: str = "",
    name: str = "custom_wiki_bpe_32k.json",
):
    """ """
    # 0: Initialise
    # assertions
    assert isinstance(d_articles, dict), "d_articles must be a dictionary of articles"
    assert all(
        isinstance(v, str) for v in d_articles.values()
    ), "All values in d_articles must be strings of sentences"
    assert isinstance(vocab_size, int), "vocab_size must be an integer"
    assert isinstance(unknown_token, str), "unknown_token must be a string"
    assert special_tokens is None or isinstance(
        special_tokens, list
    ), "special_tokens must be a list of strings or None"
    assert isinstance(path, str), "path must be a string"
    assert isinstance(name, str), "name must be a string"
    # prepare special tokens
    if special_tokens is None:
        special_tokens = ["<|pad|>", "<|endoftext|>", unknown_token]
    else:
        special_tokens = list({*special_tokens, unknown_token})
    # prepare path
    path = os.path.abspath(os.path.join(path))

    # 1: Get data and instantiate tokeniser
    tokenizer = Tokenizer(BPE(unk_token=unknown_token))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.decoder = ByteLevelDecoder()

    # 1: Define trainer object and train
    # trainer object with special tokens and desired vocab size
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
    )
    # train directly from in-memory iterator over all article texts
    tokenizer.train_from_iterator(
        d_articles.values(),
        trainer=trainer,
    )

    # Optional: add BOS/EOS around sequences (useful for causal LM training)
    # tokenizer.post_processor = TemplateProcessing(
    #     single="<bos> $A <eos>",
    #     pair="<bos> $A <eos> $B <eos>",
    #     special_tokens=[
    #         ("<bos>", tokenizer.token_to_id("<bos>")),
    #         ("<eos>", tokenizer.token_to_id("<eos>")),
    #     ],
    # )

    # Save
    tokenizer.save(f"{path}/{name}.json")


def load_custom_tokeniser(name: str = "custom_wiki_bpe_32k", path: str = ""):
    # get absolute path
    path = os.path.abspath(os.path.join(path))
    load = os.path.join(path, name)
    return Tokenizer.from_file(f"{load}.json")


if __name__ == "__main__2":
    d_articles, _ = dc.dc_wiki(n_samples=None, list_sentences=False)

    # train custom tokeniser
    train_custom_tokeniser(d_articles, name="custom_wiki_bpe_32k", path="")

    # load custom tokeniser
    tokeniser = load_custom_tokeniser(name="custom_wiki_bpe_32k", path="")
    # ----- Quick sanity check -----
    text = "World Heritage Sites are places in the world which are very important."
    enc = tokeniser.encode(text)
    print("Tokens:", enc.tokens[:30])
    print("Num tokens:", len(enc.ids))
    print(
        "padding id:",
        tokeniser.token_to_id("<|pad|>"),
        "end of text id:",
        tokeniser.token_to_id("<|endoftext|>"),
    )
