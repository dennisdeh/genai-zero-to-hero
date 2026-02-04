"""
In this script we will take a look at the first step in the general workflow of training a language model:
1. Data collection and preparation

We will process a dataset of Wikipedia articles and clean them up, i.e. remove duplicates, special characters, etc.
to prepare them for training a (small) large language model.

The dataset will then be tokenised and split into smaller segments for training and validation

See also the imported modules for more details on the data collection and processing steps.
"""

from p07_llms.c00_gpt_like_models.s01_minigpt.data_collection.data_collection import (
    dc_wiki,
)
from p07_llms.c00_gpt_like_models.s01_minigpt.tokenisation.tokeniser import (
    train_custom_tokeniser,
    load_custom_tokeniser,
)

# 1a: load Wikipedia dataset
d_articles, _ = dc_wiki(n_samples=None, list_sentences=False)

# 1b: train custom tokeniser
path_custom_tokeniser = "p07_llms/c00_gpt_like_models/s01_minigpt/trained_tokenisers"
train_custom_tokeniser(
    d_articles,
    name="custom_wiki_bpe_32k",
    path=path_custom_tokeniser,
)

# check and load the created custom tokeniser
tokeniser = load_custom_tokeniser(
    name="custom_wiki_bpe_32k", path=path_custom_tokeniser
)

# encode and decode an example sentence
text = "The Swiss banks are known for"
enc = tokeniser.encode(text)
print("Encoded:", enc.ids)
print("Tokens used:", enc.tokens[:30])
print("Num tokens:", len(enc.ids))
print(
    "padding id:",
    tokeniser.token_to_id("<|pad|>"),
    "start of text id:",
    tokeniser.token_to_id("<|startoftext|>"),
    "separator id:",
    tokeniser.token_to_id("<|separation|>"),
    "end of text id:",
    tokeniser.token_to_id("<|endoftext|>"),
    "unknown id:",
    tokeniser.token_to_id("<|unknown|>"),
)
