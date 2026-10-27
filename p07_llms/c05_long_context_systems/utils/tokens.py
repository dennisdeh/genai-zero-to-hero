"""
Token counting, chunking and memory estimates for long-context examples.

tiktoken provides the exact tokenisers of OpenAI models (cl100k_base for GPT-4/GPT-3.5, o200k_base for
GPT-4o and newer). Other models (Qwen, Llama, ...) have their own tokenisers, so tiktoken counts are only an
estimate for them - usually within +-10-20% for English text. tiktoken downloads its vocabulary files on first
use; when that is not possible (offline) we fall back to the rule of thumb "1 token ~ 4 characters".
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Union

from langchain_core.messages import BaseMessage


@lru_cache(maxsize=4)
def _get_encoding(name: str):
    """
    Load a tiktoken encoding, or return None if tiktoken/the vocabulary file is unavailable.
    """
    try:
        import tiktoken

        return tiktoken.get_encoding(name)
    except (
        Exception
    ) as e:  # no network for the vocabulary download, or tiktoken not installed
        print(
            f"Note: tiktoken encoding '{name}' unavailable ({type(e).__name__}); using ~4 characters per token."
        )
        return None


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Count tokens with tiktoken, or approximate with len(text) / 4 when tiktoken is unavailable.
    """
    encoding = _get_encoding(encoding_name)
    if encoding is None:
        return max(1, round(len(text) / 4))
    return len(encoding.encode(text))


def tiktoken_available(encoding_name: str = "cl100k_base") -> bool:
    """
    True if exact tiktoken counting is possible.
    """
    return _get_encoding(encoding_name) is not None


def message_token_counter(messages: list[BaseMessage]) -> int:
    """
    Token counter for chat messages, usable as `token_counter` in `trim_messages`.
    Adds ~4 tokens per message for role markers and separators of the chat template.
    """
    return sum(count_tokens(str(m.content)) + 4 for m in messages)


def chunk_by_tokens(
    text: str,
    chunk_tokens: int,
    overlap_tokens: int = 0,
    counter: Callable[[str], int] = count_tokens,
) -> list[str]:
    """
    Split text into chunks of about `chunk_tokens` tokens (a sliding window over sentences).

    Splitting happens at sentence boundaries ('. '), so chunks never cut a sentence in half. Each chunk is
    measured as the joined text (including the joining spaces), so it stays within `chunk_tokens` - except a
    single sentence that is longer than the budget on its own, which becomes its own (oversized) chunk. With
    `overlap_tokens > 0`, the last sentences of a chunk are repeated at the start of the next one (about
    `overlap_tokens` tokens, estimated from per-sentence counts), so that facts spanning a chunk boundary are
    not lost.
    """
    sentences = [
        s.strip() + "." for s in text.replace("\n", " ").split(". ") if s.strip()
    ]
    lengths = [counter(s) for s in sentences]
    chunks: list[str] = []
    start = 0
    while start < len(sentences):
        end = start + 1  # a chunk always holds at least one sentence
        # grow the chunk while the joined text (spaces included) still fits; the per-sentence sum is a cheap
        # lower bound that avoids re-counting once the budget is clearly exceeded
        while end < len(sentences) and (
            sum(lengths[start : end + 1]) <= chunk_tokens
            and counter(" ".join(sentences[start : end + 1])) <= chunk_tokens
        ):
            end += 1
        chunks.append(" ".join(sentences[start:end]))
        if end >= len(sentences):
            break
        # step back so that roughly overlap_tokens tokens are repeated in the next chunk
        new_start, back = end, 0
        while new_start - 1 > start and back + lengths[new_start - 1] <= overlap_tokens:
            new_start -= 1
            back += lengths[new_start]
        start = new_start
    return chunks


def truncate_tokens(text: str, max_tokens: int, keep: str = "head") -> str:
    """
    Truncate text to about max_tokens tokens.

    keep="head": keep the beginning; "tail": keep the end; "middle_out": keep beginning and end and drop the
    middle (useful because models attend best to the start and the end of the context).
    """
    if count_tokens(text) <= max_tokens:
        return text
    words = text.split(" ")

    # binary search on the number of words that fits the budget
    def fits(n_words: int) -> bool:
        if keep == "head":
            candidate = " ".join(words[:n_words])
        elif keep == "tail":
            candidate = " ".join(words[-n_words:]) if n_words else ""
        else:
            half = n_words // 2
            candidate = " ".join(
                words[:half] + ["[...]"] + words[len(words) - (n_words - half) :]
            )
        return count_tokens(candidate) <= max_tokens

    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if fits(mid) else (lo, mid - 1)
    if keep == "head":
        return " ".join(words[:lo])
    if keep == "tail":
        return " ".join(words[-lo:]) if lo else ""
    half = lo // 2
    return " ".join(words[:half] + ["[...]"] + words[len(words) - (lo - half) :])


@dataclass
class ModelShape:
    """Architecture numbers needed for a KV cache estimate (from the model's config.json)."""

    name: str
    n_layers: int
    n_kv_heads: int
    head_dim: int


def kv_cache_bytes(
    shape: ModelShape, n_tokens: int, bytes_per_value: Union[int, float] = 2
) -> float:
    """
    Memory of the key/value cache: 2 (K and V) x layers x kv_heads x head_dim x tokens x bytes per value.
    bytes_per_value=2 for fp16/bf16, 1 for an 8-bit quantised cache.
    """
    return (
        2
        * shape.n_layers
        * shape.n_kv_heads
        * shape.head_dim
        * n_tokens
        * bytes_per_value
    )
