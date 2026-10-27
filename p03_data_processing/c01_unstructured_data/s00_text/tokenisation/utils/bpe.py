"""
Byte-level byte-pair encoding (BPE) tokeniser implemented from scratch (educational).

The algorithm follows the GPT-2/GPT-4 family of tokenisers (see also A. Karpathy's minbpe):

1. Pre-tokenisation: a regular expression splits the text into chunks (words with their leading space,
   numbers, punctuation, whitespace). Merges never cross chunk boundaries, so e.g. "dog." and "dog!"
   share the token " dog".
2. Each chunk is converted to its UTF-8 bytes, so the base vocabulary has exactly 256 symbols and any
   text (any language, emojis, code) can be encoded - there is no "unknown" token.
3. Training: repeatedly count all adjacent symbol pairs and merge the most frequent pair into a new
   symbol with the next free id, until the vocabulary has the requested size.
4. Encoding: apply the learned merges to new text in the order they were learned (lowest id first).
5. Decoding: concatenate the bytes of the tokens and decode them as UTF-8.

The input must be valid Unicode text: a lone surrogate (e.g. U+D800, which can come from broken JSON
or file names) has no UTF-8 encoding and raises a ``UnicodeEncodeError`` - like in tiktoken.

For speed, training works on the *distinct* chunks weighted by their frequency rather than on the raw
text. The implementation is plain Python and meant for small corpora; production tokenisers (HF
``tokenizers``, ``tiktoken``) implement the same idea in Rust.
"""

import json
from collections import Counter

try:  # the `regex` module supports unicode classes such as \p{L}; it is a dependency of tiktoken
    import regex as re

    # pre-tokenisation pattern of OpenAI's cl100k_base (GPT-4) tokeniser, exactly as in tiktoken
    SPLIT_PATTERN = (
        r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|"""
        r"""\s+(?!\S)|\s"""
    )
except ImportError:  # simplified fallback with the standard library (not identical to cl100k_base)
    import re

    SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d{1,3}| ?[^\s\w]+|\s+(?!\S)|\s+"""


def get_pair_counts(ids: list[int], counts: Counter | None = None, weight: int = 1) -> Counter:
    """Count adjacent pairs in a list of ids, e.g. [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}."""
    counts = Counter() if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] += weight
    return counts


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of ``pair`` in ``ids`` by ``new_id``, e.g. [1, 2, 3, 1, 2], (1, 2), 9 -> [9, 3, 9]."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokeniser:
    """A minimal byte-level BPE tokeniser with regex pre-tokenisation and special tokens."""

    def __init__(self, pattern: str = SPLIT_PATTERN, special_tokens: list[str] | None = None):
        self.pattern = pattern
        self.compiled_pattern = re.compile(pattern)
        self.merges: dict[tuple[int, int], int] = {}  # (id, id) -> merged id, in training order
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_tokens: dict[str, int] = {}
        self._special_names = special_tokens or ["<|endoftext|>"]

    # ------------------------------------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------------------------------------
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """
        Learn ``vocab_size - 256`` merges from ``text``.

        :param text: training corpus
        :param vocab_size: size of the final vocabulary (without special tokens), >= 256
        :param verbose: print every merge
        """
        assert vocab_size >= 256, "the vocabulary must at least contain the 256 byte values"
        # start from the 256 byte values (calling train twice must not mix two sets of merges)
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        chunk_counts = Counter(self.compiled_pattern.findall(text))
        chunks = [(list(chunk.encode("utf-8")), freq) for chunk, freq in chunk_counts.items()]

        for new_id in range(256, vocab_size):
            pair_counts = Counter()
            for ids, freq in chunks:
                get_pair_counts(ids, pair_counts, weight=freq)
            if not pair_counts:
                break  # nothing left to merge (tiny corpus)
            best = max(pair_counts, key=pair_counts.get)
            chunks = [(merge(ids, best, new_id), freq) for ids, freq in chunks]
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            if verbose:
                print(f"merge {new_id - 255:4d}: {self.vocab[best[0]]!r} + {self.vocab[best[1]]!r} "
                      f"-> {self.vocab[new_id]!r} (count {pair_counts[best]})")

        # special tokens get the ids after the learned vocabulary
        self.special_tokens = {name: len(self.vocab) + i for i, name in enumerate(self._special_names)}

    # ------------------------------------------------------------------------------------------------
    # encoding / decoding
    # ------------------------------------------------------------------------------------------------
    def _encode_chunk(self, chunk: bytes) -> list[int]:
        """Apply the merges to one pre-token: always merge the pair that was learned first."""
        ids = list(chunk)
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text, treating special token strings as ordinary text."""
        ids = []
        for chunk in self.compiled_pattern.findall(text):
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def encode(self, text: str, allowed_special: str | set[str] = "none") -> list[int]:
        """
        Encode text, mapping the *allowed* special tokens (e.g. <|endoftext|>) to their ids.

        Special token strings in untrusted text (user input, scraped documents) must not silently become
        control tokens - a user could otherwise end a document or inject a role marker ("special token
        injection"). Like tiktoken, the default therefore raises an error if the text contains a special
        token string; pass ``allowed_special="all"`` (or a set of names) for trusted text, or use
        ``encode_ordinary`` to treat the strings as ordinary text.

        :param allowed_special: "all", "none" or a set of special token strings to recognise
        """
        allowed = (set(self.special_tokens) if allowed_special == "all"
                   else set() if allowed_special == "none" else set(allowed_special))
        disallowed = [s for s in self.special_tokens if s not in allowed and s in text]
        if disallowed:
            raise ValueError(f"text contains the special token(s) {disallowed}; pass allowed_special to "
                             f"encode them as special tokens, or use encode_ordinary to encode them as text")
        if not allowed:
            return self.encode_ordinary(text)
        special_pattern = "(" + "|".join(re.escape(s) for s in allowed) + ")"
        ids = []
        for part in re.split(special_pattern, text):
            if part in allowed:
                ids.append(self.special_tokens[part])
            elif part:
                ids.extend(self.encode_ordinary(part))
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode ids to text (invalid UTF-8 byte sequences are replaced by U+FFFD)."""
        inverse_special = {v: k.encode("utf-8") for k, v in self.special_tokens.items()}
        data = b"".join(self.vocab[i] if i in self.vocab else inverse_special[i] for i in ids)
        return data.decode("utf-8", errors="replace")

    def token_strings(self, ids: list[int]) -> list[str]:
        """Human-readable token strings (useful to look at how a text is split)."""
        inverse_special = {v: k for k, v in self.special_tokens.items()}
        return [inverse_special.get(i) or self.vocab[i].decode("utf-8", errors="replace") for i in ids]

    # ------------------------------------------------------------------------------------------------
    # persistence and interoperability
    # ------------------------------------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save pattern, merges (in order) and special tokens as JSON."""
        data = {
            "pattern": self.pattern,
            "merges": [[a, b, idx] for (a, b), idx in self.merges.items()],
            "special_tokens": self.special_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "BPETokeniser":
        """Load a tokeniser saved with ``save``."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(pattern=data["pattern"], special_tokens=list(data["special_tokens"]))
        for a, b, idx in data["merges"]:
            tok.merges[(a, b)] = idx
            tok.vocab[idx] = tok.vocab[a] + tok.vocab[b]
        tok.special_tokens = data["special_tokens"]
        return tok

    def mergeable_ranks(self) -> dict[bytes, int]:
        """Vocabulary in tiktoken's format {token bytes: rank}; the rank equals the token id."""
        return {token: idx for idx, token in self.vocab.items()}
