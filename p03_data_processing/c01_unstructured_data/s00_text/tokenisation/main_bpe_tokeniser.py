"""
Byte-pair encoding (BPE): train a tokeniser from scratch and compare it with production tokenisers.

The script uses the small bundled corpus in data/corpus.txt (no downloads needed) and

1. trains the from-scratch byte-level BPE tokeniser of utils/bpe.py and prints the first merges
2. checks that encoding followed by decoding gives back the original text, and shows example splits
3. saves/loads the tokeniser and exports it to tiktoken's fast Rust encoder - both must give identical ids
4. trains a Hugging Face `tokenizers` byte-level BPE tokeniser with the same vocabulary size
5. compares all tokenisers (plus OpenAI's cl100k_base if it can be downloaded) on held-out text:
   number of tokens and compression (bytes per token)
6. shows the trade-off between vocabulary size and sequence length

Run from the repository root:
    python -m p03_data_processing.c01_unstructured_data.s00_text.tokenisation.main_bpe_tokeniser
"""

import os
import time

import pandas as pd
import tiktoken
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from p03_data_processing.c01_unstructured_data.s00_text.tokenisation.utils.bpe import BPETokeniser

PATH_HERE = os.path.dirname(__file__)
PATH_CORPUS = os.path.join(PATH_HERE, "data", "corpus.txt")
PATH_OUTPUTS = os.path.join(PATH_HERE, "outputs")
VOCAB_SIZE = 512
EOT = "<|endoftext|>"
EXAMPLES = [
    "The interest rate on the mortgage is fixed.",
    "Compound growth: 1,000 * 1.03^10 = 1,343.92 EUR",
    "Renten på lånet er fast i tredive år.",
    "def compound(principal, rate, years):",
    "Tokenisation of unseen words: cryptocurrency, decentralisation",
]


def load_corpus(path: str = PATH_CORPUS) -> tuple[str, str]:
    """Load the corpus and split its lines into a training text (90 %) and a held-out text (10 %)."""
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    train = "\n".join(line for i, line in enumerate(lines) if i % 10 != 5)
    held_out = "\n".join(line for i, line in enumerate(lines) if i % 10 == 5)
    return train, held_out


def train_hf_bpe(text: str, vocab_size: int) -> Tokenizer:
    """Train a byte-level BPE tokeniser with the Hugging Face `tokenizers` library (Rust implementation)."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size + 1,  # + 1 for the special token, to match our tokeniser
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(text.splitlines(keepends=True), trainer)
    return tok


def to_tiktoken(tok: BPETokeniser, name: str = "scratch_bpe") -> tiktoken.Encoding:
    """Wrap our learned merges in tiktoken's (fast) Encoding class."""
    return tiktoken.Encoding(name=name, pat_str=tok.pattern, mergeable_ranks=tok.mergeable_ranks(),
                             special_tokens=tok.special_tokens)


def load_cl100k() -> tiktoken.Encoding | None:
    """OpenAI's GPT-4 tokeniser; the vocabulary file is downloaded once and cached (needs internet)."""
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception as e:  # no internet / blocked download
        print(f"(cl100k_base not available - skipped in the comparison: {type(e).__name__})")
        return None


def compression_table(text: str, encoders: dict) -> pd.DataFrame:
    """Number of tokens and bytes per token of each encoder on ``text``."""
    n_bytes = len(text.encode("utf-8"))
    rows = []
    for name, encode in encoders.items():
        n_tokens = len(encode(text))
        rows.append({"tokeniser": name, "tokens": n_tokens, "bytes_per_token": round(n_bytes / n_tokens, 2)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # 1. Train the from-scratch tokeniser
    train_text, held_out = load_corpus()
    print(f"training text: {len(train_text):,} characters, held-out text: {len(held_out):,} characters")
    tok = BPETokeniser(special_tokens=[EOT])
    t0 = time.perf_counter()
    tok.train(train_text, vocab_size=VOCAB_SIZE)
    print(f"trained {len(tok.merges)} merges in {time.perf_counter() - t0:.1f} s; first 15 merges:")
    for (a, b), idx in list(tok.merges.items())[:15]:
        print(f"  {idx}: {tok.vocab[a]!r} + {tok.vocab[b]!r} -> {tok.vocab[idx]!r}")
    longest = sorted(tok.vocab.values(), key=len, reverse=True)[:12]
    print("longest learned tokens:", [t.decode("utf-8", errors="replace") for t in longest])

    # 2. Round trip and example splits
    for text in [held_out, *EXAMPLES]:
        assert tok.decode(tok.encode(text)) == text, "round trip failed"
    print("\nround trip encode -> decode: OK")
    for text in EXAMPLES:
        ids = tok.encode(text)
        print(f"{len(ids):3d} tokens | {' | '.join(tok.token_strings(ids))}")
    ids = tok.encode(f"First document.{EOT}Second document.")
    print(f"special token handling: {tok.token_strings(ids)}")

    # 3. Persistence and export to tiktoken (identical ids, but much faster)
    os.makedirs(PATH_OUTPUTS, exist_ok=True)
    path_tok = os.path.join(PATH_OUTPUTS, "scratch_bpe.json")
    tok.save(path_tok)
    tok_loaded = BPETokeniser.load(path_tok)
    assert tok_loaded.encode(held_out) == tok.encode(held_out)
    enc_fast = to_tiktoken(tok)
    same = enc_fast.encode(held_out, allowed_special="all") == tok.encode(held_out)
    print(f"\nsaved to {path_tok}; reloaded tokeniser gives identical ids")
    print(f"tiktoken encoder built from our merges gives identical ids: {same}")
    t0 = time.perf_counter()
    for _ in range(20):
        tok.encode(train_text)
    t_py = (time.perf_counter() - t0) / 20
    t0 = time.perf_counter()
    for _ in range(20):
        enc_fast.encode_ordinary(train_text)
    t_rs = (time.perf_counter() - t0) / 20
    print(f"encoding the training text: pure Python {1000 * t_py:.1f} ms vs. tiktoken {1000 * t_rs:.2f} ms")

    # 4. Hugging Face tokenizers (same algorithm, different pre-tokenisation and tie-breaking)
    hf_tok = train_hf_bpe(train_text, VOCAB_SIZE)
    print(f"\nHF tokenizers vocabulary size: {hf_tok.get_vocab_size()}")
    for text in EXAMPLES[:3]:
        print(f"  HF: {hf_tok.encode(text).tokens}")
    assert hf_tok.decode(hf_tok.encode(held_out).ids) == held_out

    # 5. Compare on held-out text
    encoders = {
        f"scratch BPE ({VOCAB_SIZE})": tok.encode,
        f"HF tokenizers BPE ({VOCAB_SIZE})": lambda t: hf_tok.encode(t).ids,
        "raw UTF-8 bytes (256)": lambda t: list(t.encode("utf-8")),
    }
    cl100k = load_cl100k()
    if cl100k is not None:
        encoders["tiktoken cl100k_base (100k)"] = cl100k.encode_ordinary
        print("cl100k_base split:", [cl100k.decode([i]) for i in cl100k.encode_ordinary(EXAMPLES[1])])
    print("\ncompression on held-out text:")
    print(compression_table(held_out, encoders).to_string(index=False))

    # 6. Vocabulary size vs. sequence length
    print("\nvocabulary size vs. compression (held-out text):")
    for vocab_size in [256, 300, 400, 512, 768]:
        t = BPETokeniser()
        t.train(train_text, vocab_size)
        n = len(t.encode(held_out))
        print(f"  vocab {vocab_size:4d}: {n:5d} tokens, {len(held_out.encode('utf-8')) / n:.2f} bytes/token")
