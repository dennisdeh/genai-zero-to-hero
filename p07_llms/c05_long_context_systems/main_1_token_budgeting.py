"""
Token counting, context budgeting and the cost of long contexts (runs offline, no LLM needed).

1. Count tokens with tiktoken for different kinds of text (English, Danish, numbers, code, JSON) and
   compare with the "4 characters per token" rule of thumb.
2. Plan a context budget: context window = system prompt + history + retrieved documents + question +
   reserved output tokens. How many document chunks fit?
3. Estimate the KV cache memory and relative attention cost for growing context lengths - the reason why
   "just use a longer context" is not free.

How to run (from the repo root):
    python -m p07_llms.c05_long_context_systems.main_1_token_budgeting
Note: tiktoken downloads its vocabulary on first use; offline, the script falls back to ~4 chars/token.
"""

from p07_llms.c05_long_context_systems.utils.texts import make_long_report
from p07_llms.c05_long_context_systems.utils.tokens import (
    ModelShape,
    count_tokens,
    kv_cache_bytes,
    tiktoken_available,
)

SAMPLES = {
    "English prose": "The central bank raised its policy rate by a quarter of a percentage point to fight inflation.",
    "Danish prose": "Nationalbanken hævede sin udlånsrente med et kvart procentpoint for at bekæmpe inflationen.",
    "Numbers": "2024-12-31; 1,234,567.89; 0.000314; 98765432; 3.14159265358979",
    "Python code": "def f(x: list[int]) -> int:\n    return sum(v * v for v in x if v % 2 == 0)",
    "JSON": '{"ticker": "NWIND", "revenue": 4200, "currency": "DKK", "guidance": [16000, 17000]}',
}

# Architecture numbers from the models' config.json files (approximate, for illustration)
MODEL_SHAPES = [
    ModelShape(
        "Qwen3-8B (36 layers, 8 KV heads, GQA)", n_layers=36, n_kv_heads=8, head_dim=128
    ),
    ModelShape(
        "Llama-3.1-8B (32 layers, 8 KV heads, GQA)",
        n_layers=32,
        n_kv_heads=8,
        head_dim=128,
    ),
    ModelShape(
        "8B model without GQA (32 layers, 32 KV heads)",
        n_layers=32,
        n_kv_heads=32,
        head_dim=128,
    ),
]


def compare_tokenisers() -> None:
    """
    Print token counts for different encodings and the chars/4 approximation.
    """
    print(
        f"{'Text type':<15} {'chars':>6} {'cl100k':>7} {'o200k':>6} {'chars/4':>8}  chars/token"
    )
    for name, text in SAMPLES.items():
        n_cl100k = count_tokens(text, "cl100k_base")
        n_o200k = count_tokens(text, "o200k_base")
        print(
            f"{name:<15} {len(text):>6} {n_cl100k:>7} {n_o200k:>6} {round(len(text) / 4):>8}  "
            f"{len(text) / n_cl100k:.1f}"
        )


def plan_budget(
    context_window: int,
    system_tokens: int,
    history_tokens: int,
    question_tokens: int,
    reserved_output: int,
    chunk_tokens: int,
) -> int:
    """
    Return how many retrieved chunks of chunk_tokens fit into the remaining context budget, and print the plan.
    """
    used = system_tokens + history_tokens + question_tokens + reserved_output
    remaining = context_window - used
    n_chunks = max(0, remaining // chunk_tokens)
    print(f"  context window          {context_window:>7}")
    print(f"  - system prompt         {system_tokens:>7}")
    print(f"  - chat history          {history_tokens:>7}")
    print(f"  - question              {question_tokens:>7}")
    print(f"  - reserved for output   {reserved_output:>7}")
    print(
        f"  = left for documents    {remaining:>7}  -> {n_chunks} chunks of {chunk_tokens} tokens"
    )
    return n_chunks


if __name__ == "__main__":
    # 1. Tokenisation differs between text types and tokenisers
    print("===== 1. Token counts =====")
    if not all([tiktoken_available("cl100k_base"), tiktoken_available("o200k_base")]):
        print(
            "(tiktoken vocabulary not available - the counts below are approximations)"
        )
    compare_tokenisers()
    print(
        "Non-English text, numbers and code need more tokens per character - budgets based on characters or "
        "words are unreliable. Each model family has its own tokeniser; tiktoken is exact only for OpenAI models."
    )

    # 2. Budget planning for a RAG/chat request
    print("\n===== 2. Context budget =====")
    report, _ = make_long_report()
    report_tokens = count_tokens(report)
    print(
        f"Our synthetic annual report has {len(report):,} characters ~ {report_tokens:,} tokens."
    )
    budget = dict(
        system_tokens=300, history_tokens=1500, question_tokens=50, reserved_output=1000
    )
    for window in [4096, 32768]:
        print(f"\nModel with a {window}-token window:")
        plan_budget(window, chunk_tokens=500, **budget)
        fits = report_tokens + sum(budget.values()) <= window
        print(
            f"  The whole report {'fits' if fits else 'does NOT fit'} in this window."
        )

    # 3. The price of long contexts: KV cache memory grows linearly, attention compute quadratically
    print("\n===== 3. KV cache memory (fp16) and relative attention cost =====")
    lengths = [4_096, 32_768, 131_072]
    header = f"{'model':<48}" + "".join(f"{n:>12,}" for n in lengths)
    print(header + "   tokens")
    for shape in MODEL_SHAPES:
        row = "".join(f"{kv_cache_bytes(shape, n) / 2**30:>9.1f}GiB" for n in lengths)
        per_token_kib = kv_cache_bytes(shape, 1) / 1024
        print(f"{shape.name:<48}{row}   ({per_token_kib:.0f} KiB/token)")
    base = lengths[0]
    print(
        f"{'relative attention FLOPs (~n^2)':<48}"
        + "".join(f"{(n / base) ** 2:>12.0f}" for n in lengths)
    )
    print(
        "\nThe KV cache is per sequence: serving 8 concurrent 32k-token requests to Qwen3-8B needs ~"
        f"{8 * kv_cache_bytes(MODEL_SHAPES[0], 32_768) / 2**30:.0f} GiB on top of ~15 GiB of fp16 weights. "
        "This is why Ollama sizes its default num_ctx to the available VRAM and why grouped-query attention "
        "(GQA) matters."
    )
