"""
"Needle in a haystack" test of the local model's long-context retrieval.

A unique fact (the needle: a secret passphrase) is inserted at different relative depths (0% = start,
100% = end) into irrelevant filler text (the haystack) of different lengths. The model is asked for the
passphrase; we record whether the answer contains it. The result is a grid "context length x depth" that
is printed and saved as a CSV file and a heatmap in outputs/.

Important Ollama detail: the context window is set per request with `num_ctx`. If the prompt is longer than
num_ctx, Ollama silently truncates it - the model never sees part of the document. We therefore set num_ctx
from the estimated prompt length, and finally run one request WITHOUT num_ctx to show the effect: compare the
number of input tokens reported by Ollama (usage_metadata) with our estimate.

How to run (from the repo root, with Ollama running):
    python -m p07_llms.c05_long_context_systems.main_2_needle_in_haystack
Environment variables (optional): OLLAMA_BASE_URL, OLLAMA_MODEL, NIAH_CONTEXT_LENGTHS (e.g. "1000,4000,8000").
Runtime: 20 requests; the 16k-token prompts take from a few seconds (GPU) to minutes (CPU) each.
"""

import csv
import os
import time

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from p07_llms.c02_prompt_engineering_and_structured_outputs.utils.llm import (
    get_chat_model,
    strip_think,
)
from p07_llms.c05_long_context_systems.utils.texts import (
    insert_needle,
    make_haystack,
    make_needle,
)
from p07_llms.c05_long_context_systems.utils.tokens import count_tokens

CONTEXT_LENGTHS = [
    int(n) for n in os.getenv("NIAH_CONTEXT_LENGTHS", "1000,4000,8000,16000").split(",")
]
DEPTHS = [0.0, 0.25, 0.5, 0.75, 1.0]
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
SYSTEM_PROMPT = "You answer questions about the document provided by the user, using only that document."


def num_ctx_for(prompt_tokens: int, output_tokens: int = 256) -> int:
    """
    Context window to request: estimated prompt + output tokens + 25% headroom (our token count is only an
    estimate for non-OpenAI tokenisers), rounded up to a multiple of 1024.
    """
    needed = int((prompt_tokens + output_tokens) * 1.25)
    return ((needed + 1023) // 1024) * 1024


def build_messages(context_tokens: int, depth: float, seed: int) -> tuple[list, str]:
    """
    Build the chat messages for one trial and return them with the expected passphrase.
    """
    haystack = make_haystack(context_tokens, seed=seed)
    needle, question, answer = make_needle(seed)
    document = insert_needle(haystack, needle, depth)
    messages = [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(f"<document>\n{document}\n</document>\n\nQuestion: {question}"),
    ]
    return messages, answer


def run_trial(llm: BaseChatModel, messages: list, answer: str) -> dict:
    """
    Ask the model, check the answer and collect token statistics.
    """
    start = time.perf_counter()
    response = llm.invoke(messages)
    elapsed = time.perf_counter() - start
    text = strip_think(str(response.content))
    usage = response.usage_metadata or {}
    return {
        "found": answer.lower() in text.lower(),
        "answer": text[:80],
        "input_tokens_reported": usage.get("input_tokens"),
        "seconds": round(elapsed, 2),
    }


def save_results(rows: list[dict]) -> None:
    """
    Save the raw results as CSV and a heatmap (if matplotlib is installed) in outputs/.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "needle_in_haystack.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {csv_path}")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    grid = [
        [
            next(
                r["found"] for r in rows if r["context_tokens"] == n and r["depth"] == d
            )
            for d in DEPTHS
        ]
        for n in CONTEXT_LENGTHS
    ]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(DEPTHS)), [f"{d:.0%}" for d in DEPTHS])
    ax.set_yticks(range(len(CONTEXT_LENGTHS)), [f"{n:,}" for n in CONTEXT_LENGTHS])
    ax.set_xlabel("needle depth in the document")
    ax.set_ylabel("context length (tokens)")
    ax.set_title("Needle in a haystack: green = found")
    fig.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, "needle_in_haystack.png")
    fig.savefig(png_path, dpi=120)
    print(f"Saved {png_path}")


if __name__ == "__main__":
    # 1. Run the grid of trials: one model object per context length, with a matching num_ctx
    rows = []
    for context_tokens in CONTEXT_LENGTHS:
        prompt_tokens = count_tokens(
            build_messages(context_tokens, 0.5, seed=0)[0][1].content
        )
        num_ctx = num_ctx_for(prompt_tokens)
        llm = get_chat_model(temperature=0, num_ctx=num_ctx, num_predict=256)
        print(f"\nContext ~{prompt_tokens:,} tokens (num_ctx={num_ctx}):")
        for i, depth in enumerate(DEPTHS):
            seed = context_tokens + i
            messages, answer = build_messages(context_tokens, depth, seed)
            result = run_trial(llm, messages, answer)
            rows.append(
                {
                    "context_tokens": context_tokens,
                    "depth": depth,
                    "num_ctx": num_ctx,
                    "estimated_tokens": prompt_tokens,
                    **result,
                }
            )
            print(
                f"  depth {depth:>4.0%}: {'FOUND ' if result['found'] else 'MISSED'} "
                f"(expected {answer}, got {result['answer']!r}, "
                f"input tokens reported {result['input_tokens_reported']}, {result['seconds']}s)"
            )

    # 2. Summary grid
    print("\nSummary (x = found, . = missed)")
    print("context  " + " ".join(f"{d:>5.0%}" for d in DEPTHS))
    for n in CONTEXT_LENGTHS:
        marks = [
            next(
                r["found"] for r in rows if r["context_tokens"] == n and r["depth"] == d
            )
            for d in DEPTHS
        ]
        print(f"{n:>7,}  " + " ".join(f"{'x' if m else '.':>5}" for m in marks))
    save_results(rows)

    # 3. The num_ctx pitfall: the same long prompt with Ollama's default context window
    longest = max(CONTEXT_LENGTHS)
    messages, answer = build_messages(longest, 0.0, seed=12345)
    llm_default = get_chat_model(temperature=0, num_predict=256)
    result = run_trial(llm_default, messages, answer)
    print(
        f"\nWithout num_ctx (server default), needle at the start of a ~{longest:,}-token document:"
    )
    print(
        f"  {'FOUND' if result['found'] else 'MISSED'}; input tokens reported by the server: "
        f"{result['input_tokens_reported']} (estimate {count_tokens(messages[1].content):,})"
    )
    print(
        "  If the reported number is much smaller than the estimate, the prompt was truncated."
    )
