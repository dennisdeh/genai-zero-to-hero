"""
Strategies for documents that are longer than the context window: truncation, map-reduce and refine.

We summarise a long synthetic annual report (~8k tokens, utils/texts.py) whose 12 sections each contain one
key figure buried in filler text. A model with a small *document budget* (DOC_BUDGET tokens per call,
simulating a small context window) has to cope with it using
  1. truncation (keep the head)                      - 1 call, loses everything after the cut,
  2. truncation (keep head and tail, drop the middle) - 1 call,
  3. map-reduce: summarise overlapping chunks in parallel (map), then combine the summaries (reduce,
     recursively if they are still too long),
  4. refine: walk through the chunks sequentially and update a running summary,
  5. stuff: the whole document in one call with a large num_ctx (baseline, if the model supports it).
For each strategy we report the fact recall (share of the 12 key figures that survive in the summary), the
number of LLM calls and the run time.

How to run (from the repo root, with Ollama running):
    python -m p07_llms.c05_long_context_systems.main_3_long_document_strategies
"""

import time
from typing import Callable

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from p07_llms.c02_prompt_engineering_and_structured_outputs.utils.llm import (
    get_chat_model,
    strip_think,
)
from p07_llms.c05_long_context_systems.utils.texts import make_long_report
from p07_llms.c05_long_context_systems.utils.tokens import (
    chunk_by_tokens,
    count_tokens,
    truncate_tokens,
)

DOC_BUDGET = 2_000  # tokens of document text per call
CHUNK_OVERLAP = 100  # tokens repeated between consecutive chunks (sliding window)
SMALL_NUM_CTX = 4_096  # context window used for the budgeted strategies
LARGE_NUM_CTX = 16_384  # context window used for the "stuff everything" baseline

INSTRUCTIONS = (
    "Write a concise summary as bullet points. Keep every key figure (numbers with their units) exactly as "
    "written; ignore generic market commentary."
)
SUMMARISE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", INSTRUCTIONS),
        ("human", "Summarise this part of an annual report:\n\n{text}"),
    ]
)
COMBINE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", INSTRUCTIONS),
        ("human", "Combine these partial summaries into one summary:\n\n{text}"),
    ]
)
REFINE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", INSTRUCTIONS),
        (
            "human",
            "Here is the summary so far:\n{summary}\n\nUpdate it with the new information below. Keep all "
            "figures from the existing summary.\n\nNew text:\n{text}",
        ),
    ]
)


class CallCounter(BaseCallbackHandler):
    """Counts chat model calls (a LangChain callback is invoked at the start of every LLM call)."""

    def __init__(self) -> None:
        self.calls = 0

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        self.calls += 1


def summarise_truncated(llm: BaseChatModel, report: str, keep: str) -> str:
    """
    Strategies 1 and 2: cut the document to the budget, then summarise in a single call.
    """
    text = truncate_tokens(report, DOC_BUDGET, keep=keep)
    return (SUMMARISE_PROMPT | llm | StrOutputParser()).invoke({"text": text})


def summarise_map_reduce(llm: BaseChatModel, report: str) -> str:
    """
    Strategy 3: map (summarise chunks in parallel), then reduce (combine, recursively if too long).
    """
    chunks = chunk_by_tokens(report, DOC_BUDGET, overlap_tokens=CHUNK_OVERLAP)
    summarise = SUMMARISE_PROMPT | llm | StrOutputParser()
    combine = COMBINE_PROMPT | llm | StrOutputParser()

    summaries = [strip_think(s) for s in summarise.batch([{"text": c} for c in chunks])]
    # collapse step: combine pairs of summaries until everything fits the budget
    while len(summaries) > 1 and count_tokens("\n\n".join(summaries)) > DOC_BUDGET:
        groups = [summaries[i : i + 2] for i in range(0, len(summaries), 2)]
        summaries = [
            strip_think(s)
            for s in combine.batch([{"text": "\n\n".join(g)} for g in groups])
        ]
    return combine.invoke({"text": "\n\n".join(summaries)})


def summarise_refine(llm: BaseChatModel, report: str) -> str:
    """
    Strategy 4: sequentially refine a running summary with each chunk.
    """
    chunks = chunk_by_tokens(report, DOC_BUDGET, overlap_tokens=CHUNK_OVERLAP)
    summary = strip_think(
        (SUMMARISE_PROMPT | llm | StrOutputParser()).invoke({"text": chunks[0]})
    )
    refine = REFINE_PROMPT | llm | StrOutputParser()
    for chunk in chunks[1:]:
        summary = strip_think(refine.invoke({"summary": summary, "text": chunk}))
    return summary


def summarise_stuff(llm: BaseChatModel, report: str) -> str:
    """
    Strategy 5: everything in one call (needs a large enough context window).
    """
    return (SUMMARISE_PROMPT | llm | StrOutputParser()).invoke({"text": report})


def fact_recall(summary: str, facts: list[str]) -> float:
    """
    Share of key figures that appear in the summary.
    """
    cleaned = summary.replace(",", "")
    return sum(fact in cleaned for fact in facts) / len(facts)


if __name__ == "__main__":
    # 1. The long document and how it is split
    report, facts = make_long_report()
    chunks = chunk_by_tokens(report, DOC_BUDGET, overlap_tokens=CHUNK_OVERLAP)
    print(
        f"Report: ~{count_tokens(report):,} tokens, {len(facts)} key figures: {', '.join(facts)}"
    )
    print(
        f"Document budget per call: {DOC_BUDGET} tokens -> {len(chunks)} chunks "
        f"(sizes {[count_tokens(c) for c in chunks]}, overlap {CHUNK_OVERLAP})"
    )

    # 2. Run all strategies
    small_llm = get_chat_model(temperature=0, num_ctx=SMALL_NUM_CTX)
    large_llm = get_chat_model(temperature=0, num_ctx=LARGE_NUM_CTX)
    strategies: dict[str, Callable[[BaseChatModel, BaseChatModel], str]] = {
        "truncate (head)": lambda small, large: summarise_truncated(
            small, report, keep="head"
        ),
        "truncate (middle out)": lambda small, large: summarise_truncated(
            small, report, keep="middle_out"
        ),
        "map-reduce": lambda small, large: summarise_map_reduce(small, report),
        "refine": lambda small, large: summarise_refine(small, report),
        "stuff (large num_ctx)": lambda small, large: summarise_stuff(large, report),
    }
    results = []
    for name, run in strategies.items():
        # attach a fresh call counter to copies of both models
        counter = CallCounter()
        small = small_llm.model_copy(update={"callbacks": [counter]})
        large = large_llm.model_copy(update={"callbacks": [counter]})
        start = time.perf_counter()
        summary = strip_think(run(small, large))
        elapsed = time.perf_counter() - start
        recall = fact_recall(summary, facts)
        results.append((name, recall, counter.calls, elapsed))
        print(
            f"\n===== {name}: recall {recall:.0%}, {counter.calls} LLM calls, {elapsed:.1f}s ====="
        )
        print(summary[:1200])

    # 3. Comparison
    print("\nStrategy                 fact recall   LLM calls   seconds")
    for name, recall, calls, elapsed in results:
        print(f"{name:<24} {recall:>10.0%} {calls:>11} {elapsed:>9.1f}")
    print(
        "\nTruncation is cheap but blind to the cut-off part. Map-reduce parallelises well but each chunk is "
        "summarised without global context; refine keeps context but is sequential and can 'forget' early "
        "details as the summary is rewritten. Stuffing works when the window is large enough - at the price "
        "of a large KV cache and the lost-in-the-middle effect."
    )
