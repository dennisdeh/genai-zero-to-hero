"""
Workflow pattern 1: prompt chaining with a gate.

A task is decomposed into a fixed sequence of LLM calls, each working on the output of the previous one.
Between the steps, a programmatic *gate* checks that the intermediate result is on track and stops the
chain early if it is not.

Use case: turn raw facts about a company into a short investment newsletter item in Danish:
    outline (LLM) -> gate (Python checks) -> draft (LLM) -> translate (LLM)
                       \\-> END (gate failed)

How to run (from the repo root, with Ollama running):
    python -m p07_llms.c06_workflows.main_1_prompt_chaining
"""

import re
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from p07_llms.c02_prompt_engineering_and_structured_outputs.utils.llm import (
    get_chat_model,
    strip_think,
)
from p07_llms.c06_workflows.utils.graphs import save_graph

MIN_BULLETS, MAX_BULLETS = 3, 6

FACTS = """
Company: Nordic Wind A/S (Copenhagen), wind turbine service and repair.
Q3 revenue DKK 4.2bn (+18% y/y); operating profit DKK 610m (Q3 last year: DKK 450m).
Order backlog DKK 31bn, record high. Full-year revenue guidance raised to DKK 16-17bn.
Risks: higher steel prices, a pending warranty dispute with a German utility (claim DKK 300m).
New CEO from 1 January: Sofie Lund, previously COO.
"""


class ChainState(TypedDict, total=False):
    facts: str
    outline: str
    gate_passed: bool
    gate_reason: str
    draft: str
    translation: str


def check_outline(outline: str) -> tuple[bool, str]:
    """
    The gate: deterministic checks on the outline (no LLM call needed).
    """
    bullets = [
        line
        for line in outline.splitlines()
        if re.match(r"^\s*([-*•]|\d+[.)])\s+", line)
    ]
    if not MIN_BULLETS <= len(bullets) <= MAX_BULLETS:
        return (
            False,
            f"expected {MIN_BULLETS}-{MAX_BULLETS} bullet points, got {len(bullets)}",
        )
    if not any(re.search(r"\d", b) for b in bullets):
        return False, "the outline contains no figures"
    return True, f"{len(bullets)} bullet points with figures"


def build_graph(llm: BaseChatModel) -> CompiledStateGraph:
    """
    Build the chain outline -> gate -> draft -> translate as a LangGraph StateGraph.
    """

    def outline(state: ChainState) -> ChainState:
        messages = [
            SystemMessage(
                f"Write an outline with {MIN_BULLETS}-{MAX_BULLETS} bullet points ('- ...') for a short "
                "investment newsletter item. Include the key figures. Output only the bullet points."
            ),
            HumanMessage(state["facts"]),
        ]
        return {"outline": strip_think(str(llm.invoke(messages).content))}

    def gate(state: ChainState) -> ChainState:
        passed, reason = check_outline(state["outline"])
        return {"gate_passed": passed, "gate_reason": reason}

    def draft(state: ChainState) -> ChainState:
        messages = [
            SystemMessage(
                "Write a newsletter item of at most 120 words in English from this outline. "
                "Neutral tone, no investment advice."
            ),
            HumanMessage(state["outline"]),
        ]
        return {"draft": strip_think(str(llm.invoke(messages).content))}

    def translate(state: ChainState) -> ChainState:
        messages = [
            SystemMessage(
                "Translate the text into Danish. Keep all figures and names unchanged. "
                "Output only the translation."
            ),
            HumanMessage(state["draft"]),
        ]
        return {"translation": strip_think(str(llm.invoke(messages).content))}

    builder = StateGraph(ChainState)
    builder.add_node("outline", outline)
    builder.add_node("gate", gate)
    builder.add_node("draft", draft)
    builder.add_node("translate", translate)
    builder.add_edge(START, "outline")
    builder.add_edge("outline", "gate")
    builder.add_conditional_edges(
        "gate",
        lambda s: "pass" if s["gate_passed"] else "fail",
        {"pass": "draft", "fail": END},
    )
    builder.add_edge("draft", "translate")
    builder.add_edge("translate", END)
    return builder.compile()


if __name__ == "__main__":
    # 1. Build the workflow and save its structure
    llm = get_chat_model(temperature=0)
    graph = build_graph(llm)
    save_graph(graph, "workflow_1_prompt_chaining")

    # 2. Run it; stream_mode="updates" yields the output of each node as it finishes
    for update in graph.stream({"facts": FACTS}, stream_mode="updates"):
        for node, output in update.items():
            print(f"\n===== {node} =====")
            for key, value in output.items():
                print(f"[{key}]\n{value}\n")
            if node == "gate" and not output["gate_passed"]:
                print(
                    "The gate stopped the chain early - no draft is written from a bad outline."
                )
