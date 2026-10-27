"""
Workflow pattern 5: evaluator-optimiser.

One LLM call generates a response, another evaluates it against explicit criteria and returns structured
feedback; the generator revises its answer using the feedback. The loop stops when the evaluator accepts the
answer or after MAX_ITERATIONS rounds (always bound loops - the evaluator might never be satisfied).
The evaluation combines deterministic checks (length, forbidden phrases) with an LLM judge for the
criteria that need language understanding (tone, concrete next step).

Use case: drafting a reply to an angry customer complaint that must follow the bank's communication policy.

How to run (from the repo root, with Ollama running):
    python -m p07_llms.c06_workflows.main_5_evaluator_optimiser
"""

import operator
import re
from typing import Annotated, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from p07_llms.c02_prompt_engineering_and_structured_outputs.utils.llm import (
    get_chat_model,
    strip_think,
)
from p07_llms.c06_workflows.utils.graphs import save_graph

MAX_ITERATIONS = 3
MAX_WORDS = 120
# Legal says: no promises in the first reply. We match whole *promise* phrases with word boundaries rather than
# single words, so that harmless sentences such as "we cannot guarantee ..." or "compensation is only decided
# after the investigation" do not trigger the check (a negation between subject and verb breaks the match).
FORBIDDEN = {
    "promise of a refund/compensation": r"\b(?:we|i)(?: will| shall|'ll)? (?:refund|reimburse|compensate)\b",
    "promise of a refund/compensation to the customer": r"\byou(?: will|'ll) (?:be (?:refunded|reimbursed|compensated)"
    r"|(?:receive|get) (?:(?:a|an|full|your) ){0,2}(?:refund|reimbursement|compensation))\b",
    "guarantee": r"\b(?:we|i) (?:can )?guarantee\b",
    "promise to pay": r"\b(?:we|i)(?: will| shall|'ll) pay\b",
}

COMPLAINT = (
    "I have been charged an overdraft fee of DKK 250 three months in a row even though my salary arrives "
    "every month on the 1st. Your app is useless and nobody answers the phone. I want this fixed NOW or I "
    "am moving all my money to another bank!"
)

POLICY = f"""Communication policy for complaint replies:
1. Polite, calm and empathetic; acknowledge the specific problem.
2. Contains a concrete next step and a time frame (complaints are investigated within 5 working days).
3. At most {MAX_WORDS} words.
4. Never promise compensation, refunds or guarantees before the investigation."""


class Evaluation(BaseModel):
    grade: Literal["pass", "fail"]
    feedback: str = Field(
        description="If fail: concrete instructions for improving the reply"
    )


class LoopState(TypedDict, total=False):
    complaint: str
    draft: str
    grade: str
    feedback: str
    iterations: int
    history: Annotated[list[dict], operator.add]


def rule_checks(text: str) -> list[str]:
    """
    Deterministic part of the evaluation: cheap, exact and explainable.
    """
    problems = []
    n_words = len(text.split())
    if n_words > MAX_WORDS:
        problems.append(f"The reply has {n_words} words; the maximum is {MAX_WORDS}.")
    normalised = text.lower().replace("\u2019", "'")  # typographic apostrophe -> ASCII
    for name, pattern in FORBIDDEN.items():
        match = re.search(pattern, normalised)
        if match:
            problems.append(
                f"Remove the {name} ('{match.group()}') - no promises before the investigation."
            )
    return problems


def build_graph(llm: BaseChatModel) -> CompiledStateGraph:
    """
    generator <-> evaluator loop with a maximum number of iterations.
    """
    judge = llm.with_structured_output(Evaluation)

    def generator(state: LoopState) -> LoopState:
        messages = [
            SystemMessage(
                f"You write replies to customer complaints for Nordic Bank.\n{POLICY}\n"
                "Output only the reply."
            ),
            HumanMessage(f"Complaint:\n{state['complaint']}"),
        ]
        if state.get("feedback"):
            messages.append(
                HumanMessage(
                    f"Your previous draft:\n{state['draft']}\n\n"
                    f"Reviewer feedback:\n{state['feedback']}\n\nWrite an improved reply."
                )
            )
        draft = strip_think(str(llm.invoke(messages).content))
        return {"draft": draft, "iterations": state.get("iterations", 0) + 1}

    def evaluator(state: LoopState) -> LoopState:
        problems = rule_checks(state["draft"])
        evaluation = judge.invoke(
            [
                SystemMessage(
                    f"You review complaint replies against this policy:\n{POLICY}\n"
                    "Grade 'pass' only if all points are satisfied."
                ),
                HumanMessage(
                    f"Complaint:\n{state['complaint']}\n\nReply to review:\n{state['draft']}"
                ),
            ]
        )
        if evaluation.grade == "fail":
            problems.append(evaluation.feedback)
        grade = "fail" if problems else "pass"
        feedback = "\n".join(f"- {p}" for p in problems)
        return {
            "grade": grade,
            "feedback": feedback,
            "history": [
                {
                    "iteration": state["iterations"],
                    "draft": state["draft"],
                    "grade": grade,
                    "feedback": feedback,
                }
            ],
        }

    def route(state: LoopState) -> str:
        if state["grade"] == "pass" or state["iterations"] >= MAX_ITERATIONS:
            return "done"
        return "revise"

    builder = StateGraph(LoopState)
    builder.add_node("generator", generator)
    builder.add_node("evaluator", evaluator)
    builder.add_edge(START, "generator")
    builder.add_edge("generator", "evaluator")
    builder.add_conditional_edges(
        "evaluator", route, {"revise": "generator", "done": END}
    )
    return builder.compile()


if __name__ == "__main__":
    # 1. Build the workflow and save its structure
    llm = get_chat_model(
        temperature=0.3
    )  # a little creativity helps the generator to actually change text
    graph = build_graph(llm)
    save_graph(graph, "workflow_5_evaluator_optimiser")

    # 2. Run the loop and show every round
    result = graph.invoke({"complaint": COMPLAINT})
    for round_ in result["history"]:
        print("=" * 100)
        print(f"Iteration {round_['iteration']}: {round_['grade'].upper()}")
        print(round_["draft"])
        if round_["feedback"]:
            print(f"\nFeedback:\n{round_['feedback']}")
    status = (
        "accepted"
        if result["grade"] == "pass"
        else f"stopped after {MAX_ITERATIONS} iterations"
    )
    print("=" * 100)
    print(f"Final reply {status}.")
