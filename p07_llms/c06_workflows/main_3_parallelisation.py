"""
Workflow pattern 3: parallelisation - sectioning and voting.

Sectioning: a task is split into independent subtasks that run in parallel, and the results are aggregated
by a final step. Here three analysts look at the same news article from different angles (financial impact,
risks, ESG); static parallel edges fan out from START and `add_edge([...], "aggregate")` fans in, i.e. the
aggregator waits for all three branches.

Voting: the *same* task is run several times (with different prompts/perspectives) and the answers are
combined, e.g. by majority vote - more robust than a single call. Here N reviewers decide whether an e-mail
is a phishing attempt. The number of voters is dynamic, so we fan out with the `Send` API, and the votes are
collected in a list with an `operator.add` reducer (parallel writes to the same key are appended, not
overwritten).

How to run (from the repo root, with Ollama running):
    python -m p07_llms.c06_workflows.main_3_parallelisation
"""

import operator
from collections import Counter
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from p07_llms.c02_prompt_engineering_and_structured_outputs.utils.llm import (
    get_chat_model,
    strip_think,
)
from p07_llms.c06_workflows.utils.graphs import save_graph

ARTICLE = (
    "Kattegat Shipping reported a 35% drop in quarterly profit as container freight rates fell. The company "
    "will retrofit 12 vessels to run on green methanol by 2027, financed by a DKK 2bn green bond. A labour "
    "dispute at its Aarhus terminal is unresolved, and the EU emissions trading system will cover shipping "
    "from next year."
)

EMAILS = [
    "Dear customer, your Nordic Bank account is locked. Confirm your identity within 24 hours at "
    "http://nordic-bank-secure-login.co/verify or your account will be closed.",
    "Hi Dennis, the slides for Thursday's model review are in the shared team folder. Let me know if anything "
    "is missing. Best, Anna",
]

VOTER_PERSPECTIVES = [
    "You are a security analyst. Focus on links, domains and requests for credentials.",
    "You are a fraud investigator. Focus on urgency, threats and pressure tactics.",
    "You are a sceptical colleague. Focus on whether the message is plausible in a normal business context.",
]


# ---------------------------------------------------------------------------
# Sectioning
# ---------------------------------------------------------------------------
class SectioningState(TypedDict, total=False):
    article: str
    financial: str
    risks: str
    esg: str
    report: str


def build_sectioning_graph(llm: BaseChatModel) -> CompiledStateGraph:
    """
    START -> {financial, risks, esg} (in parallel) -> aggregate -> END
    """

    def analyst(key: str, focus: str):
        def node(state: SectioningState) -> SectioningState:
            reply = llm.invoke(
                [
                    SystemMessage(
                        f"You are an equity analyst. In at most 3 bullet points, analyse only: {focus}."
                    ),
                    HumanMessage(state["article"]),
                ]
            )
            return {key: strip_think(str(reply.content))}

        return node

    def aggregate(state: SectioningState) -> SectioningState:
        # no LLM needed to assemble the sections; an LLM could also write a synthesis here
        report = (
            f"## Financial impact\n{state['financial']}\n\n## Risks\n{state['risks']}\n\n"
            f"## ESG\n{state['esg']}"
        )
        return {"report": report}

    builder = StateGraph(SectioningState)
    builder.add_node(
        "financial",
        analyst("financial", "the financial impact (profit, margins, financing)"),
    )
    builder.add_node("risks", analyst("risks", "the operational and regulatory risks"))
    builder.add_node(
        "esg", analyst("esg", "environmental, social and governance aspects")
    )
    builder.add_node("aggregate", aggregate)
    for node in ["financial", "risks", "esg"]:
        builder.add_edge(START, node)
    builder.add_edge(
        ["financial", "risks", "esg"], "aggregate"
    )  # fan-in: wait for all three
    builder.add_edge("aggregate", END)
    return builder.compile()


# ---------------------------------------------------------------------------
# Voting
# ---------------------------------------------------------------------------
class Vote(BaseModel):
    is_phishing: bool
    reason: str = Field(description="One short sentence")


class VotingState(TypedDict, total=False):
    email: str
    votes: Annotated[
        list[Vote], operator.add
    ]  # reducer: parallel results are concatenated
    verdict: str


class VoterInput(TypedDict):
    email: str
    perspective: str


def build_voting_graph(llm: BaseChatModel) -> CompiledStateGraph:
    """
    START -(Send x N)-> voter -> tally -> END
    """
    structured = llm.with_structured_output(Vote)

    def fan_out(state: VotingState) -> list[Send]:
        # one Send per voter; each voter gets its own private input instead of the whole state
        return [
            Send("voter", {"email": state["email"], "perspective": p})
            for p in VOTER_PERSPECTIVES
        ]

    def voter(inp: VoterInput) -> VotingState:
        vote = structured.invoke(
            [
                SystemMessage(
                    inp["perspective"]
                    + " Decide whether the e-mail is a phishing attempt."
                ),
                HumanMessage(inp["email"]),
            ]
        )
        return {"votes": [vote]}

    def tally(state: VotingState) -> VotingState:
        counts = Counter(v.is_phishing for v in state["votes"])
        verdict = "PHISHING" if counts[True] > counts[False] else "legitimate"
        return {
            "verdict": f"{verdict} ({counts[True]} of {len(state['votes'])} voters say phishing)"
        }

    builder = StateGraph(VotingState)
    builder.add_node("voter", voter)
    builder.add_node("tally", tally)
    builder.add_conditional_edges(START, fan_out, ["voter"])
    builder.add_edge("voter", "tally")
    builder.add_edge("tally", END)
    return builder.compile()


if __name__ == "__main__":
    llm = get_chat_model(temperature=0)

    # 1. Sectioning: three analyses in parallel, then aggregation
    sectioning = build_sectioning_graph(llm)
    save_graph(sectioning, "workflow_3a_sectioning")
    print("===== Sectioning =====")
    print(sectioning.invoke({"article": ARTICLE})["report"])

    # 2. Voting: N reviewers in parallel via Send, majority vote
    voting = build_voting_graph(llm)
    save_graph(voting, "workflow_3b_voting")
    print("\n===== Voting =====")
    for email in EMAILS:
        result = voting.invoke({"email": email})
        print("-" * 100)
        print(email)
        for vote in result["votes"]:
            print(
                f"  vote: {'phishing' if vote.is_phishing else 'ok':<9} {vote.reason}"
            )
        print(f"  => {result['verdict']}")
