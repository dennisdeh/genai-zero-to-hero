"""
Workflow pattern 2: routing.

An LLM classifier with structured output (a pydantic model with a `Literal` category) decides which
specialised branch handles an input. Each branch has its own prompt (and could have its own model, tools or
retrieval). Separating the concerns keeps each prompt short and focused, and lets cheap requests go to cheap
handlers.

Use case: customer messages to a bank are routed to card support, investment questions, complaints or a
general fallback.

How to run (from the repo root, with Ollama running):
    python -m p07_llms.c06_workflows.main_2_routing
"""

from typing import Literal, TypedDict

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

Category = Literal["cards", "investments", "complaint", "general"]

HANDLER_PROMPTS: dict[str, str] = {
    "cards": "You are the card support team of Nordic Bank. Give concrete steps (app menu paths). "
    "If a card is lost or stolen, first tell the customer to block it in the app.",
    "investments": "You are an investment information assistant at Nordic Bank. Explain concepts neutrally. "
    "Never give personal investment advice; suggest booking an adviser meeting instead.",
    "complaint": "You handle complaints at Nordic Bank. Acknowledge the problem, apologise once, explain the "
    "next step and the expected response time (5 working days). Do not promise compensation.",
    "general": "You are a helpful assistant at Nordic Bank. Answer briefly and politely.",
}

MESSAGES = [
    "I think someone stole my Visa card at the train station, what do I do?",
    "What is the difference between an index fund and an actively managed fund?",
    "This is the third time my transfer has been delayed and nobody answers the phone. Unacceptable!",
    "Are you open on Saturdays?",
]


class Route(BaseModel):
    """Routing decision for a customer message."""

    reasoning: str = Field(description="One short sentence explaining the choice")
    category: Category = Field(
        description="cards: debit/credit cards, PIN, limits; investments: funds, shares, pensions; "
        "complaint: the customer is dissatisfied; general: everything else"
    )


class RouterState(TypedDict, total=False):
    message: str
    route: Route
    answer: str


def build_graph(llm: BaseChatModel) -> CompiledStateGraph:
    """
    Classifier node + one handler node per category, connected by a conditional edge.
    """
    classifier = llm.with_structured_output(Route)

    def classify(state: RouterState) -> RouterState:
        route = classifier.invoke(
            [
                SystemMessage("Classify the customer message for routing."),
                HumanMessage(state["message"]),
            ]
        )
        return {"route": route}

    def make_handler(category: str):
        def handle(state: RouterState) -> RouterState:
            reply = llm.invoke(
                [
                    SystemMessage(HANDLER_PROMPTS[category]),
                    HumanMessage(state["message"]),
                ]
            )
            return {"answer": strip_think(str(reply.content))}

        return handle

    builder = StateGraph(RouterState)
    builder.add_node("classify", classify)
    for category in HANDLER_PROMPTS:
        builder.add_node(category, make_handler(category))
        builder.add_edge(category, END)
    builder.add_edge(START, "classify")
    # the routing function returns the name of the next node
    builder.add_conditional_edges(
        "classify", lambda s: s["route"].category, list(HANDLER_PROMPTS)
    )
    return builder.compile()


if __name__ == "__main__":
    # 1. Build the workflow and save its structure
    llm = get_chat_model(temperature=0)
    graph = build_graph(llm)
    save_graph(graph, "workflow_2_routing")

    # 2. Route each message and print the decision and the specialised answer
    for message in MESSAGES:
        result = graph.invoke({"message": message})
        route: Route = result["route"]
        print("=" * 100)
        print(f"Message : {message}")
        print(f"Route   : {route.category}  ({route.reasoning})")
        print(f"Answer  : {result['answer'][:500]}")
