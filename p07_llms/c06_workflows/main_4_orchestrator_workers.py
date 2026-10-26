"""
Workflow pattern 4: orchestrator-workers.

An orchestrator LLM breaks a task into subtasks that are *not known in advance* (unlike sectioning, where the
split is hard-coded), delegates each subtask to a worker LLM call and a synthesiser combines the results.
In LangGraph the dynamic fan-out is done with the `Send` API: the routing function returns one
`Send("worker", payload)` per planned subtask, and the workers' outputs are collected with a list reducer.

Use case: write a short internal briefing on a topic; the orchestrator plans the sections (structured output),
workers write one section each in parallel, the synthesiser assembles them in the planned order.

How to run (from the repo root, with Ollama running):
    python -m p07_llms.c06_workflows.main_4_orchestrator_workers
"""

import operator
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

TOPIC = (
    "Introducing LLM-based document processing in a mid-sized bank's credit department"
)
MAX_SECTIONS = 5


class Section(BaseModel):
    title: str = Field(description="Section heading")
    description: str = Field(description="What the section should cover, 1-2 sentences")


class Plan(BaseModel):
    sections: list[Section] = Field(
        description=f"Between 3 and {MAX_SECTIONS} sections"
    )


class OrchestratorState(TypedDict, total=False):
    topic: str
    plan: Plan
    completed: Annotated[
        list[tuple[int, str]], operator.add
    ]  # (section index, text) from the workers
    briefing: str


class WorkerInput(TypedDict):
    index: int
    section: Section
    topic: str


def build_graph(llm: BaseChatModel) -> CompiledStateGraph:
    """
    orchestrator -(Send per section)-> worker -> synthesiser
    """
    planner = llm.with_structured_output(Plan)

    def orchestrator(state: OrchestratorState) -> OrchestratorState:
        plan = planner.invoke(
            [
                SystemMessage(
                    f"Plan a short internal briefing with 3-{MAX_SECTIONS} sections. "
                    "Sections must not overlap."
                ),
                HumanMessage(f"Topic: {state['topic']}"),
            ]
        )
        plan.sections = plan.sections[
            :MAX_SECTIONS
        ]  # guard against over-eager planners
        return {"plan": plan}

    def assign_workers(state: OrchestratorState) -> list[Send]:
        return [
            Send("worker", {"index": i, "section": s, "topic": state["topic"]})
            for i, s in enumerate(state["plan"].sections)
        ]

    def worker(inp: WorkerInput) -> OrchestratorState:
        section = inp["section"]
        reply = llm.invoke(
            [
                SystemMessage(
                    "Write one section of an internal briefing: at most 100 words, no heading, "
                    "concrete and practical."
                ),
                HumanMessage(
                    f"Briefing topic: {inp['topic']}\nSection: {section.title}\n"
                    f"It should cover: {section.description}"
                ),
            ]
        )
        return {"completed": [(inp["index"], strip_think(str(reply.content)))]}

    def synthesiser(state: OrchestratorState) -> OrchestratorState:
        # workers finish in arbitrary order -> sort by the planned index
        texts = dict(state["completed"])
        parts = [
            f"## {s.title}\n{texts[i]}" for i, s in enumerate(state["plan"].sections)
        ]
        return {"briefing": f"# {state['topic']}\n\n" + "\n\n".join(parts)}

    builder = StateGraph(OrchestratorState)
    builder.add_node("orchestrator", orchestrator)
    builder.add_node("worker", worker)
    builder.add_node("synthesiser", synthesiser)
    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges("orchestrator", assign_workers, ["worker"])
    builder.add_edge("worker", "synthesiser")
    builder.add_edge("synthesiser", END)
    return builder.compile()


if __name__ == "__main__":
    # 1. Build the workflow and save its structure
    llm = get_chat_model(temperature=0)
    graph = build_graph(llm)
    save_graph(graph, "workflow_4_orchestrator_workers")

    # 2. Run it and show the plan and the final briefing
    result = graph.invoke({"topic": TOPIC})
    print("===== Plan from the orchestrator =====")
    for i, section in enumerate(result["plan"].sections, start=1):
        print(f"{i}. {section.title}: {section.description}")
    print(
        f"\n===== Briefing ({len(result['completed'])} sections written in parallel) =====\n"
    )
    print(result["briefing"])
