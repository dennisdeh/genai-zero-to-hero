"""
Prompt templates with variables, partial variables and chat history, and the effect of temperature.

Part A - templates: `PromptTemplate` and `ChatPromptTemplate` with input variables, `partial()` for values
that are fixed early (e.g. today's date), `MessagesPlaceholder` for chat history, and an LCEL chain
(`prompt | llm | parser`) run over several inputs with `batch()`. The rendering part runs without an LLM.

Part B - temperature: the same creative prompt is sampled several times at different temperatures. We count
distinct answers and the average pairwise word overlap (Jaccard similarity) to quantify how temperature
trades determinism for diversity.

How to run (from the repo root, with Ollama running):
    python -m p07_llms.c02_prompt_engineering_and_structured_outputs.s00_prompt_engineering.main_3_templates_and_temperature
"""

import datetime as dt
import itertools

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)

from p07_llms.c02_prompt_engineering_and_structured_outputs.utils.llm import (
    get_chat_model,
    strip_think,
)

TEMPERATURES = [0.0, 0.8, 1.5]
N_SAMPLES = 5


def build_templates() -> tuple[PromptTemplate, ChatPromptTemplate]:
    """
    Build a plain string template and a chat template with history.
    """
    # 1. A string template with three variables; 'today' is filled in once with partial()
    summary_template = PromptTemplate.from_template(
        "Today is {today}. Summarise the following {doc_type} for a {audience} in at most two sentences:\n\n{text}"
    ).partial(today=dt.date.today().isoformat())

    # 2. A chat template: system message with a variable, history placeholder and the new question
    chat_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant for {company}. Keep answers short.",
            ),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]
    )
    return summary_template, chat_template


def jaccard(a: str, b: str) -> float:
    """
    Jaccard similarity between the word sets of two strings.
    """
    set_a, set_b = set(a.lower().split()), set(b.lower().split())
    return len(set_a & set_b) / max(1, len(set_a | set_b))


def diversity_stats(samples: list[str]) -> tuple[int, float]:
    """
    Number of distinct samples and mean pairwise Jaccard similarity (1.0 = identical wording).
    """
    pairs = list(itertools.combinations(samples, 2))
    mean_sim = sum(jaccard(a, b) for a, b in pairs) / max(1, len(pairs))
    return len(set(samples)), mean_sim


if __name__ == "__main__":
    # 1. Part A: render templates (no LLM needed)
    summary_template, chat_template = build_templates()
    print("Input variables of the summary template:", summary_template.input_variables)
    print("\nRendered summary prompt:\n")
    print(
        summary_template.format(
            doc_type="quarterly report",
            audience="retail investor",
            text="Revenue grew 8% to DKK 12.1bn, EBIT margin 21%. Guidance for the year is unchanged.",
        )
    )
    history = [
        HumanMessage("Do you offer savings accounts?"),
        AIMessage("Yes, we offer a flexible savings account with a variable rate."),
    ]
    print("\nRendered chat prompt with history:")
    for message in chat_template.format_messages(
        company="Nordic Bank", history=history, question="And what is the current rate?"
    ):
        print(f"  [{message.type}] {message.content}")

    # 2. Part A: run the summary template over several inputs with an LCEL chain
    llm = get_chat_model(temperature=0)
    chain = summary_template | llm | StrOutputParser()
    inputs = [
        {
            "doc_type": "press release",
            "audience": "child",
            "text": "The central bank raised its policy rate by 0.25 percentage points to 3.75% to fight inflation.",
        },
        {
            "doc_type": "press release",
            "audience": "bond portfolio manager",
            "text": "The central bank raised its policy rate by 0.25 percentage points to 3.75% to fight inflation.",
        },
    ]
    print("\n===== Same text, different audience variable =====")
    for inp, out in zip(inputs, chain.batch(inputs)):
        print(f"\n[{inp['audience']}] {strip_think(out)}")

    # 3. Part B: temperature sweep
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                "Invent one short, catchy slogan for a budgeting app for students. Reply with the slogan only.",
            )
        ]
    )
    print("\n===== Temperature sweep =====")
    for temperature in TEMPERATURES:
        llm_t = get_chat_model(temperature=temperature, num_predict=40)
        samples = [
            strip_think(s).strip('"')
            for s in (prompt | llm_t | StrOutputParser()).batch([{}] * N_SAMPLES)
        ]
        n_distinct, mean_sim = diversity_stats(samples)
        print(
            f"\ntemperature={temperature}: {n_distinct}/{N_SAMPLES} distinct, mean Jaccard similarity {mean_sim:.2f}"
        )
        for s in samples:
            print(f"   - {s}")
    print(
        "\nLow temperature -> (almost) identical samples, good for extraction/classification. "
        "High temperature -> more diverse (and more error-prone) text, useful for brainstorming."
    )
