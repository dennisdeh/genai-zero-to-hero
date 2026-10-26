"""
Zero-shot vs few-shot prompting on a small financial-news sentiment task.

The same classification task (positive / negative / neutral headline) is solved with three prompts:
  1. zero-shot: only the instruction,
  2. zero-shot + system role and label definitions,
  3. few-shot: the same as 2. plus a handful of worked examples, built with
     `FewShotChatMessagePromptTemplate`.
For each prompt we measure accuracy on a small labelled set and count answers that are not one of the
allowed labels (a typical failure of under-specified prompts).

How to run (from the repo root, with Ollama running, see p07_llms/c01_running_llms/s01_ollama):
    python -m p07_llms.c02_prompt_engineering_and_structured_outputs.s00_prompt_engineering.main_1_zero_vs_few_shot
Environment variables (optional): OLLAMA_BASE_URL (default http://localhost:07011), OLLAMA_MODEL
(default qwen3:8b), LLM_TO_USE=openai to use OpenAI instead.
"""

from collections import Counter

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from p07_llms.c02_prompt_engineering_and_structured_outputs.utils.llm import (
    get_chat_model,
    strip_think,
)

LABELS = ["positive", "negative", "neutral"]

# Small hand-written evaluation set (headline, gold label)
EVAL_SET = [
    (
        "Novo Nordisk raises full-year sales guidance after strong demand for Wegovy",
        "positive",
    ),
    ("Shipping group warns of lower earnings as freight rates collapse", "negative"),
    ("Company to hold its annual general meeting on 14 March", "neutral"),
    ("Bank fined EUR 2bn for anti-money-laundering failures", "negative"),
    ("Chipmaker beats analyst expectations and announces share buyback", "positive"),
    ("Retailer appoints new chief financial officer from within the group", "neutral"),
    ("Airline cuts 3,000 jobs as fuel costs soar", "negative"),
    (
        "Utility completes refinancing of its revolving credit facility on unchanged terms",
        "neutral",
    ),
    ("Biotech shares jump 40% after successful phase III trial", "positive"),
    ("Car maker recalls 500,000 vehicles over faulty brakes", "negative"),
]

# Worked examples for the few-shot prompt (not part of the evaluation set)
FEW_SHOT_EXAMPLES = [
    {
        "headline": "Brewer's operating profit rises 12% on higher prices",
        "label": "positive",
    },
    {
        "headline": "Telecom operator swings to a loss after network write-down",
        "label": "negative",
    },
    {
        "headline": "Insurer publishes its financial calendar for next year",
        "label": "neutral",
    },
    {
        "headline": "Mining company's output falls short of target due to strikes",
        "label": "negative",
    },
]


def build_prompts() -> dict[str, ChatPromptTemplate]:
    """
    Build the three prompt variants that are compared.
    """
    # 1. Plain zero-shot instruction
    zero_shot = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                "What is the sentiment of this financial news headline?\n\n{headline}",
            )
        ]
    )

    # 2. System role + precise task definition + output format
    system_text = (
        "You are a financial analyst classifying news headlines by their likely impact on the company's "
        "share price.\n"
        "Labels:\n"
        "- positive: good news for shareholders (growth, beats, upgrades, buybacks)\n"
        "- negative: bad news for shareholders (losses, fines, layoffs, recalls, warnings)\n"
        "- neutral: routine information without a clear price impact\n"
        "Answer with exactly one word: positive, negative or neutral."
    )
    zero_shot_role = ChatPromptTemplate.from_messages(
        [("system", system_text), ("human", "{headline}")]
    )

    # 3. Few-shot: the examples are rendered as alternating human/ai messages
    example_prompt = ChatPromptTemplate.from_messages(
        [("human", "{headline}"), ("ai", "{label}")]
    )
    few_shot_examples = FewShotChatMessagePromptTemplate(
        examples=FEW_SHOT_EXAMPLES,
        example_prompt=example_prompt,
    )
    few_shot = ChatPromptTemplate.from_messages(
        [("system", system_text), few_shot_examples, ("human", "{headline}")]
    )
    return {
        "zero_shot": zero_shot,
        "zero_shot_role": zero_shot_role,
        "few_shot": few_shot,
    }


def normalise_label(answer: str) -> str:
    """
    Map a free-text answer to one of the labels, or 'invalid' if no unique label is found.
    """
    text = strip_think(answer).lower()
    found = [label for label in LABELS if label in text]
    return found[0] if len(found) == 1 else "invalid"


def evaluate_prompt(
    llm: BaseChatModel, prompt: ChatPromptTemplate
) -> tuple[float, Counter, list[str]]:
    """
    Run a prompt over the evaluation set and return accuracy, a counter of predicted labels and raw answers.
    """
    chain = prompt | llm | StrOutputParser()
    # batch() sends the requests concurrently (Ollama queues them)
    raw_answers = chain.batch([{"headline": h} for h, _ in EVAL_SET])
    predictions = [normalise_label(a) for a in raw_answers]
    accuracy = sum(p == gold for p, (_, gold) in zip(predictions, EVAL_SET)) / len(
        EVAL_SET
    )
    return accuracy, Counter(predictions), raw_answers


if __name__ == "__main__":
    # 1. Build the prompt variants and show what is actually sent to the model
    prompts = build_prompts()
    for name, prompt in prompts.items():
        print(f"\n===== Prompt '{name}' rendered for one headline =====")
        for message in prompt.format_messages(headline=EVAL_SET[0][0]):
            print(f"[{message.type}] {message.content}")

    # 2. Get the LLM (temperature 0 for reproducible classification)
    llm = get_chat_model(temperature=0)

    # 3. Evaluate each prompt variant
    results = {}
    for name, prompt in prompts.items():
        accuracy, counts, raw = evaluate_prompt(llm, prompt)
        results[name] = accuracy
        print(f"\n===== {name} =====")
        print(f"Accuracy: {accuracy:.0%}   predicted labels: {dict(counts)}")
        print(f"Example raw answer: {strip_think(raw[0])[:200]!r}")

    # 4. Summary
    print("\nSummary (accuracy):")
    for name, accuracy in results.items():
        print(f"  {name:<15} {accuracy:.0%}")
    print(
        "\nTypically the plain zero-shot prompt returns long explanations (often 'invalid'), while the "
        "role + label definitions fix the format and the few-shot examples calibrate borderline cases."
    )
