"""
System prompts (roles) and chain-of-thought style instructions.

Part A - roles: the same question is answered under different system prompts (audience, tone, length),
showing that the system message is the main lever for controlling style.

Part B - reasoning: small finance word problems with a known numeric answer are solved with
  1. a "direct" prompt (answer with the number only),
  2. a chain-of-thought (CoT) prompt (reason step by step, then give 'ANSWER: <number>'),
  3. the model's built-in thinking mode (qwen3 / other reasoning models via `reasoning=True`).
The final number is extracted with a regular expression and compared with the ground truth.

How to run (from the repo root, with Ollama running):
    python -m p07_llms.c02_prompt_engineering_and_structured_outputs.s00_prompt_engineering.main_2_roles_and_chain_of_thought
"""

import re
import time
from typing import Union

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from p07_llms.c02_prompt_engineering_and_structured_outputs.utils.llm import (
    get_chat_model,
    strip_think,
)

ROLES = {
    "teacher": "You are a friendly teacher explaining concepts to a 10-year-old. Use a simple analogy.",
    "analyst": "You are a CFA charterholder writing for professional investors. Be precise and technical.",
    "terse": "You answer in a single sentence of at most 20 words.",
}

# Word problems with exact numeric answers (rounded to 2 decimals)
PROBLEMS = [
    (
        "A share costs 80 EUR. It rises 25% and then falls 20%. What is the price now in EUR?",
        80.0,
    ),
    (
        "You deposit 1000 EUR at 4% annual interest, compounded yearly. How many EUR do you have after 3 years? "
        "Round to 2 decimals.",
        1124.86,
    ),
    (
        "A portfolio holds 30 shares at 12 EUR and 50 shares at 8 EUR. What is the average price per share in EUR?",
        9.5,
    ),
    (
        "A fund charges a 2% annual fee. An investor starts with 10000 EUR and the market return is 0%. "
        "How many EUR are left after 2 years? Round to 2 decimals.",
        9604.0,
    ),
    (
        "Revenue was 200 million and grew to 260 million. Operating margin went from 10% to 15%. "
        "By how many million did operating profit increase?",
        19.0,
    ),
]


def run_roles(llm: BaseChatModel, question: str) -> None:
    """
    Answer the same question with different system prompts.
    """
    for role, system_text in ROLES.items():
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_text), ("human", "{question}")]
        )
        answer = (prompt | llm | StrOutputParser()).invoke({"question": question})
        print(f"\n--- role: {role} ---\n{strip_think(answer)[:600]}")


def extract_number(text: str) -> Union[float, None]:
    """
    Extract the final answer: prefer a line 'ANSWER: <number>', otherwise take the last number in the text.
    """
    text = strip_think(text).replace(",", "")
    match = re.search(r"ANSWER:\s*[^\d\-]*(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return float(numbers[-1]) if numbers else None


def solve_all(chain, label: str) -> None:
    """
    Run a chain over all problems and print accuracy and time.
    """
    start = time.perf_counter()
    answers = chain.batch([{"problem": p} for p, _ in PROBLEMS])
    elapsed = time.perf_counter() - start
    correct = 0
    for (problem, truth), answer in zip(PROBLEMS, answers):
        value = extract_number(answer if isinstance(answer, str) else answer.content)
        ok = value is not None and abs(value - truth) < 0.01
        correct += ok
        print(f"  [{'OK ' if ok else 'BAD'}] expected {truth:>9}  got {value}")
    print(f"  => {label}: {correct}/{len(PROBLEMS)} correct in {elapsed:.1f}s")


if __name__ == "__main__":
    # 1. Part A: same question, different system prompts
    llm = get_chat_model(temperature=0)
    print("===== Part A: system prompts / roles =====")
    run_roles(llm, "What is an exchange-traded fund (ETF)?")

    # 2. Part B: direct answer vs chain-of-thought instruction
    print("\n===== Part B: direct vs chain-of-thought =====")
    direct_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer with the final number only. Do not explain."),
            ("human", "{problem}"),
        ]
    )
    cot_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Solve the problem step by step. Write out each intermediate calculation. "
                "Finish with a final line of the form 'ANSWER: <number>'.",
            ),
            ("human", "{problem}"),
        ]
    )
    print("\nDirect answers:")
    solve_all(direct_prompt | llm | StrOutputParser(), "direct")
    print("\nChain-of-thought answers:")
    solve_all(cot_prompt | llm | StrOutputParser(), "chain-of-thought")

    # 3. Built-in reasoning ("thinking") mode: the model reasons before answering, the reasoning is
    #    returned separately in additional_kwargs['reasoning_content'] (Ollama only)
    print("\nBuilt-in thinking mode with the direct prompt:")
    try:
        llm_thinking = get_chat_model(temperature=0, think=True)
        solve_all(direct_prompt | llm_thinking, "thinking mode")
    except Exception as e:  # e.g. a model without thinking support
        print(f"  Thinking mode not available for this model: {e}")

    print(
        "\nChain-of-thought trades latency (more output tokens) for accuracy on multi-step problems. "
        "Reasoning models internalise this, so an explicit CoT instruction matters less for them."
    )
