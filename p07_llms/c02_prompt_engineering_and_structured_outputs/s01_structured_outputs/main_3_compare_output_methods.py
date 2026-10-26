"""
Comparison of ways to get structured output from an LLM.

The same extraction task (financial news -> NewsExtraction) is solved with
  1. prompt + manual parsing: format instructions in the prompt, then json.loads + model_validate,
  2. prompt + `PydanticOutputParser`: the same idea packaged by LangChain (also strips ```json fences),
  3. JSON mode (`method="json_mode"`): the server guarantees syntactically valid JSON, but the schema is
     only described in the prompt,
  4. JSON schema (`method="json_schema"`): constrained decoding against the schema (the default),
  5. tool / function calling (`method="function_calling"`): the schema is offered as a tool to call.
For each method we count valid results and measure the time taken.

How to run (from the repo root, with Ollama running):
    python -m p07_llms.c02_prompt_engineering_and_structured_outputs.s01_structured_outputs.main_3_compare_output_methods
"""

import json
import re
import time
from typing import Callable

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from p07_llms.c02_prompt_engineering_and_structured_outputs.s01_structured_outputs.utils.data import (
    NEWS_ARTICLES,
)
from p07_llms.c02_prompt_engineering_and_structured_outputs.s01_structured_outputs.utils.schemas import (
    NewsExtraction,
)
from p07_llms.c02_prompt_engineering_and_structured_outputs.utils.llm import (
    get_chat_model,
    strip_think,
)

PARSER = PydanticOutputParser(pydantic_object=NewsExtraction)

# Prompt WITH the schema written into it (needed for methods 1-3)
PROMPT_WITH_FORMAT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract structured data from financial news.\n{format_instructions}",
        ),
        ("human", "{article}"),
    ]
).partial(format_instructions=PARSER.get_format_instructions())

# Prompt WITHOUT schema (methods 4-5 pass the schema through the API)
PROMPT_PLAIN = ChatPromptTemplate.from_messages(
    [
        ("system", "You extract structured data from financial news."),
        ("human", "{article}"),
    ]
)


def parse_manually(text: str) -> NewsExtraction:
    """
    The "do it yourself" approach: find the outermost {...} block, json.loads it and validate.
    """
    text = strip_think(text)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("no JSON object found in the answer")
    return NewsExtraction.model_validate(json.loads(match.group(0)))


def make_methods(llm: BaseChatModel) -> dict[str, Callable[[str], NewsExtraction]]:
    """
    Return a function per method that maps an article to a NewsExtraction (or raises).
    """
    text_chain = PROMPT_WITH_FORMAT | llm | StrOutputParser()

    def structured(
        method: str, prompt: ChatPromptTemplate
    ) -> Callable[[str], NewsExtraction]:
        chain = prompt | llm.with_structured_output(
            NewsExtraction, method=method, include_raw=True
        )

        def run(article: str) -> NewsExtraction:
            out = chain.invoke({"article": article})
            if out["parsing_error"] is not None or out["parsed"] is None:
                raise ValueError(
                    out["parsing_error"] or "model returned no structured output"
                )
            return out["parsed"]

        return run

    return {
        "manual json.loads": lambda a: parse_manually(
            text_chain.invoke({"article": a})
        ),
        "PydanticOutputParser": lambda a: PARSER.parse(
            strip_think(text_chain.invoke({"article": a}))
        ),
        "json_mode": structured("json_mode", PROMPT_WITH_FORMAT),
        "json_schema": structured("json_schema", PROMPT_PLAIN),
        "function_calling": structured("function_calling", PROMPT_PLAIN),
    }


if __name__ == "__main__":
    # 1. What the format instructions look like (this is what methods 1-3 put in the prompt)
    print("Format instructions from PydanticOutputParser (first 600 chars):")
    print(PARSER.get_format_instructions()[:600], "...\n")

    # 2. Run every method on every article
    llm = get_chat_model(temperature=0)
    summary = []
    for name, method in make_methods(llm).items():
        n_ok, errors = 0, []
        start = time.perf_counter()
        for article in NEWS_ARTICLES:
            try:
                method(article)
                n_ok += 1
            except (
                OutputParserException,
                ValidationError,
                ValueError,
                json.JSONDecodeError,
            ) as e:
                errors.append(str(e).splitlines()[0][:120])
            except Exception as e:  # e.g. a model that does not support tool calling
                errors.append(f"{type(e).__name__}: {str(e)[:100]}")
        elapsed = time.perf_counter() - start
        summary.append((name, n_ok, elapsed))
        print(f"{name:<22} valid {n_ok}/{len(NEWS_ARTICLES)}  ({elapsed:.1f}s)")
        for err in errors[:2]:
            print(f"    error: {err}")

    # 3. Summary table
    print("\nMethod                 valid   seconds")
    for name, n_ok, elapsed in summary:
        print(f"{name:<22} {n_ok}/{len(NEWS_ARTICLES)}     {elapsed:6.1f}")
    print(
        "\nRule of thumb: use json_schema (constrained decoding) when the backend supports it, tool calling "
        "when the output should trigger an action, and a parser + retries for backends without either."
    )
