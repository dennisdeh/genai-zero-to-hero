"""
Structured outputs with pydantic and `llm.with_structured_output(Model)`.

We extract companies, event type, sentiment and financial figures from short financial news articles into
a pydantic model with enums, nested models and lists (see utils/schemas.py). With Ollama the pydantic
model is converted to a JSON schema and passed as the `format` parameter, so the server constrains
decoding to valid JSON matching the schema; with OpenAI the "structured outputs" feature does the same.
The result is a validated Python object instead of free text.

How to run (from the repo root, with Ollama running):
    python -m p07_llms.c02_prompt_engineering_and_structured_outputs.s01_structured_outputs.main_1_with_structured_output
"""

import json

from langchain_core.prompts import ChatPromptTemplate

from p07_llms.c02_prompt_engineering_and_structured_outputs.s01_structured_outputs.utils.data import (
    NEWS_ARTICLES,
)
from p07_llms.c02_prompt_engineering_and_structured_outputs.s01_structured_outputs.utils.schemas import (
    NewsExtraction,
    Sentiment,
)
from p07_llms.c02_prompt_engineering_and_structured_outputs.utils.llm import (
    get_chat_model,
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract structured data from financial news. Use only information stated in the text; "
            "use null for unknown optional fields. Do not invent tickers.",
        ),
        ("human", "{article}"),
    ]
)


def show_schema() -> None:
    """
    Print (part of) the JSON schema that is sent to the model.
    """
    schema = NewsExtraction.model_json_schema()
    print("JSON schema sent to the model (truncated):")
    print(json.dumps(schema, indent=2)[:1200], "...\n")


if __name__ == "__main__":
    # 1. Inspect the schema derived from the pydantic model
    show_schema()

    # 2. Bind the schema to the model: the chain now returns NewsExtraction objects
    llm = get_chat_model(temperature=0)
    extractor = PROMPT | llm.with_structured_output(NewsExtraction)

    # 3. Extract from all articles in one batch
    results: list[NewsExtraction] = extractor.batch(
        [{"article": a} for a in NEWS_ARTICLES]
    )

    # 4. The results are typed Python objects - we can use them directly in code
    for article, result in zip(NEWS_ARTICLES, results):
        print("=" * 100)
        print(article)
        print("-" * 100)
        print(result.model_dump_json(indent=2))

    print("\n===== Aggregations on typed objects =====")
    negatives = [r for r in results if r.sentiment == Sentiment.NEGATIVE]
    print(f"Negative news items: {len(negatives)} of {len(results)}")
    for r in results:
        subject = next((c.name for c in r.companies if c.role == "subject"), "?")
        print(
            f"  {subject:<25} {r.event_type.value:<20} {r.sentiment.value:<9} {len(r.figures)} figures"
        )

    # 5. include_raw=True returns the raw AIMessage and a parsing error instead of raising
    raw_extractor = PROMPT | llm.with_structured_output(
        NewsExtraction, include_raw=True
    )
    out = raw_extractor.invoke({"article": NEWS_ARTICLES[0]})
    print("\n===== include_raw=True =====")
    print("keys:", list(out.keys()))
    print("raw content (first 300 chars):", str(out["raw"].content)[:300])
    print("parsing_error:", out["parsing_error"])
