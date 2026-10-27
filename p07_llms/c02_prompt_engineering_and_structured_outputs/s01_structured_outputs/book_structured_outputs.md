### Structured Outputs

When the consumer of an LLM answer is a program, we need data, not prose: a label, a list of entities, an
invoice with line items. Asking politely for JSON works most of the time, and "most of the time" is not good
enough for a pipeline that processes thousands of documents. This section shows the reliable approach:
**define the output as a pydantic model, let the backend constrain the generation to that schema, and
validate the result.**

#### Schemas as pydantic models

`utils/schemas.py` defines two schemas.

`NewsExtraction` (financial news) uses the common building blocks:

- **enums** (`Sentiment`, `EventType`) and `Literal[...]` restrict a field to a closed set of values,
- **nested models** (`Company`, `FinancialFigure`) and **lists** of them,
- `Optional[...]` with default `None` for information that may be missing - this gives the model a legal way
  to say "not stated" instead of inventing a value (an important anti-hallucination device),
- `Field(description=...)`: the descriptions end up in the JSON schema and act as per-field instructions.

`Invoice` adds **validation**: field constraints (`quantity > 0`, `0 <= vat_rate <= 1`, a regex for the
ISO currency code, `datetime.date` for dates) and `model_validator`s that check that line totals, subtotal,
VAT and total add up and that the due date is after the issue date.

#### How `with_structured_output` works

```python
extractor = prompt | llm.with_structured_output(NewsExtraction)
result: NewsExtraction = extractor.invoke({"article": text})
```

`with_structured_output` converts the pydantic model to a JSON schema (`NewsExtraction.model_json_schema()`),
passes it to the backend and attaches a parser that turns the answer into a `NewsExtraction` instance. The
`method` argument selects the mechanism:

| method | mechanism | guarantees |
|---|---|---|
| `json_schema` (default for Ollama) | the schema is sent as `format`; the server restricts sampling to tokens that keep the output valid against the schema (grammar-constrained decoding) | valid JSON that matches the schema's structure |
| `function_calling` | the schema is offered as a tool; the model "calls" it with arguments | depends on the model's tool-calling skill |
| `json_mode` | the server only guarantees syntactically valid JSON; the schema must be described in the prompt | valid JSON, any structure |

**Constrained decoding** works by masking the logits: at each step, tokens that would make the partial output
invalid under the grammar derived from the JSON schema get probability zero. The model therefore *cannot*
produce a missing bracket or an unknown enum value. It can still produce wrong *content* - a number copied
incorrectly, a sentiment the text does not support - which is why semantic validation remains necessary.

With `include_raw=True` the chain returns a dict `{"raw": AIMessage, "parsed": ..., "parsing_error": ...}`
instead of raising, which is what we need for retries and for logging failures.

#### Validation and retry

`utils/retry.py` implements a small validation-aware retry loop:

1. call `llm.with_structured_output(schema, include_raw=True)`,
2. if `parsing_error` is set (JSON or pydantic validation failed), append the model's previous answer and a
   message containing the validation error to the conversation, and ask for a corrected answer,
3. stop after `max_attempts` and return the errors.

Not every provider reports validation failures through `parsing_error`. With `ChatOpenAI` and the default
`method="json_schema"`, the openai SDK parses the response into the pydantic model inside the call, so a
failing validator *raises* a `ValidationError` even with `include_raw=True`. The loop therefore also catches
`ValidationError`/`OutputParserException` and feeds that error back (without the previous answer, which is
not available in that case).

`main_2_invoice_extraction_with_retry.py` first shows what pydantic catches on a hand-made bad record (no
LLM needed), then extracts two invoices. The first is consistent and usually validates at once. The second
contains a typo in the source document (the stated total is 565.00, but 444.00 + 111.00 = 555.00).
Validation flags it; on retry the model may either recompute the total (silently "correcting" the document)
or fail repeatedly. Both outcomes show an important design point: **retries fix model errors, not data
errors**. A production pipeline should route records that still fail validation, together with the error
messages, to a human review queue rather than retry forever.

LangChain also has ready-made helpers for the same idea (e.g. `OutputFixingParser` and `RetryOutputParser`
in `langchain-classic`), and libraries such as *Instructor* are built entirely around "pydantic + retry".

#### Comparison of methods

`main_3_compare_output_methods.py` solves the news extraction task five ways and reports the number of valid
results and the run time:

1. **manual parsing**: schema in the prompt, then regex + `json.loads` + `model_validate`,
2. **`PydanticOutputParser`**: the same, packaged - `get_format_instructions()` produces the schema text for
   the prompt and `parse()` strips markdown code fences before validating,
3. **JSON mode**, 4. **JSON schema** and 5. **function calling** via `with_structured_output(method=...)`.

The prompt-only approaches (1, 2) work with any model, including old completion APIs, but fail when the
model adds commentary, uses the wrong key names or truncates the output. JSON mode removes syntax errors,
the JSON schema method removes structural errors as well. Practical recommendations:

- prefer `json_schema` (constrained decoding) where the backend supports it (Ollama, OpenAI, vLLM, llama.cpp),
- use tool calling when the structured output *is* an action (agents, chapter p08),
- keep schemas small and flat where possible, use enums for categories and `Optional` for missing data,
- turn off the thinking mode of reasoning models for simple extraction (faster), or keep it on for
  extraction that requires reasoning (e.g. computing derived fields),
- always validate, log failures, and evaluate extraction quality field by field on a labelled set (c07).
