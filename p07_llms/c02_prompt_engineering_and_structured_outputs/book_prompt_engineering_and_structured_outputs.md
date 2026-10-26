## Prompt Engineering and Structured Outputs

A pretrained and instruction-tuned LLM is a general-purpose text-to-text function. Before reaching for
fine-tuning (chapter c03) or retrieval (chapter c04), the cheapest and fastest lever we have is the *input*:
the prompt. This chapter covers two closely related skills:

- **Prompt engineering** (section s00): how to phrase instructions, use system prompts (roles), give
  examples (few-shot), ask for step-by-step reasoning, build reusable prompt templates and pick a sampling
  temperature.
- **Structured outputs** (section s01): how to make the model return data that a program can consume -
  JSON that matches a schema, validated by pydantic - instead of free text.

The two belong together: in applications an LLM is rarely the last step. Its answer is parsed, stored in a
database, used to route a request or passed on to a tool. A good prompt that produces an unparseable answer
is a broken system.

All examples use the local Ollama server from chapter c01 through `get_llm()` (default model `qwen3:8b`)
and can be switched to OpenAI with `LLM_TO_USE=openai`. The helper `utils/llm.py` adds two conveniences on
top of `get_llm()`: changing the temperature (via pydantic's `model_copy`) and switching the "thinking" mode
of reasoning models such as qwen3 on or off (`reasoning=True/False` in `ChatOllama`).

Some general guidelines that the examples illustrate:

1. **Be explicit about the task, the audience and the output format.** Ambiguity is the main source of
   failures; the model cannot ask clarifying questions.
2. **Separate instructions from data.** Put instructions in the system message and the (untrusted) data in
   the human message, ideally clearly delimited. This also reduces the risk of prompt injection.
3. **Show, don't only tell.** A handful of examples (few-shot) pins down format and decision boundaries
   better than a paragraph of rules.
4. **Let the model think before it answers** on multi-step problems, either with an explicit instruction
   (chain-of-thought) or by using a reasoning model.
5. **Constrain the output** with a schema when a program consumes it, and **validate** the result.
6. **Measure.** Prompt changes are model changes: evaluate them on a small labelled set (see chapter c07).
