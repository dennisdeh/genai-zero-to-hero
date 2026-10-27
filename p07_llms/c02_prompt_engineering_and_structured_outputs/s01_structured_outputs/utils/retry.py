"""
Validation-aware retry for structured extraction.

Constrained decoding guarantees *syntactically* valid JSON, but not *semantically* valid data: numbers may
not add up, dates may be impossible. Pydantic validators detect this; here we feed the validation error back
to the model and ask it to correct its answer, up to `max_attempts` times.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Type

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError


@dataclass
class ExtractionResult:
    parsed: Optional[BaseModel]
    attempts: int
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.parsed is not None


def short_error(error: object) -> str:
    """
    Keep only the validation part of a LangChain parsing error (drop the echoed completion and doc links).
    """
    text = str(error)
    if ". Got: " in text:
        text = text.split(". Got: ", 1)[1]
    lines = [
        line
        for line in text.splitlines()
        if "errors.pydantic.dev" not in line and "For troubleshooting" not in line
    ]
    return "\n".join(lines).strip()


def extract_with_retry(
    llm: BaseChatModel,
    schema: Type[BaseModel],
    system_prompt: str,
    text: str,
    max_attempts: int = 3,
    **structured_kwargs: Any,
) -> ExtractionResult:
    """
    Extract `schema` from `text`; on a parsing/validation error (reported via `parsing_error` or
    raised during the call), show the model its previous answer (when available) and the error message and ask
    for a corrected answer.

    Args:
        llm: Chat model supporting `with_structured_output`.
        schema: Pydantic model to extract.
        system_prompt: Instructions for the extraction.
        text: Source document.
        max_attempts: Maximum number of model calls.
        **structured_kwargs: Passed on to `with_structured_output` (e.g. method="json_schema").

    Returns:
        ExtractionResult with the parsed object (or None), the number of attempts and the error messages.
    """
    structured_llm = llm.with_structured_output(
        schema, include_raw=True, **structured_kwargs
    )
    messages: list[BaseMessage] = [SystemMessage(system_prompt), HumanMessage(text)]
    errors: list[str] = []

    for attempt in range(1, max_attempts + 1):
        raw: Optional[AIMessage] = None
        try:
            out = structured_llm.invoke(messages)
        except (ValidationError, OutputParserException) as exc:
            # Some providers validate inside the call instead of reporting `parsing_error`: e.g. ChatOpenAI
            # with method="json_schema" lets the openai SDK parse the response into the pydantic model, so a
            # failing validator raises here. The raw answer is not available then; we only feed back the error.
            error = short_error(exc)
        else:
            if out["parsing_error"] is None and out["parsed"] is not None:
                return ExtractionResult(
                    parsed=out["parsed"], attempts=attempt, errors=errors
                )
            error = short_error(
                out["parsing_error"] or "the model returned no structured output"
            )
            raw = out["raw"]
        errors.append(error)
        # keep the conversation: previous (wrong) answer, if known, + feedback with the validation error
        if raw is not None:
            messages.append(AIMessage(content=raw.content or str(raw.tool_calls)))
        messages.append(
            HumanMessage(
                "Your answer failed validation with this error:\n"
                f"{error}\n"
                "Re-read the document carefully and return a corrected answer that satisfies the schema."
            )
        )
    return ExtractionResult(parsed=None, attempts=max_attempts, errors=errors)
