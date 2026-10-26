"""
LLM-as-a-judge: grade an answer against a reference with a rubric and structured output.

The judge returns a `JudgeVerdict` (rationale first, so that the model "reasons" before it scores; then a
1-5 correctness score with anchored descriptions and a hallucination flag). Using structured output makes
the scores machine-readable and avoids parsing free text.

Known judge biases to keep in mind: position bias (in pairwise comparisons), verbosity bias (longer answers
look better), self-preference (a model rates its own outputs higher) and limited ability on questions it
cannot answer itself. Mitigations: a reference answer, an explicit rubric, a different/stronger judge model,
and checking agreement with human labels on a sample.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

RUBRIC = """You are a strict grader. Compare the ANSWER to the REFERENCE answer for the QUESTION.

Correctness score:
5 - fully correct and complete; wording may differ from the reference
4 - correct, with minor imprecision or unnecessary but harmless extra text
3 - partially correct, or the correct answer is buried among errors
2 - mostly incorrect, but contains a relevant element
1 - incorrect, irrelevant or a refusal

contains_unsupported_claims: true if the answer states facts that contradict the reference or are
unsupported and likely false (hallucinations); extra correct background information is fine."""

JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RUBRIC),
        ("human", "QUESTION: {question}\n\nREFERENCE: {reference}\n\nANSWER: {answer}"),
    ]
)


class JudgeVerdict(BaseModel):
    rationale: str = Field(description="One or two sentences justifying the score")
    correctness: int = Field(
        ge=1, le=5, description="Correctness score 1-5 according to the rubric"
    )
    contains_unsupported_claims: bool


def judge_answer(
    judge_llm: BaseChatModel, question: str, reference: str, answer: str
) -> JudgeVerdict:
    """
    Grade one answer with the judge model.
    """
    chain = JUDGE_PROMPT | judge_llm.with_structured_output(JudgeVerdict)
    return chain.invoke(
        {"question": question, "reference": reference, "answer": answer}
    )
