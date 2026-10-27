"""
The same evaluation with MLflow's GenAI evaluation API: `mlflow.genai.evaluate(data, predict_fn, scorers)`.

Instead of looping and logging by hand (main_1), we describe
  - the data: a list of {"inputs": {...}, "expectations": {...}} records,
  - the application: `predict_fn(question)`, called with the "inputs" as keyword arguments,
  - the scorers: functions decorated with `@scorer` that receive inputs/outputs/expectations and return a
    number, a bool or an `mlflow.entities.Feedback` (value + rationale).
MLflow runs the predictions, records a trace per question, applies the scorers, logs the aggregated metrics
to a run and attaches the per-question assessments to the traces ("Traces"/"Evaluations" tabs in the UI).

MLflow also ships built-in LLM judges (e.g. `mlflow.genai.scorers.Correctness`, `Safety`,
`RelevanceToQuery`); they need a judge model configured through a model URI (by default an OpenAI model).
Here the judge is our own rubric (utils/judge.py) wrapped as a custom scorer, so everything runs locally.

How to run (from the repo root, with Ollama running):
    python -m p07_llms.c07_validation.main_2_mlflow_genai_evaluate
    mlflow ui --backend-store-uri sqlite:///p07_llms/c07_validation/outputs/mlflow.db
"""

import os

import mlflow
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer

from p07_llms.c02_prompt_engineering_and_structured_outputs.utils.llm import (
    get_chat_model,
    strip_think,
)
from p07_llms.c07_validation.utils.dataset import EVAL_SET
from p07_llms.c07_validation.utils.judge import judge_answer
from p07_llms.c07_validation.utils.metrics import exact_match, rouge_l, token_f1
from p07_llms.c07_validation.utils.tracking import setup_mlflow

EXPERIMENT_NAME = "llm_qa_evaluation_genai"
SYSTEM_PROMPT = (
    "Answer the question with the shortest possible answer: a few words or a number."
)


def build_data() -> list[dict]:
    """
    Convert the evaluation set to MLflow's record format.
    """
    return [
        {
            "inputs": {"question": item["question"]},
            "expectations": {
                "expected_response": item["reference"],
                "aliases": item["aliases"],
            },
        }
        for item in EVAL_SET
    ]


def build_scorers(judge_llm: BaseChatModel) -> list:
    """
    Custom scorers; the function name becomes the metric name (e.g. 'exact_match_score/mean').
    """

    @scorer
    def exact_match_score(outputs: str, expectations: dict) -> float:
        return exact_match(
            outputs, expectations["expected_response"], expectations["aliases"]
        )

    @scorer
    def token_f1_score(outputs: str, expectations: dict) -> float:
        return token_f1(
            outputs, expectations["expected_response"], expectations["aliases"]
        )

    @scorer
    def rouge_l_score(outputs: str, expectations: dict) -> float:
        return rouge_l(
            outputs, expectations["expected_response"], expectations["aliases"]
        )

    @scorer
    def concise(outputs: str) -> bool:
        # a simple format check without reference: at most 8 words
        return len(outputs.split()) <= 8

    @scorer
    def judge_correctness(inputs: dict, outputs: str, expectations: dict) -> Feedback:
        verdict = judge_answer(
            judge_llm, inputs["question"], expectations["expected_response"], outputs
        )
        return Feedback(value=verdict.correctness, rationale=verdict.rationale)

    return [
        exact_match_score,
        token_f1_score,
        rouge_l_score,
        concise,
        judge_correctness,
    ]


def make_predict_fn(llm: BaseChatModel):
    """
    The application under test, as a function of the 'inputs' keys.
    """

    def predict_fn(question: str) -> str:
        response = llm.invoke([SystemMessage(SYSTEM_PROMPT), HumanMessage(question)])
        return strip_think(str(response.content))

    return predict_fn


if __name__ == "__main__":
    # 1. Models and MLflow
    llm = get_chat_model(temperature=0)
    judge_llm = get_chat_model(temperature=0, model=os.getenv("OLLAMA_JUDGE_MODEL"))
    print(f"MLflow tracking URI: {setup_mlflow(EXPERIMENT_NAME)}")

    # 2. Run the evaluation inside a named run
    with mlflow.start_run(run_name="answer_only_prompt"):
        mlflow.log_param("system_prompt", SYSTEM_PROMPT)
        result = mlflow.genai.evaluate(
            data=build_data(),
            predict_fn=make_predict_fn(llm),
            scorers=build_scorers(judge_llm),
        )

    # 3. Aggregated metrics and per-question results
    print("\nAggregated metrics:")
    for name, value in sorted(result.metrics.items()):
        print(f"  {name:<32} {value:.3f}")
    table = result.tables.get("eval_results") if result.tables else None
    if table is not None:
        scorer_names = [s.name for s in build_scorers(judge_llm)]
        columns = [
            c
            for c in table.columns
            if c.split("/")[0] in scorer_names and c.endswith("/value")
        ]
        columns += [c for c in ("response",) if c in table.columns]
        print("\nPer-question results (first rows):")
        print(table[columns].head(12).to_string(max_colwidth=40))
