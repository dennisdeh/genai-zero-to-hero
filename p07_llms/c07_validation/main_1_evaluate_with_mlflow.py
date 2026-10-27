"""
Evaluating an LLM on a small Q&A set and logging everything to MLflow.

Two prompt variants (a terse "answer only" system prompt and a generic "helpful assistant" prompt) are run
over a hand-written finance/ML question set (utils/dataset.py). Each answer is scored with
  - reference-based metrics: exact match, contains-answer, token F1, ROUGE-L (utils/metrics.py),
  - an LLM judge: rubric-based 1-5 correctness and a hallucination flag via structured output (utils/judge.py),
  - latency.
Every variant becomes one MLflow run with parameters (model, prompt, temperature), aggregated metrics, the
prompt as a text artefact and the per-question results as a table - so prompt or model changes can be
compared side by side in the MLflow UI. This uses only the stable core logging API (log_params, log_metrics,
log_text, log_table); main_2 shows the higher-level `mlflow.genai.evaluate`.

How to run (from the repo root, with Ollama running):
    python -m p07_llms.c07_validation.main_1_evaluate_with_mlflow
    mlflow ui --backend-store-uri sqlite:///p07_llms/c07_validation/outputs/mlflow.db
Environment variables (optional): OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_JUDGE_MODEL (a different, ideally
stronger, judge model), LLM_TO_USE=openai.
"""

import os
import statistics
import time

import mlflow
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from p07_llms.c02_prompt_engineering_and_structured_outputs.utils.llm import (
    get_chat_model,
    strip_think,
)
from p07_llms.c07_validation.utils.dataset import EVAL_SET
from p07_llms.c07_validation.utils.judge import judge_answer
from p07_llms.c07_validation.utils.metrics import ALL_METRICS
from p07_llms.c07_validation.utils.tracking import setup_mlflow

EXPERIMENT_NAME = "llm_qa_evaluation"
TEMPERATURE = 0.0
PROMPT_VARIANTS = {
    "answer_only": "Answer the question with the shortest possible answer: a few words or a number. "
    "No explanation, no full sentence.",
    "helpful_assistant": "You are a helpful assistant.",
}


def model_name(llm: BaseChatModel) -> str:
    """
    Model identifier for logging (works for ChatOllama and ChatOpenAI).
    """
    return (
        getattr(llm, "model", None)
        or getattr(llm, "model_name", None)
        or type(llm).__name__
    )


def evaluate_variant(
    llm: BaseChatModel, judge_llm: BaseChatModel, system_prompt: str
) -> list[dict]:
    """
    Answer every question with the given system prompt and score the answers.
    """
    rows = []
    for item in EVAL_SET:
        start = time.perf_counter()
        response = llm.invoke(
            [SystemMessage(system_prompt), HumanMessage(item["question"])]
        )
        latency = time.perf_counter() - start
        answer = strip_think(str(response.content))

        row = {
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "reference": item["reference"],
            "answer": answer,
            "latency_s": round(latency, 3),
        }
        for name, metric in ALL_METRICS.items():
            row[name] = metric(answer, item["reference"], item["aliases"])
        verdict = judge_answer(judge_llm, item["question"], item["reference"], answer)
        row["judge_correctness"] = verdict.correctness
        row["judge_hallucination"] = float(verdict.contains_unsupported_claims)
        row["judge_rationale"] = verdict.rationale
        rows.append(row)
        print(
            f"  {item['id']:<7} EM={row['exact_match']:.0f} F1={row['token_f1']:.2f} "
            f"judge={verdict.correctness}  {answer[:70]!r}"
        )
    return rows


def aggregate(rows: list[dict]) -> dict[str, float]:
    """
    Mean of every numeric metric, overall and per category.
    """
    keys = [*ALL_METRICS, "judge_correctness", "judge_hallucination", "latency_s"]
    metrics = {f"{k}_mean": statistics.mean(r[k] for r in rows) for k in keys}
    for category in sorted({r["category"] for r in rows}):
        subset = [r for r in rows if r["category"] == category]
        metrics[f"token_f1_{category}"] = statistics.mean(r["token_f1"] for r in subset)
        metrics[f"judge_correctness_{category}"] = statistics.mean(
            r["judge_correctness"] for r in subset
        )
    metrics["answer_words_mean"] = statistics.mean(
        len(r["answer"].split()) for r in rows
    )
    return metrics


if __name__ == "__main__":
    # 1. Models: the system under test and the judge (ideally a different, stronger model)
    llm = get_chat_model(temperature=TEMPERATURE)
    judge_llm = get_chat_model(temperature=0, model=os.getenv("OLLAMA_JUDGE_MODEL"))
    tracking_uri = setup_mlflow(EXPERIMENT_NAME)
    print(f"MLflow tracking URI: {tracking_uri}")

    # 2. One MLflow run per prompt variant
    summary = {}
    for variant, system_prompt in PROMPT_VARIANTS.items():
        print(f"\n===== Prompt variant '{variant}' =====")
        with mlflow.start_run(run_name=f"{model_name(llm)}-{variant}"):
            mlflow.log_params(
                {
                    "model": model_name(llm),
                    "judge_model": model_name(judge_llm),
                    "backend": os.getenv("LLM_TO_USE", "ollama"),
                    "prompt_variant": variant,
                    "temperature": TEMPERATURE,
                    "n_questions": len(EVAL_SET),
                }
            )
            mlflow.log_text(system_prompt, "prompt/system_prompt.txt")
            rows = evaluate_variant(llm, judge_llm, system_prompt)
            metrics = aggregate(rows)
            mlflow.log_metrics(metrics)
            # per-question results: visible in the run's "Artifacts" tab and comparable across runs
            mlflow.log_table(
                data={k: [r[k] for r in rows] for k in rows[0]},
                artifact_file="eval_results.json",
            )
            summary[variant] = metrics

    # 3. Compare the variants
    print("\nMetric                    " + "".join(f"{v:>20}" for v in summary))
    for key in [
        "exact_match_mean",
        "contains_answer_mean",
        "token_f1_mean",
        "rouge_l_mean",
        "judge_correctness_mean",
        "judge_hallucination_mean",
        "answer_words_mean",
        "latency_s_mean",
    ]:
        print(f"{key:<26}" + "".join(f"{summary[v][key]:>20.2f}" for v in summary))
    print(
        "\nThe verbose prompt scores poorly on exact match although its answers are often correct - "
        "reference-based metrics measure format as much as correctness. Contains-answer and the judge are "
        "more forgiving; the judge is the most flexible but also the least reproducible metric."
    )
    print(
        "\nOpen the MLflow UI with:\n  mlflow ui --backend-store-uri "
        "sqlite:///p07_llms/c07_validation/outputs/mlflow.db"
    )
