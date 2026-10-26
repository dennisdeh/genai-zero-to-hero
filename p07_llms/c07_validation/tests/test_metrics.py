"""
Unit tests for the reference-based metrics (no LLM needed).

Run from the repo root:
    python -m pytest p07_llms/c07_validation/tests/test_metrics.py -v
"""

import pytest

from p07_llms.c07_validation.utils.dataset import EVAL_SET
from p07_llms.c07_validation.utils.metrics import (
    contains_answer,
    exact_match,
    normalise_answer,
    rouge_l,
    token_f1,
)


def test_normalise_answer() -> None:
    assert normalise_answer("  The F1-Score! ") == "f1 score"
    assert normalise_answer("Price/Earnings ratio") == "price earnings ratio"
    assert normalise_answer("10%") == "10%"


@pytest.mark.parametrize(
    "prediction, reference, aliases, expected",
    [
        ("ReLU", "ReLU", [], 1.0),
        ("relu.", "ReLU", [], 1.0),
        ("Rectified Linear Unit", "ReLU", ["rectified linear unit"], 1.0),
        ("The answer is ReLU", "ReLU", [], 0.0),
    ],
)
def test_exact_match(
    prediction: str, reference: str, aliases: list[str], expected: float
) -> None:
    assert exact_match(prediction, reference, aliases) == expected


def test_contains_answer_is_lenient_but_respects_word_boundaries() -> None:
    assert contains_answer("The answer is ReLU.", "ReLU") == 1.0
    assert contains_answer("It was introduced in 2002.", "2002") == 1.0
    assert contains_answer("It was introduced in 20021.", "2002") == 0.0


def test_token_f1() -> None:
    assert token_f1("exchange-traded fund", "exchange-traded fund") == pytest.approx(
        1.0
    )
    # 'an exchange traded fund product': 3 of 4 predicted tokens match (article removed), recall 3/3
    assert token_f1(
        "an exchange traded fund product", "exchange-traded fund"
    ) == pytest.approx(2 * 0.75 / 1.75)
    assert token_f1("bond", "exchange-traded fund") == 0.0


def test_rouge_l() -> None:
    assert rouge_l(
        "attention is all you need", "Attention Is All You Need"
    ) == pytest.approx(1.0)
    # LCS of 'attention is what you need' and 'attention is all you need' = 4 tokens
    assert rouge_l(
        "attention is what you need", "Attention Is All You Need"
    ) == pytest.approx(0.8)
    assert rouge_l("", "Attention Is All You Need") == 0.0


def test_dataset_is_well_formed() -> None:
    ids = [item["id"] for item in EVAL_SET]
    assert len(ids) == len(set(ids)), "ids must be unique"
    for item in EVAL_SET:
        assert {"id", "category", "question", "reference", "aliases"} <= item.keys()
        # every reference must match itself perfectly under all metrics
        assert exact_match(item["reference"], item["reference"], item["aliases"]) == 1.0
