"""
Unit tests for the reference-based metrics (no LLM needed).

Run from the repo root:
    python -m pytest p07_llms/c07_validation/tests/test_metrics.py -v
"""

import pytest

from p07_llms.c07_validation.utils.dataset import EVAL_SET
from p07_llms.c07_validation.utils.metrics import (
    ALL_METRICS,
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
    assert normalise_answer("It was introduced in 2002.") == "it was introduced in 2002"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("1.0%", "1%"),  # the decimal point is kept ...
        ("10.0%", "10%"),  # ... and numbers are canonicalised: 10.0 == 10
        ("3.75%", "3.75%"),
        ("0.50", "0.5"),
        ("1,000", "1000"),  # English thousands separators are removed
        ("DKK 1,234,567.00", "dkk 1234567"),
        ("-5", "-5"),
        ("covid-19", "covid 19"),  # a hyphen glued to a word is not a minus sign
        (
            "1,2",
            "1 2",
        ),  # not a thousands group: treated as a list, digits are not glued together
        ("F1-score", "f1 score"),  # digits inside words are left alone
    ],
)
def test_normalise_answer_numbers(text: str, expected: str) -> None:
    assert normalise_answer(text) == expected


@pytest.mark.parametrize(
    "prediction, reference, aliases, expected",
    [
        ("ReLU", "ReLU", [], 1.0),
        ("relu.", "ReLU", [], 1.0),
        ("Rectified Linear Unit", "ReLU", ["rectified linear unit"], 1.0),
        ("The answer is ReLU", "ReLU", [], 0.0),
        # numbers: removing '.' inside numbers would make all of these wrong
        ("1.0%", "10%", [], 0.0),
        ("3.75%", "37.5%", [], 0.0),
        ("10.0%", "10%", ["10", "10 percent"], 1.0),
        ("10.0", "10%", ["10", "10 percent"], 1.0),
        ("1,000", "1000", [], 1.0),
        ("1,000", "1.000", [], 0.0),
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


def test_numbers_in_token_f1_and_rouge_l() -> None:
    assert token_f1("a rate of 10.0%", "rate of 10%") == pytest.approx(1.0)
    assert token_f1("a rate of 1.0%", "rate of 10%") == pytest.approx(2 / 3)
    assert rouge_l("revenue grew 3.75%", "revenue grew 37.5%") == pytest.approx(2 / 3)


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
        # every reference (and every alias) must match itself perfectly under all metrics
        for answer in [item["reference"], *item["aliases"]]:
            for name, metric in ALL_METRICS.items():
                score = metric(answer, item["reference"], item["aliases"])
                assert score == pytest.approx(1.0), f"{item['id']} {name}({answer!r})"
