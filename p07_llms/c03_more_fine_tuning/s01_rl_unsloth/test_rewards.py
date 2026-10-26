"""
Unit tests for the GRPO reward functions and answer-extraction helpers (CPU only, no GPU libraries needed).

Run from the repo root:
    python -m pytest p07_llms/c03_more_fine_tuning/s01_rl_unsloth/test_rewards.py -v
"""

import pytest

from p07_llms.c03_more_fine_tuning.s01_rl_unsloth.utils.rewards import (
    REWARD_FUNCS,
    SYSTEM_PROMPT,
    XML_COT_FORMAT,
    build_prompt,
    completion_to_text,
    correctness_reward_func,
    count_xml,
    extract_hash_answer,
    extract_xml_answer,
    format_gsm8k_example,
    int_reward_func,
    normalise_number,
    score_responses,
    soft_format_reward_func,
    strict_format_reward_func,
    xmlcount_reward_func,
)

PERFECT = XML_COT_FORMAT.format(reasoning="3 + 4 = 7", answer="7")
SLOPPY = "<reasoning>3 + 4 = 7</reasoning> <answer>7</answer>"
NO_FORMAT = "The answer is 7."


def as_chat(text: str) -> list:
    """Wrap text the way TRL passes completions for conversational datasets."""
    return [{"role": "assistant", "content": text}]


# 1. Extraction helpers
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("1,234", "1234"),
        ("$18.", "18"),
        ("The answer is 7 apples", "7"),
        ("3.50", "3.5"),
        ("42.0", "42"),
        ("-5", "-5"),
        ("no number here", None),
        (None, None),
    ],
)
def test_normalise_number(raw, expected):
    assert normalise_number(raw) == expected


def test_extract_hash_answer():
    solution = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n48+24 = <<48+24=72>>72\n#### 72"
    assert extract_hash_answer(solution) == "72"
    assert extract_hash_answer("#### 1,000") == "1000"
    assert extract_hash_answer("no marker") is None


def test_extract_xml_answer():
    assert extract_xml_answer(PERFECT) == "7"
    assert extract_xml_answer(SLOPPY) == "7"
    assert extract_xml_answer("<answer>\n12") == "12"  # missing closing tag
    assert extract_xml_answer(NO_FORMAT) == ""
    # the last <answer> block wins
    assert extract_xml_answer("<answer>1</answer> ... <answer>2</answer>") == "2"


def test_completion_to_text_handles_both_formats():
    assert completion_to_text("abc") == "abc"
    assert completion_to_text(as_chat("abc")) == "abc"
    assert completion_to_text([]) == ""


# 2. Reward functions
def test_correctness_reward():
    completions = [as_chat(PERFECT), as_chat(SLOPPY), as_chat(NO_FORMAT), as_chat(PERFECT.replace("7", "8"))]
    rewards = correctness_reward_func(prompts=[None] * 4, completions=completions, answer=["7"] * 4)
    # NO_FORMAT has the right number but not inside <answer>, so no reward
    assert rewards == [2.0, 2.0, 0.0, 0.0]


def test_correctness_reward_normalises_numbers():
    completion = as_chat(XML_COT_FORMAT.format(reasoning="...", answer="$1,200.00"))
    assert correctness_reward_func(None, [completion], answer=["1200"]) == [2.0]


def test_int_reward():
    completions = [PERFECT, XML_COT_FORMAT.format(reasoning="r", answer="7 apples"), NO_FORMAT]
    assert int_reward_func(None, completions) == [0.5, 0.0, 0.0]


def test_strict_and_soft_format_rewards():
    completions = [as_chat(PERFECT), as_chat(SLOPPY), as_chat(NO_FORMAT)]
    assert strict_format_reward_func(None, completions) == [0.5, 0.0, 0.0]
    assert soft_format_reward_func(None, completions) == [0.5, 0.5, 0.0]


def test_count_xml():
    assert count_xml(PERFECT) == pytest.approx(0.5)
    assert count_xml(NO_FORMAT) == 0.0
    # trailing text after </answer> is penalised
    assert count_xml(PERFECT + "some extra rambling") < count_xml(PERFECT)
    assert xmlcount_reward_func(None, [as_chat(PERFECT)]) == [pytest.approx(0.5)]


def test_all_reward_funcs_return_one_float_per_completion():
    completions = [as_chat(PERFECT), as_chat(SLOPPY), as_chat(NO_FORMAT)]
    for func in REWARD_FUNCS:
        rewards = func(prompts=[None] * 3, completions=completions, answer=["7"] * 3)
        assert len(rewards) == 3
        assert all(isinstance(r, float) for r in rewards)


def test_perfect_completion_gets_maximal_total_reward():
    totals = {}
    for name, text in [("perfect", PERFECT), ("sloppy", SLOPPY), ("none", NO_FORMAT)]:
        totals[name] = sum(f(None, [as_chat(text)], answer=["7"])[0] for f in REWARD_FUNCS)
    assert totals["perfect"] == pytest.approx(0.5 + 0.5 + 0.5 + 0.5 + 2.0)
    assert totals["perfect"] > totals["sloppy"] > totals["none"]


# 3. Dataset and evaluation helpers
def test_build_prompt_and_format_example():
    messages = build_prompt("  What is 3 + 4?  ")
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1] == {"role": "user", "content": "What is 3 + 4?"}

    row = format_gsm8k_example({"question": "What is 3 + 4?", "answer": "3+4=7\n#### 7"})
    assert row["answer"] == "7"
    assert row["prompt"][-1]["content"] == "What is 3 + 4?"


def test_score_responses():
    metrics = score_responses([PERFECT, SLOPPY, NO_FORMAT, PERFECT], ["7", "8", "7", "7"])
    assert metrics == {"accuracy": 0.5, "format_rate": 0.75, "n": 4}
    assert score_responses([], [])["n"] == 0
    with pytest.raises(ValueError):
        score_responses(["a"], [])
