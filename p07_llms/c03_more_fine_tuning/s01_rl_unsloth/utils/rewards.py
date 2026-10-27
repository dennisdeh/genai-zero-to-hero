# Portions of this file are derived from:
#   Unsloth notebook "Qwen2.5_(3B)-GRPO.ipynb"
#   https://github.com/unslothai/notebooks/blob/main/nb/Qwen2.5_(3B)-GRPO.ipynb
#   Original author: Unsloth AI (Daniel Han, Michael Han and contributors); License: LGPL-3.0
#   (https://github.com/unslothai/notebooks/blob/main/LICENSE)
#   which in turn takes the data preparation and reward functions from the GRPO demo by Will Brown (@willccbb):
#   https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb (no separate licence stated in the gist)
# Derived parts: SYSTEM_PROMPT, XML_COT_FORMAT, extract_hash_answer, extract_xml_answer, count_xml and the
#   strict/soft format, integer and correctness reward functions.
# Changes: type hints and docstrings; support for both string and chat-message completions; numeric
#   normalisation of predicted and reference answers (normalise_number); negative integers accepted by
#   int_reward_func; optional trailing newline in the strict format; count_xml penalises only the text after the
#   last </answer>; dataset/evaluation helpers (build_prompt, format_gsm8k_example, score_responses) added.
"""
Prompt formatting, answer extraction and rule-based reward functions for GRPO on GSM8K.

Everything in this module is plain Python (only the standard library is used), so the logic can be
unit-tested on a CPU without torch, transformers, trl or unsloth installed:

    python -m pytest p07_llms/c03_more_fine_tuning/s01_rl_unsloth/test_rewards.py

The reward functions follow the TRL (>= 0.24) ``GRPOTrainer`` convention:

    def reward(prompts, completions, **kwargs) -> list[float]

* ``prompts`` and ``completions`` are lists of equal length (one entry per sampled completion, i.e.
  ``batch_size * num_generations`` entries).
* For a *conversational* dataset (a ``prompt`` column holding a list of chat messages), each completion is a
  list of messages such as ``[{"role": "assistant", "content": "..."}]``. For a plain-text dataset each
  completion is a string. Both are supported here.
* Every additional dataset column (here ``answer``) is passed as a keyword argument with one value per
  completion.
"""

import re
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence, Union

# A completion is either a string or a list of chat messages ({"role": ..., "content": ...}).
Completion = Union[str, List[Dict[str, str]]]

# 1. Prompt format the model is asked (and rewarded) to follow
SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
The answer must be a single number without units."""

XML_COT_FORMAT = """<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def build_prompt(question: str, system_prompt: str = SYSTEM_PROMPT) -> List[Dict[str, str]]:
    """Build the chat messages (system + user) for one maths question."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question.strip()},
    ]


# 2. Answer extraction helpers
def extract_hash_answer(text: str) -> Optional[str]:
    """
    Extract the reference answer from a GSM8K solution.

    GSM8K solutions end with a line like ``#### 1,234``; the number after ``####`` is the final answer.
    Returns the normalised number as a string, or None if no ``####`` marker is present.
    """
    if "####" not in text:
        return None
    return normalise_number(text.split("####")[-1])


def extract_xml_answer(text: str) -> str:
    """
    Return the content of the (last) ``<answer>...</answer>`` block, stripped.

    If the closing tag is missing, everything after the last ``<answer>`` is returned. If there is no
    ``<answer>`` tag at all, an empty string is returned.
    """
    if "<answer>" not in text:
        return ""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


# A number is an optional minus sign, then digits with an optional decimal part ("12", "12.", "12.5") or a bare
# decimal part (".5"), optionally followed by an exponent ("1e3", "2.5E-2"; at most 3 exponent digits).
_NUMBER_RE = re.compile(r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d{1,3}(?!\d))?")
# A comma between a digit and exactly three digits is a thousands separator ("1,234,567"); other commas
# (e.g. "3,4" or "3, 4") separate numbers and are kept.
_THOUSANDS_SEP_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


def normalise_number(text: Optional[str]) -> Optional[str]:
    """
    Normalise a numeric answer so that equivalent spellings compare equal.

    Removes thousands separators, currency/percent signs and surrounding text, maps the unicode minus sign
    (U+2212) to "-", then takes the last number in the string. The number is parsed exactly with Decimal and
    written in plain notation without trailing zeros or an exponent, so the result is idempotent
    (normalise_number(normalise_number(x)) == normalise_number(x)). Returns None if no number is found.

    Examples:
        "1,234" -> "1234", "$18." -> "18", "The answer is 7 apples" -> "7", "3.50" -> "3.5", ".5" -> "0.5",
        unicode minus + "3" -> "-3", "1e3" -> "1000", "1234567.5" -> "1234567.5"
    """
    if text is None:
        return None
    cleaned = text.replace("\u2212", "-")
    cleaned = _THOUSANDS_SEP_RE.sub("", cleaned).replace("$", "").replace("%", "").strip()
    matches = _NUMBER_RE.findall(cleaned)
    if not matches:
        return None
    value = Decimal(matches[-1]).normalize()
    if value == 0:
        value = Decimal(0)  # "-0" and "0.00" -> "0"
    return format(value, "f")


def completion_to_text(completion: Completion) -> str:
    """Return the generated text of a completion (string or list of chat messages)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        return str(completion[0].get("content", ""))
    return ""


# 3. Rule-based reward functions (TRL GRPOTrainer signature)
def correctness_reward_func(
    prompts: Sequence[Any], completions: Sequence[Completion], answer: Sequence[str], **kwargs: Any
) -> List[float]:
    """
    Verifiable reward: 2.0 if the number inside <answer> equals the reference answer, else 0.0.

    This is the reward that actually encodes the task; the format rewards below only shape the output.
    """
    responses = [completion_to_text(c) for c in completions]
    predicted = [normalise_number(extract_xml_answer(r)) for r in responses]
    references = [normalise_number(a) for a in answer]
    return [2.0 if p is not None and p == ref else 0.0 for p, ref in zip(predicted, references)]


def int_reward_func(prompts: Sequence[Any], completions: Sequence[Completion], **kwargs: Any) -> List[float]:
    """0.5 if the content of <answer> is a plain (possibly negative) integer, else 0.0."""
    responses = [completion_to_text(c) for c in completions]
    return [0.5 if re.fullmatch(r"-?\d+", extract_xml_answer(r)) else 0.0 for r in responses]


_STRICT_FORMAT_RE = re.compile(
    r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n?$", flags=re.DOTALL
)
_SOFT_FORMAT_RE = re.compile(r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>", flags=re.DOTALL)


def strict_format_reward_func(
    prompts: Sequence[Any], completions: Sequence[Completion], **kwargs: Any
) -> List[float]:
    """0.5 if the whole completion exactly follows the XML template (tags on their own lines), else 0.0."""
    responses = [completion_to_text(c) for c in completions]
    return [0.5 if _STRICT_FORMAT_RE.match(r) else 0.0 for r in responses]


def soft_format_reward_func(
    prompts: Sequence[Any], completions: Sequence[Completion], **kwargs: Any
) -> List[float]:
    """0.5 if the completion contains a <reasoning> block followed by an <answer> block, else 0.0."""
    responses = [completion_to_text(c) for c in completions]
    return [0.5 if _SOFT_FORMAT_RE.search(r) else 0.0 for r in responses]


def count_xml(text: str) -> float:
    """
    Dense format reward: 0.125 for each of the four tags appearing exactly once (on its own line),
    minus 0.001 per character of text after the last </answer> (surrounding whitespace ignored), to discourage
    rambling after the answer. A trailing newline after </answer> is therefore not penalised, and neither is any
    text before </answer> (e.g. a long reasoning).
    """
    score = 0.0
    if text.count("<reasoning>\n") == 1:
        score += 0.125
    if text.count("\n</reasoning>\n") == 1:
        score += 0.125
    if text.count("\n<answer>\n") == 1:
        score += 0.125
    if text.count("\n</answer>") == 1:
        score += 0.125
    if "</answer>" in text:
        tail = text.rsplit("</answer>", 1)[1]
        score -= len(tail.strip()) * 0.001
    return score


def xmlcount_reward_func(prompts: Sequence[Any], completions: Sequence[Completion], **kwargs: Any) -> List[float]:
    """Partial credit per correctly placed XML tag (see count_xml)."""
    return [count_xml(completion_to_text(c)) for c in completions]


REWARD_FUNCS = [
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func,
]


# 4. Dataset and evaluation helpers
def format_gsm8k_example(example: Dict[str, str], system_prompt: str = SYSTEM_PROMPT) -> Dict[str, Any]:
    """Map a raw GSM8K row ({"question", "answer"}) to the GRPO format ({"prompt", "answer"})."""
    return {
        "prompt": build_prompt(example["question"], system_prompt=system_prompt),
        "answer": extract_hash_answer(example["answer"]),
    }


def score_responses(responses: Sequence[str], references: Sequence[str]) -> Dict[str, float]:
    """
    Compute simple evaluation metrics for a list of generated responses.

    Returns:
        dict with "accuracy" (fraction of correct numeric answers), "format_rate" (fraction following the
        soft XML format) and "n" (number of examples).
    """
    if len(responses) != len(references):
        raise ValueError("responses and references must have the same length")
    n = len(responses)
    if n == 0:
        return {"accuracy": 0.0, "format_rate": 0.0, "n": 0}
    correct = correctness_reward_func(None, list(responses), answer=list(references))
    formatted = soft_format_reward_func(None, list(responses))
    return {
        "accuracy": sum(1 for c in correct if c > 0) / n,
        "format_rate": sum(1 for f in formatted if f > 0) / n,
        "n": n,
    }
