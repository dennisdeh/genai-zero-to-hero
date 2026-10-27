"""
Reference-based metrics for short free-text answers.

- normalise_answer: SQuAD-style normalisation (lower case, no punctuation, no articles, single spaces), but
  numbers are kept intact and canonicalised ('10.0' -> '10', '1,000' -> '1000'), so '1.0%' != '10%',
- exact_match: 1 if the normalised prediction equals a normalised reference,
- contains_answer: 1 if a normalised reference occurs in the normalised prediction (lenient; useful for
  chatty models that wrap the answer in a sentence),
- token_f1: harmonic mean of token precision and recall (SQuAD),
- rouge_l: F-measure based on the longest common subsequence (ROUGE-L, Lin 2004).
All metrics take the maximum over the reference and its aliases.
"""

import re
import string
from collections import Counter
from decimal import Decimal, InvalidOperation
from typing import Iterable

_ARTICLES = re.compile(r"\b(a|an|the)\b")
# keep '%' so that '10%' and '10' remain different answers
_PUNCTUATION = set(string.punctuation) - {"%"}


# A number: optional minus, digits with optional English thousands separators ('1,000,000'), optional decimal
# part. It must not be glued to letters ('f1', '10th') and must not continue with '.<digit>' or ',<digit>' (so
# version strings such as '1.2.3' or lists such as '1,2' are not read as numbers). Decimal commas ('3,5') are
# deliberately NOT supported: the evaluation set uses English number formatting.
_NUMBER = re.compile(
    r"(?<![\w.,])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![.,]?\d)(?!\w)"
)
_PUNCT_BETWEEN_DIGITS = re.compile(r"(?<=\d)[.,](?=\d)")


def _canonical_number(token: str) -> str:
    """
    '10.0' -> '10', '1,000' -> '1000', '0.50' -> '0.5' (exact decimal arithmetic, no float rounding).
    """
    try:
        value = Decimal(token.replace(",", ""))
    except InvalidOperation:
        return token  # cannot happen for _NUMBER matches; kept as a safe fallback
    if value == 0:
        value = Decimal(0)  # '-0' and '0.00' -> '0'
    return format(value.normalize(), "f")


def _normalise_text(text: str) -> str:
    # remaining separators between digits ('1,2', '1.2.3') become spaces instead of gluing the digits together
    text = _PUNCT_BETWEEN_DIGITS.sub(" ", text)
    text = text.replace("-", " ").replace("/", " ")
    return "".join(ch for ch in text if ch not in _PUNCTUATION)


def normalise_answer(text: str) -> str:
    """
    Lower-case, replace hyphens/slashes with spaces, remove punctuation and articles, collapse whitespace.
    Numbers are protected from punctuation removal and canonicalised, so '10.0%' == '10%' and
    '1,000' == '1000', while '1.0%' != '10%'.
    """
    text = text.lower()
    parts: list[str] = []
    position = 0
    for match in _NUMBER.finditer(text):
        parts.append(_normalise_text(text[position : match.start()]))
        parts.append(_canonical_number(match.group()))
        position = match.end()
    parts.append(_normalise_text(text[position:]))
    text = _ARTICLES.sub(" ", "".join(parts))
    return " ".join(text.split())


def _references(reference: str, aliases: Iterable[str] = ()) -> list[str]:
    return [reference, *aliases]


def exact_match(prediction: str, reference: str, aliases: Iterable[str] = ()) -> float:
    pred = normalise_answer(prediction)
    return float(
        any(pred == normalise_answer(r) for r in _references(reference, aliases))
    )


def contains_answer(
    prediction: str, reference: str, aliases: Iterable[str] = ()
) -> float:
    pred = f" {normalise_answer(prediction)} "
    return float(
        any(f" {normalise_answer(r)} " in pred for r in _references(reference, aliases))
    )


def _f1(pred_tokens: list[str], ref_tokens: list[str]) -> float:
    common = Counter(pred_tokens) & Counter(ref_tokens)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred_tokens)
    recall = n_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction: str, reference: str, aliases: Iterable[str] = ()) -> float:
    pred_tokens = normalise_answer(prediction).split()
    return max(
        _f1(pred_tokens, normalise_answer(r).split())
        for r in _references(reference, aliases)
    )


def _lcs_length(a: list[str], b: list[str]) -> int:
    """
    Length of the longest common subsequence (dynamic programming, O(len(a) * len(b))).
    """
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b, start=1):
            current.append(
                previous[j - 1] + 1
                if token_a == token_b
                else max(previous[j], current[j - 1])
            )
        previous = current
    return previous[-1]


def rouge_l(prediction: str, reference: str, aliases: Iterable[str] = ()) -> float:
    pred_tokens = normalise_answer(prediction).split()
    best = 0.0
    for r in _references(reference, aliases):
        ref_tokens = normalise_answer(r).split()
        lcs = _lcs_length(pred_tokens, ref_tokens)
        if lcs:
            precision, recall = lcs / len(pred_tokens), lcs / len(ref_tokens)
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


ALL_METRICS = {
    "exact_match": exact_match,
    "contains_answer": contains_answer,
    "token_f1": token_f1,
    "rouge_l": rouge_l,
}
