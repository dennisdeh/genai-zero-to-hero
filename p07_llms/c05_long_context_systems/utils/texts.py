"""
Deterministic synthetic texts for the long-context experiments (no downloads needed).

- `make_haystack`: plausible but irrelevant "filler" text of a given token length,
- `make_needle` / `insert_needle`: a unique fact placed at a relative depth of the haystack,
- `make_long_report`: a long fictional annual report with a list of key facts, used to measure how many
  facts survive different summarisation strategies.
"""

import random
from typing import Callable

from p07_llms.c05_long_context_systems.utils.tokens import count_tokens

_COMPANIES = [
    "Nordlys Energy",
    "Kattegat Shipping",
    "Vestas Robotics",
    "Hygge Retail",
    "Fjord Analytics",
    "Bornholm Foods",
    "Skagen Pharma",
    "Odin Insurance",
    "Limfjord Logistics",
    "Aros Software",
]
_REGIONS = [
    "the Nordics",
    "Germany",
    "the Benelux",
    "the UK",
    "North America",
    "Asia",
    "Southern Europe",
]
_METRICS = [
    "revenue",
    "operating profit",
    "order intake",
    "free cash flow",
    "customer churn",
    "gross margin",
]
_DIRECTIONS = [
    "increased",
    "decreased",
    "remained broadly stable",
    "recovered",
    "softened",
]
_TEMPLATES = [
    "In {region}, {company} reported that {metric} {direction} compared with the previous quarter.",
    "Analysts noted that {company}'s {metric} in {region} {direction}, in line with the wider market.",
    "Management at {company} said the outlook for {region} depends on interest rates and energy prices.",
    "{company} continued its cost programme, focusing on procurement and logistics in {region}.",
    "The board of {company} discussed capital allocation, dividends and possible acquisitions in {region}.",
    "Market conditions in {region} were described as competitive, with pressure on prices and volumes.",
]
_COLOURS = ["blue", "amber", "crimson", "silver", "jade", "violet", "copper", "ivory"]
_ANIMALS = ["falcon", "otter", "lynx", "heron", "badger", "orca", "raven", "moose"]


def _filler_sentence(rng: random.Random) -> str:
    return rng.choice(_TEMPLATES).format(
        company=rng.choice(_COMPANIES),
        region=rng.choice(_REGIONS),
        metric=rng.choice(_METRICS),
        direction=rng.choice(_DIRECTIONS),
    )


def make_haystack(
    n_tokens: int,
    seed: int = 0,
    counter: Callable[[str], int] = count_tokens,
) -> list[str]:
    """
    Return a list of filler sentences with (about) n_tokens tokens in total.
    """
    rng = random.Random(seed)
    sentences: list[str] = []
    total = 0
    while total < n_tokens:
        sentence = _filler_sentence(rng)
        sentences.append(sentence)
        total += counter(sentence) + 1  # +1 for the joining space
    return sentences


def make_needle(seed: int) -> tuple[str, str, str]:
    """
    Return (needle sentence, question, expected answer) with a unique passphrase per seed.
    """
    rng = random.Random(10_000 + seed)
    passphrase = f"{rng.choice(_COLOURS)}-{rng.choice(_ANIMALS)}-{rng.randint(10, 99)}"
    needle = f"The secret passphrase for the archive room in Aarhus is {passphrase}."
    question = "What is the secret passphrase for the archive room in Aarhus? Answer with the passphrase only."
    return needle, question, passphrase


def insert_needle(sentences: list[str], needle: str, depth: float) -> str:
    """
    Insert the needle at a relative depth (0.0 = beginning, 1.0 = end) and join to one text.
    """
    position = round(depth * len(sentences))
    return " ".join(sentences[:position] + [needle] + sentences[position:])


def make_long_report(
    n_sections: int = 12, sentences_per_section: int = 25, seed: int = 1
) -> tuple[str, list[str]]:
    """
    Build a long fictional annual report. Each section contains one key fact (a specific number) buried in
    filler text. Returns (report text, list of key fact strings that a good summary should keep).
    """
    rng = random.Random(seed)
    company = "Aros Software A/S"
    topics = [
        ("Revenue", "Group revenue reached DKK {v} million"),
        ("Profitability", "The EBIT margin was {v} percent"),
        ("Employees", "The number of employees at year end was {v}"),
        ("Dividend", "The board proposes a dividend of DKK {v} per share"),
        ("Cloud", "Cloud subscriptions grew to {v} thousand seats"),
        ("Acquisitions", "The company acquired Limfjord Data for DKK {v} million"),
        ("Research", "Research and development spending was DKK {v} million"),
        ("Customers", "The largest customer accounted for {v} percent of revenue"),
        ("Sustainability", "Scope 1 and 2 emissions fell to {v} tonnes of CO2"),
        ("Outlook", "Revenue guidance for next year is DKK {v} million"),
        ("Capital", "Net interest-bearing debt was DKK {v} million"),
        ("Risk", "A cyber incident caused a {v}-hour outage of the payroll system"),
    ]
    sections, facts = [], []
    # unique 3-digit values, so they cannot be confused with section numbers or with each other
    values = rng.sample(range(101, 980), n_sections)
    for i, value in enumerate(values):
        title, fact_template = topics[i % len(topics)]
        fact = fact_template.format(v=value) + "."
        filler = [_filler_sentence(rng) for _ in range(sentences_per_section)]
        filler.insert(rng.randint(0, len(filler)), fact)
        sections.append(f"Section {i + 1}: {title}. " + " ".join(filler))
        facts.append(str(value))
    report = f"Annual report {company}.\n\n" + "\n\n".join(sections)
    return report, facts
