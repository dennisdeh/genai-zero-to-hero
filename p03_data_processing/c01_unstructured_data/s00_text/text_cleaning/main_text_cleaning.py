"""
Cleaning and normalising raw text before tokenisation, search or model training.

Text scraped from websites, PDFs or user input contains HTML, inconsistent Unicode, odd whitespace,
personal data and duplicates. This script builds a small cleaning pipeline using only the standard
library and applies it to a handful of messy example documents:

1. HTML removal and entity unescaping
2. Unicode normalisation (NFC vs. NFKC), removal of control/zero-width characters, quote normalisation
3. masking of personal data (emails, phone numbers, IBAN-like numbers) and URLs
4. whitespace normalisation
5. optional, task-dependent steps: case folding, accent stripping, stop word removal
6. exact and near-duplicate detection (character shingles + Jaccard similarity)

Note: for LLM pre-training one keeps casing, accents and stop words (the model needs them); aggressive
normalisation is for classical NLP such as bag-of-words models or search indices.

Run from the repository root:
    python -m p03_data_processing.c01_unstructured_data.s00_text.text_cleaning.main_text_cleaning
"""

import hashlib
import html
import re
import unicodedata
from html.parser import HTMLParser
from itertools import combinations

DOCUMENTS = [
    "<p>Our <b>savings account</b> pays 3&nbsp;% interest &amp; has no fees!</p><script>track()</script>",
    "Our savings account pays 3 % interest & has no fees!",
    "OUR SAVINGS ACCOUNT PAYS 3 % INTEREST & HAS NO FEES!!!",
    "Our savings account now pays 3.5 % interest and has no hidden fees.",
    "Contact   jane.doe@example.com or call +45 12 34 56 78 – visit https://example.com/offer",
    "The ﬁnancial report (ﬁscal year ２０２４) was published​ today.\t\t\n\nSee page ⅳ.",
    "Café vs. Café: identical on screen, different bytes",
    "Transfer the amount to DK50 0040 0440 1162 43 before Friday.",
    "“Smart quotes” and ‘apostrophes’ — typographic punctuation.",
]
STOP_WORDS = {"a", "an", "and", "the", "or", "of", "to", "in", "on", "is", "are", "was", "has", "our", "vs", "be"}


class _TextExtractor(HTMLParser):
    """Collect the text content of an HTML document, skipping <script> and <style> blocks."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        self._skip = tag in {"script", "style"}
        if tag in {"p", "br", "div", "li", "h1", "h2", "h3"}:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        self._skip = False

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def strip_html(text: str) -> str:
    """Remove tags and decode entities (&amp; -> &, &nbsp; -> no-break space)."""
    parser = _TextExtractor()
    parser.feed(text)
    return html.unescape("".join(parser.parts))


def normalise_unicode(text: str, form: str = "NFKC") -> str:
    """
    Unicode normalisation.

    NFC composes characters (e + combining accent -> é) without changing meaning; NFKC additionally maps
    "compatibility" characters to their plain form (ligature ﬁ -> fi, full-width ２ -> 2, ⅳ -> iv).
    Zero-width and other invisible format/control characters are removed, typographic quotes and dashes
    are mapped to ASCII.
    """
    text = unicodedata.normalize(form, text)
    text = "".join(ch for ch in text if unicodedata.category(ch) not in {"Cf", "Cc"} or ch in "\n\t")
    return text.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                                         "–": "-", "—": "-", " ": " "}))


def mask_pii(text: str) -> str:
    """Replace emails, URLs, IBAN-like account numbers and phone numbers by placeholders."""
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "<EMAIL>", text)
    text = re.sub(r"https?://\S+", "<URL>", text)
    text = re.sub(r"\b[A-Z]{2}\d{2}(?: ?[0-9A-Z]{4}){2,7}(?: ?[0-9A-Z]{1,4})?\b", "<IBAN>", text)
    text = re.sub(r"\+?\d{1,3}(?:[ -]?\d{2,4}){3,5}\b", "<PHONE>", text)
    return text


def normalise_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs/newlines into single spaces and trim."""
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """The generic, meaning-preserving cleaning pipeline (suitable before LLM tokenisation)."""
    return normalise_whitespace(mask_pii(normalise_unicode(strip_html(text))))


def strip_accents(text: str) -> str:
    """Remove diacritics: decompose (NFD) and drop the combining marks (café -> cafe)."""
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")


def normalise_for_matching(text: str) -> str:
    """Aggressive normalisation for search/deduplication: case folding, no accents, no punctuation."""
    text = strip_accents(clean_text(text)).casefold()
    return normalise_whitespace(re.sub(r"[^\w<>\s]", " ", text))


def remove_stop_words(text: str) -> str:
    """Drop very frequent function words (only useful for bag-of-words style models)."""
    return " ".join(w for w in text.split() if w not in STOP_WORDS)


def shingles(text: str, k: int = 5) -> set[str]:
    """Set of overlapping character k-grams."""
    return {text[i:i + k] for i in range(max(1, len(text) - k + 1))}


def jaccard(a: set, b: set) -> float:
    """Jaccard similarity |A n B| / |A u B|."""
    return len(a & b) / len(a | b) if a | b else 1.0


if __name__ == "__main__":
    # 1.-4. Generic cleaning
    print("=== generic cleaning (HTML, Unicode, PII, whitespace) ===")
    cleaned = [clean_text(d) for d in DOCUMENTS]
    for raw, clean in zip(DOCUMENTS, cleaned):
        print(f"raw:   {raw!r}\nclean: {clean!r}\n")

    # Unicode details: visually identical strings that differ in bytes
    a, b = "Café", "Café"
    print(f"'{a}' == '{b}': {a == b}; after NFC: {unicodedata.normalize('NFC', a) == unicodedata.normalize('NFC', b)}")
    print(f"NFC keeps the ligature: {unicodedata.normalize('NFC', 'ﬁscal')!r}, "
          f"NFKC resolves it: {unicodedata.normalize('NFKC', 'ﬁscal')!r}")

    # 5. Task-dependent normalisation
    print("\n=== aggressive normalisation for matching / bag-of-words ===")
    for d in DOCUMENTS[:4]:
        norm = normalise_for_matching(d)
        print(f"{norm!r} -> without stop words: {remove_stop_words(norm)!r}")

    # 6. Deduplication: exact (hash of the normalised text) and near duplicates (Jaccard of shingles)
    print("\n=== duplicate detection ===")
    hashes = [hashlib.sha1(normalise_for_matching(d).encode()).hexdigest()[:10] for d in DOCUMENTS]
    for i, j in combinations(range(len(DOCUMENTS)), 2):
        if hashes[i] == hashes[j]:
            print(f"exact duplicates after normalisation: documents {i} and {j}")
    sets = [shingles(normalise_for_matching(d)) for d in DOCUMENTS]
    for i, j in combinations(range(len(DOCUMENTS)), 2):
        sim = jaccard(sets[i], sets[j])
        if sim > 0.3 and hashes[i] != hashes[j]:
            print(f"near duplicates: documents {i} and {j} (Jaccard similarity of 5-gram shingles {sim:.2f})")
