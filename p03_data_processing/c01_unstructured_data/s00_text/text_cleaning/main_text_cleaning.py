"""
Cleaning and normalising raw text before tokenisation, search or model training.

Text scraped from websites, PDFs or user input contains HTML, inconsistent Unicode, odd whitespace,
personal data and duplicates. This script builds a small cleaning pipeline using only the standard
library and applies it to a handful of messy example documents:

1. HTML removal and entity unescaping (only for text that looks like HTML)
2. Unicode normalisation (NFC by default, NFKC as a lossy option), removal of invisible control/format
   characters, quote normalisation
3. masking of personal data (emails, phone numbers, IBANs with checksum validation) and URLs - regular
   expressions are a best-effort baseline, not a guarantee
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


BLOCK_TAGS = {"p", "br", "div", "li", "ul", "ol", "tr", "td", "th", "table", "h1", "h2", "h3", "h4", "h5", "h6",
              "section", "article", "header", "footer", "blockquote", "pre", "hr", "title"}
# heuristics for "this text is HTML": a closing tag, a void element such as <br>, a comment/doctype, or an entity
HTML_TAG = re.compile(r"</[a-zA-Z][\w-]*\s*>|<(?:br|hr|img|meta|link|input)\b[^<>]*>|<!(?:--|doctype)", re.I)
HTML_ENTITY = re.compile(r"&(?:[a-zA-Z][a-zA-Z0-9]*|#\d+|#x[0-9a-fA-F]+);")


class _TextExtractor(HTMLParser):
    """Collect the text content of an HTML document, skipping <script> and <style> blocks."""

    def __init__(self):
        super().__init__(convert_charrefs=True)  # decodes entities (&amp; -> &) exactly once
        self.parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self._skip = True
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self._skip = False
        if tag in BLOCK_TAGS:  # "<p>a</p><p>b</p>" -> "a\nb", not "ab"
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def strip_html(text: str) -> str:
    """
    Remove tags and decode entities (&amp; -> &, &nbsp; -> no-break space) - only if the text looks like HTML.

    Plain text such as "a<b and c>d" is returned unchanged; text with entities but without tags is
    unescaped once (so "&amp;lt;" becomes the literal text "&lt;", not "<").
    """
    if HTML_TAG.search(text):
        parser = _TextExtractor()
        parser.feed(text)
        parser.close()
        return "".join(parser.parts)
    if HTML_ENTITY.search(text):
        return html.unescape(text)
    return text


KEEP_FORMAT_CHARS = {"‌", "‍"}  # zero-width (non-)joiner: needed for emoji sequences and e.g. Persian


def normalise_unicode(text: str, form: str = "NFC") -> str:
    """
    Unicode normalisation.

    NFC (default) composes characters (e + combining accent -> é) without changing meaning. NFKC
    additionally maps "compatibility" characters to their plain form (ligature ﬁ -> fi, full-width ２ -> 2,
    ⅳ -> iv) - useful for search and matching, but *lossy*: x² -> x2, 10⁶ -> 106, ½ -> 1⁄2.
    Invisible format/control characters (zero-width space, byte order mark, ...) are removed, except the
    zero-width joiner and non-joiner; typographic quotes and dashes are mapped to ASCII.
    """
    text = unicodedata.normalize(form, text)
    text = "".join(ch for ch in text
                   if unicodedata.category(ch) not in {"Cf", "Cc"} or ch in "\n\t" or ch in KEEP_FORMAT_CHARS)
    return text.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                                         "–": "-", "—": "-", " ": " "}))


# --- personal data (best effort: regular expressions miss some PII and occasionally mask too much) ---
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL_PATTERN = re.compile(r"https?://\S+")
# IBAN candidates: country code, 2 check digits, then groups of up to 4 characters, optionally separated by a
# space or hyphen; only candidates that pass the ISO 13616 mod-97 checksum are masked
IBAN_PATTERN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ -]?[0-9A-Z]{4}){2,7}(?:[ -]?[0-9A-Z]{1,4})?\b", re.I)
# phone candidates: optional +country code, optional (area code), digit groups separated by a space or hyphen;
# must not continue a longer token (version numbers, ISBNs, decimals)
PHONE_PATTERN = re.compile(r"""
    (?<![\w.,+(-])                   # not the continuation of a longer token
    (?:\+\d{1,3}[ -]?)?              # country code, e.g. +45
    (?:\(\d{1,4}\)[ -]?)?            # area code in parentheses, e.g. (555)
    \d{1,15}(?:[ -]\d{1,4}){0,4}     # digit groups ...
    (?:[ -]\d{1,8})?                 # ... the last one may be longer (subscriber number)
    (?![\w,-]|\.\d)                  # not followed by more of the same token
""", re.X)
DATE_LIKE = re.compile(r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}-\d{1,2}-\d{4}")
THOUSANDS_LIKE = re.compile(r"\d{1,3}(?: \d{3})+")  # "1 250 000"


def is_valid_iban(candidate: str) -> bool:
    """ISO 13616 check: move the first 4 characters to the end, letters -> 10..35, number mod 97 must be 1."""
    s = re.sub(r"[ -]", "", candidate).upper()
    if not (15 <= len(s) <= 34 and s[:2].isalpha() and s[2:4].isdigit() and s.isalnum()):
        return False
    return int("".join(str(int(ch, 36)) for ch in s[4:] + s[:4])) % 97 == 1


def _mask_iban(match: re.Match) -> str:
    """Mask the longest prefix of the candidate (cut at a separator) that is a valid IBAN."""
    candidate = match.group()
    cuts = [len(candidate)] + [m.start() for m in reversed(list(re.finditer(r"[ -]", candidate)))]
    for cut in cuts:  # the regex may have swallowed a following word, e.g. "... 1234 from"
        if is_valid_iban(candidate[:cut]):
            return "<IBAN>" + candidate[cut:]
    return candidate


def _mask_phone(match: re.Match) -> str:
    """Mask a phone candidate if it has a phone-like structure (and is not a date or a formatted number)."""
    candidate = match.group()
    n_digits = sum(ch.isdigit() for ch in candidate)
    has_prefix = candidate.startswith(("+", "("))
    n_groups = len(re.findall(r"\d+", candidate))
    if (not 7 <= n_digits <= 15 or not (has_prefix or n_groups >= 3)
            or DATE_LIKE.fullmatch(candidate) or THOUSANDS_LIKE.fullmatch(candidate)):
        return candidate
    return "<PHONE>"


def mask_pii(text: str) -> str:
    """Replace URLs, emails, IBANs and phone numbers by placeholders (best effort, see the module notes)."""
    text = URL_PATTERN.sub("<URL>", text)
    text = EMAIL_PATTERN.sub("<EMAIL>", text)
    text = IBAN_PATTERN.sub(_mask_iban, text)
    text = PHONE_PATTERN.sub(_mask_phone, text)
    return text


def normalise_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs/newlines into single spaces and trim."""
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str, unicode_form: str = "NFC") -> str:
    """
    The generic cleaning pipeline (suitable before LLM tokenisation).

    With the default NFC it preserves the meaning of the text; ``unicode_form="NFKC"`` is a lossy option
    for search and matching.
    """
    return normalise_whitespace(mask_pii(normalise_unicode(strip_html(text), form=unicode_form)))


def strip_accents(text: str) -> str:
    """Remove diacritics: decompose (NFD) and drop the combining marks (café -> cafe)."""
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")


def normalise_for_matching(text: str) -> str:
    """Aggressive normalisation for search/deduplication: case folding, no accents, no punctuation."""
    text = strip_accents(clean_text(text, unicode_form="NFKC")).casefold()
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
    print(f"clean_text with NFKC (for search/matching): {clean_text(DOCUMENTS[5], unicode_form='NFKC')!r}")
    lossy = "x² + y², 10⁶ cells, ½ cup"
    print(f"but NFKC is lossy: {lossy!r} -> {unicodedata.normalize('NFKC', lossy)!r}")

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
