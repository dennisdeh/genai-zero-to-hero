### Text
Text looks simple - a sequence of characters - but a computer sees a sequence of *bytes* that encode
Unicode *code points*. Most problems in text processing come from this gap:

- the same visible text can have different code points (`é` as one character or as `e` + a combining
  accent; the ligature `ﬁ` (U+FB01) versus `f` + `i`; full-width digits versus ASCII digits),
- invisible characters (zero-width spaces, byte order marks, control characters) break string
  comparisons and searches,
- typographic punctuation (`“ ” ‘ ’ –`) differs from its ASCII counterparts,
- one character can take 1-4 bytes in UTF-8, so "length" depends on what is counted.

The two subsections handle the cleaning of raw text and its tokenisation.

#### Text cleaning and normalisation (`text_cleaning/main_text_cleaning.py`)
The script applies a small pipeline to a few messy example documents:

- **HTML removal** with the standard library's `html.parser` (skipping `<script>` and `<style>`), which
  also decodes entities (`&amp;` → `&`, `&nbsp;` → no-break space) - exactly once, so the escaped text
  `&amp;lt;script&amp;gt;` becomes the literal text `&lt;script&gt;` and not a `<script>` tag. A line
  break is inserted at block-level tags (`<p>`, `<br>`, `<li>`, ...) so that words from neighbouring
  paragraphs are not glued together. The step only runs if the text looks like HTML (it contains a
  closing tag, a `<br>` or an entity); plain text such as `a<b and c>d` is left alone.
- **Unicode normalisation** with `unicodedata.normalize`. *NFC* composes characters without changing
  their meaning and is applied by default. *NFKC* additionally maps compatibility characters to their
  plain equivalents (U+FB01 `ﬁ` → `fi`, full-width digits → ASCII digits, the Roman numeral character
  U+2173 → `iv`). This helps search and matching, but it is *lossy* - `x²` becomes `x2`, `10⁶` becomes
  `106` and `½` becomes `1`, a fraction slash (U+2044) and `2` - so `clean_text` only uses it when
  asked to (`unicode_form="NFKC"`).
  Invisible format and control characters (Unicode categories `Cf`, `Cc`, e.g. zero-width spaces and
  byte order marks) are dropped, *except* the zero-width joiner and non-joiner: the joiner glues emoji
  sequences together (the family emoji is *man* + joiner + *woman* + joiner + *girl*), and the
  non-joiner is part of the correct spelling of many Persian words. Typographic quotes and dashes are
  mapped to ASCII.
- **Masking personal data** (PII) with regular expressions: URLs, emails, IBANs and phone numbers are
  replaced by placeholders such as `<EMAIL>`. The patterns try to balance misses against false alarms:
  an IBAN candidate (any case, optionally separated by spaces or hyphens) is only masked if it passes
  the ISO 13616 mod-97 checksum, and a phone number must have a phone-like structure (a `+` country
  code, an area code in parentheses, or at least three digit groups) and must not be a date such as
  `2024-01-15`, a version number or an ISBN. Regex masking is nevertheless *best effort*: unusual
  formats (`555-1234`, `12345678`) slip through, and names or addresses are not detected at all.
  Production systems combine rules with named-entity recognition models and still treat the output
  as potentially sensitive.
- **Whitespace normalisation**: runs of spaces, tabs and newlines become a single space.

Which further steps are appropriate depends on the task. For bag-of-words models and search indices,
*case folding* (`str.casefold`), accent stripping (decompose with NFD and drop the combining marks) and
stop word removal reduce the vocabulary. For language models these steps destroy information - the
model needs to see "US" vs. "us" and the word "not" - so only the meaning-preserving steps (with
NFC) are applied; `normalise_for_matching` uses NFKC, case folding and accent stripping.

**Deduplication** is one of the most effective cleaning steps for training data: duplicated documents
waste compute and make models memorise. Exact duplicates are found by hashing the normalised text.
Near duplicates are found by comparing sets of character $k$-grams ("shingles") with the Jaccard
similarity $J(A, B) = |A \cap B| / |A \cup B|$. For millions of documents the pairwise comparison is too
expensive; MinHash and locality-sensitive hashing approximate it efficiently.
