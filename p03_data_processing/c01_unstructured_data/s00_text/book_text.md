### Text
Text looks simple - a sequence of characters - but a computer sees a sequence of *bytes* that encode
Unicode *code points*. Most problems in text processing come from this gap:

- the same visible text can have different code points (`é` as one character or as `e` + a combining
  accent; the ligature `ﬁ` versus `f` + `i`; full-width digits versus ASCII digits),
- invisible characters (zero-width spaces, byte order marks, control characters) break string
  comparisons and searches,
- typographic punctuation (`“ ” ‘ ’ –`) differs from its ASCII counterparts,
- one character can take 1-4 bytes in UTF-8, so "length" depends on what is counted.

The two subsections handle the cleaning of raw text and its tokenisation.

#### Text cleaning and normalisation (`text_cleaning/main_text_cleaning.py`)
The script applies a small pipeline to a few messy example documents:

- **HTML removal** with the standard library's `html.parser` (skipping `<script>` and `<style>`),
  followed by `html.unescape` (`&amp;` → `&`, `&nbsp;` → no-break space).
- **Unicode normalisation** with `unicodedata.normalize`. *NFC* composes characters without changing
  their meaning and should always be applied; *NFKC* additionally maps compatibility characters to
  their plain equivalents (`ﬁ` → `fi`, full-width digits → ASCII digits, the Roman numeral character U+2173 → `iv`). Format and control characters
  (Unicode categories `Cf`, `Cc`) are dropped and typographic quotes and dashes are mapped to ASCII.
- **Masking personal data** (PII) with regular expressions: emails, URLs, IBAN-like account numbers and
  phone numbers are replaced by placeholders such as `<EMAIL>`. Regular expressions are a pragmatic
  baseline; production systems combine them with named-entity recognition models.
- **Whitespace normalisation**: runs of spaces, tabs and newlines become a single space.

Which further steps are appropriate depends on the task. For bag-of-words models and search indices,
*case folding* (`str.casefold`), accent stripping (decompose with NFD and drop the combining marks) and
stop word removal reduce the vocabulary. For language models these steps destroy information - the
model needs to see "US" vs. "us" and the word "not" - so only the meaning-preserving steps are applied.

**Deduplication** is one of the most effective cleaning steps for training data: duplicated documents
waste compute and make models memorise. Exact duplicates are found by hashing the normalised text.
Near duplicates are found by comparing sets of character $k$-grams ("shingles") with the Jaccard
similarity $J(A, B) = |A \cap B| / |A \cup B|$. For millions of documents the pairwise comparison is too
expensive; MinHash and locality-sensitive hashing approximate it efficiently.
