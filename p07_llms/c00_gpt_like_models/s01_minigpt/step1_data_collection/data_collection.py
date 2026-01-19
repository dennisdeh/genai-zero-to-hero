"""
In this script we will take a look at the data collection and preparation for training an LLM.
"""

from datasets import load_dataset
import nltk
import re
import unicodedata


def dc_wiki(
    n_samples: int = None,
    language: str = "english",
    name: str = "20231101.simple",
    threshold_long_sentence: int = 1000,
    threshold_short_article: int = 100,
    list_sentences: bool = False,
    skip_short: bool = True,
    save_skipped_examples: bool = True,
):
    """
    A function to process and clean a specified subset of articles from a Wikipedia dataset.
    It performs segmentation of articles into smaller sentence-based segments while
    optionally filtering sentences based on length and certain patterns. The function also
    provides the ability to skip and optionally save examples of filtered sentences.

    Parameters:
    n_samples: int, None
        Number of articles to process. If None, all articles in the dataset will be processed.
    language: str
        Language used for sentence tokenisation. Defaults to "english".
    name: str
        Name of the Wikipedia dataset to load. Defaults to "20231101.simple" (Simple English).
    threshold_long_sentence: int
        Maximum length of sentences allowed. Sentences exceeding this threshold are skipped.
        Defaults to 1000.
    threshold_short_article: int
        Minimum acceptable length for articles after cleaning. Articles shorter than this
        threshold will be removed. Defaults to 100.
    list_sentences: bool
        If True, returns segmented sentences as a list instead of a concatenated string for
        each article. Defaults to False.
    skip_short: bool
        If True, skips very short sentences (e.g., less than or equal to one space-separated
        word). Defaults to True.

    save_skipped_examples: bool
        If True, saves skipped examples in the output dictionary of skipped entries. Defaults
        to True.

    Return:
        A tuple containing two dictionaries:
             - A dictionary of cleaned articles where keys are article titles and values are
               either concatenated strings of cleaned text or lists of cleaned sentences
               depending on the value of `list_sentences`.
             - A dictionary of skipped examples where keys are article titles and values are lists
               of skipped sentences.
    """
    # 0: Initialisation and data collection
    print("\n\n *** 1. Data collection and preparation *** ")
    # Download the punctuation tokeniser form nltk if not already downloaded
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    # Set global parameters

    # Use the Wikipedia dataset (as of 2023-11-01); this contains only data labelled
    # "train"; we split it manually into train and validation sets), and we take the
    # "instruction" column
    ds = load_dataset(path="wikimedia/wikipedia", name=name, split="train")

    # Sample only n_samples if specified
    ls_text_raw = list(ds["text"])
    ls_article_names = list(ds["title"])
    assert len(ls_text_raw) == len(ls_article_names)

    if n_samples is not None and n_samples < len(ls_text_raw):
        ls_text_raw = ls_text_raw[:n_samples]
        ls_article_names = ls_article_names[:n_samples]
        print(f"Sampled {n_samples} articles from the dataset")
    else:
        print(f"Using all {len(ls_text_raw)} articles from the dataset")

    # 1: Cleaning and preprocessing of articles
    d_articles = {}
    n_segments = 0
    n_skipped_short = 0
    n_skipped_long = 0
    n_skipped_nn = 0
    n_skipped_numbers = 0
    n_skipped_punctuation = 0
    n_skipped_code = 0
    d_skipped = {}
    for n in range(len(ls_text_raw)):
        t = ls_article_names[n]
        article = ls_text_raw[n]
        if not article or len(article.strip()) == 0:
            continue

        # Split article into sentences robustly using nltk and store in a dictionary
        sentences = nltk.sent_tokenize(article, language=language)
        # prepare format of output dictionary
        d_articles[t] = []
        if save_skipped_examples:
            d_skipped[t] = []
        for s in sentences:
            if s.strip():
                s = s.strip()
                # skip very short sentences (e.g. "See also", "He was")
                if not skip_short:
                    if s.count(" ") <= 1:
                        n_skipped_short += 1
                        if save_skipped_examples:
                            d_skipped[t].append(s)
                        continue
                # skip very long sentences (e.g. References, lists, svg graphics, tables)
                if len(s) > threshold_long_sentence:
                    n_skipped_long += 1
                    if save_skipped_examples:
                        d_skipped[t].append(s)
                    continue
                # skip sentences that contains "\n\n"
                if "\n\n" in s:
                    n_skipped_nn += 1
                    if save_skipped_examples:
                        d_skipped[t].append(s)
                    continue
                # skip sentences that contains only numbers (e.g. "1. 2. 3.")
                if s.replace(".", "", 1).isdigit():
                    n_skipped_numbers += 1
                    if save_skipped_examples:
                        d_skipped[t].append(s)
                    continue
                # skip sentences that contains only punctuation (e.g. ". . .")
                if s.replace(".", "", 1).isalpha():
                    n_skipped_punctuation += 1
                    if save_skipped_examples:
                        d_skipped[t].append(s)
                    continue
                # skip sentences that can be identified as code (e.g. "```python")
                if "```" in s:
                    n_skipped_code += 1
                    if save_skipped_examples:
                        d_skipped[t].append(s)
                    continue

                # Regular expressions to be removed
                s = re.sub(r"^.*\n+", "", s, flags=re.DOTALL)
                s = re.sub(r"^.*?\n", "", s)
                # Remove content in brackets
                s = re.sub(r"\(.*?\)|\[.*?\]|\{.*?\}", "", s, flags=re.DOTALL)
                # replacement of multiple spaces with single spaces
                s = re.sub(r" +", " ", s)
                # Remove accents/diacritics
                s = unicodedata.normalize("NFD", s)
                s = "".join([c for c in s if not unicodedata.combining(c)])
                # remove leading and trailing whitespace
                s = s.strip()
                # add cleaned sentence to the dictionary for the current article
                d_articles[t].append(s)
        n_segments += len(d_articles[t])
        if list_sentences:
            pass
        else:
            d_articles[t] = " ".join(d_articles[t])
            if len(d_articles[t]) < threshold_short_article:
                del d_articles[t]

    if list_sentences:
        print(
            f"Text example from Article 0: "
            f"{d_articles[list(d_articles.keys())[0]][0:2]}"
        )
    else:
        print(
            f"Text example from Article 0: {d_articles[list(d_articles.keys())[0]][:100]}"
        )
    print(f"Total text segments after splitting: {n_segments}")
    print(f"   Skipped short sentences: {n_skipped_short}")
    print(f"   Skipped long sentences: {n_skipped_long}")
    print(f"   Skipped sentences containing double newlines: {n_skipped_nn}")
    print(f"   Skipped sentences containing only numbers: {n_skipped_numbers}")
    print(f"   Skipped sentences containing only punctuation: {n_skipped_punctuation}")
    print(f"   Skipped sentences containing code blocks: {n_skipped_code}")

    return d_articles, d_skipped


if __name__ == "__main__":
    d, _ = dc_wiki(n_samples=1000)
