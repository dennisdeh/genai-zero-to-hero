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
    threshold_long_sentence: int = 700,
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

    The cleaning is done in two stages: The first stage splits the article into sentences and
    does a regex-based cleaning to remove unwanted characters and patterns. The second stage
    further refines the sentences.

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
    # assertions
    assert isinstance(language, str), "language must be a string"
    assert isinstance(name, str), "name must be a string"
    assert isinstance(
        threshold_long_sentence, int
    ), "threshold_long_sentence must be an integer"
    assert isinstance(
        threshold_short_article, int
    ), "threshold_short_article must be an integer"
    assert isinstance(list_sentences, bool), "list_sentences must be a boolean"
    assert isinstance(skip_short, bool), "skip_short must be a boolean"
    assert isinstance(
        save_skipped_examples, bool
    ), "save_skipped_examples must be a boolean"

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
        # Get article name and text
        t = ls_article_names[n]
        article = ls_text_raw[n]
        if not article or len(article.strip()) == 0:
            continue
        # prepare format of output dictionary
        d_articles[t] = []

        # First split of article into sentences robustly using nltk and store in a dictionary
        sentences = nltk.sent_tokenize(article, language=language)
        s1 = ""
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
                # Remove accents/diacritics by normalising the text
                s = unicodedata.normalize("NFD", s)
                s = "".join([c for c in s if not unicodedata.combining(c)])
                # remove leading and trailing whitespace
                s = s.strip()
                # add the cleaned sentence to the first article cleaning step
                if len(s1) == 0:
                    s1 = s
                else:
                    s1 += " " + s

        # Second split of article into sentences robustly using nltk and store in a dictionary
        sentences = nltk.sent_tokenize(s1, language=language)
        for s in sentences:
            if s.strip():
                s = s.strip()
                # some regular expressions cleaning
                if (
                    s == "."
                    or s == ","
                    or s.find(" ,") > -1
                    or s.find(" .") > -1
                    or s.find(" :") > -1
                    or s.find(" ;") > -1
                    or s.find(" !") > -1
                    or s.find(" ?") > -1
                    or s.find("()") > -1
                    or s.find("''") > -1
                    or s.find('""') > -1
                    or any(item in s[-4:] for item in ["e.g.", ")."])
                ):
                    continue
                # check that the sentence is not just a single word
                if s.count(" ") <= 1:
                    continue
                # try to remove sentences involving dates that originates from lists: check when such a list begins
                """
                 This pattern covers:
                 1. Month Day range (No Year): `"April 26/27:"`
                 2. Month Day, Year: `"August 1 1914: "`
                 3. Year First: `"1961 - "` or `"1970, December 15 – "`
                 4. Historical Years: `"December 25, 800 - "`
                 """
                if (
                    re.match(
                        r"^(?:(?:[A-Z][a-z]+ \d{1,2}(?:[/\-]\d{1,2})?[, ]+\d{1,4})|"
                        r"(?:\d{1,4}(?:,\s+[A-Z][a-z]+ \d{1,2}(?:[/\-]\d{1,2})?)?))\s*[\-–:]\s*",
                        s,
                    )
                    is not None
                ):
                    break
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


def dc_finance_qna(n_samples: int = None, threshold_skip_long_text: int = 700) -> dict:
    """
    Processes a financial Q&A dataset by filtering and preparing the data according to
    specified conditions, such as sample size and text length thresholds.

    This function collects data from the "finance-alpaca" dataset, performs filtering
    to eliminate duplicates, cleans invalid entries (e.g., overly long or empty content),
    and returns the processed dataset as a dictionary.

    :param n_samples: The number of samples to include in the dataset. If None, all
        available samples are processed.
    :type n_samples: int, optional
    :param threshold_skip_long_text: The maximum allowable length for both questions
        and answers. Entries exceeding this threshold are removed during preprocessing.
    :type threshold_skip_long_text: int
    :return: A dictionary mapping questions to their corresponding answers, where only
        valid entries (filtered and cleaned) are included.
    :rtype: dict
    """
    # 0: Initialisation and data collection
    print("\n\n *** 1. Data collection and preparation *** ")
    # assertions
    assert isinstance(n_samples, (int, type(None))), "n_samples must be an integer"
    assert isinstance(
        threshold_skip_long_text, int
    ), "threshold_skip_long_text must be an integer"
    assert threshold_skip_long_text > 0, "threshold_skip_long_text must be positive"

    # 1: Data collection
    # Use the finance-alpaca dataset (this contains only data labelled "train";
    # we take the "instruction" and "output" columns
    ds = load_dataset("gbharti/finance-alpaca", split="train")
    if n_samples is not None:
        df = ds.to_pandas()[:n_samples]
    else:
        df = ds.to_pandas()
    df = df[["instruction", "output"]].dropna()
    df = df.set_index("instruction")
    # remove duplicated questions
    df = df[~df.index.duplicated(keep="first")]
    # convert to dict so that the keys are questions and the values are answers
    d = dict(zip(df.index, df["output"]))

    # 2: Cleaning and preprocessing of questions and answers
    n_long_questions = 0
    n_long_answers = 0
    n_empty_questions = 0
    n_empty_answers = 0
    for k, v in d.copy().items():
        if len(k) > threshold_skip_long_text:
            n_long_questions += 1
            del d[k]
            continue
        if len(v) > threshold_skip_long_text:
            n_long_answers += 1
            del d[k]
            continue
        if len(k) == 0:
            n_empty_questions += 1
            del d[k]
            continue
        if len(v) == 0:
            n_empty_answers += 1
            del d[k]
            continue
    print(f"Total QnAs after preprocessing: {len(d)}")
    print(f"   Skipped long questions: {n_long_questions}")
    print(f"   Skipped long answers: {n_long_answers}")
    print(f"   Skipped empty questions: {n_empty_questions}")
    print(f"   Skipped empty answers: {n_empty_answers}")

    return d


if __name__ == "__main__2":
    # Wiki dataset
    d, _ = dc_wiki(n_samples=100, list_sentences=False)
    # Finance Q&A dataset
    d = dc_finance_qna(n_samples=100)
