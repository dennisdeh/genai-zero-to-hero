"""
A tiny, hand-written sentiment dataset with the same schema as "cornell-movie-review-data/rotten_tomatoes"
(columns 'text' and 'label', label 0 = negative, 1 = positive). It is used as an offline fallback so that
the processing examples can be run without access to the Hugging Face Hub.
"""

from datasets import ClassLabel, Dataset, DatasetDict, Features, Value

_POSITIVE = [
    "a warm , funny and beautifully acted film that stays with you .",
    "the best thriller of the year , tense from start to finish .",
    "smart writing and a terrific cast make this a joy to watch .",
    "an inventive , heartfelt story told with real craft .",
    "gorgeous to look at and surprisingly moving .",
    "a crowd-pleaser in the best sense of the word .",
    "the director's confident hand turns a simple premise into something special .",
    "charming , clever and never condescending to its audience .",
    "one of those rare sequels that improves on the original .",
    "a delightful surprise .",
    "funny , touching and wonderfully strange .",
    "the performances alone are worth the price of admission .",
    "it is a small film with a very big heart .",
    "a riveting documentary that raises important questions .",
    "the animation is stunning and the jokes land for kids and adults alike .",
    "great .",
]
_NEGATIVE = [
    "a dull , plodding mess that wastes a talented cast .",
    "the plot makes no sense and the dialogue is painful .",
    "two hours of my life i will never get back .",
    "loud , long and utterly forgettable .",
    "the jokes fall flat and the pacing is glacial .",
    "a lazy sequel that exists only to sell tickets .",
    "even the special effects cannot save this tired script .",
    "it wants to be profound but ends up merely pretentious .",
    "the characters are so thin you could read the newspaper through them .",
    "an ugly , cynical film with nothing to say .",
    "badly edited and worse acted .",
    "the twist is visible from the first scene .",
    "a promising idea buried under clichés .",
    "boring .",
    "it drags on and on without ever finding its rhythm .",
    "a film that confuses noise with excitement .",
]

FEATURES = Features({"text": Value("string"), "label": ClassLabel(names=["neg", "pos"])})


def offline_rotten_tomatoes() -> DatasetDict:
    """Return the tiny offline sentiment dataset as a DatasetDict with 'train' and 'test' splits."""
    texts = _POSITIVE + _NEGATIVE
    labels = [1] * len(_POSITIVE) + [0] * len(_NEGATIVE)
    ds = Dataset.from_dict({"text": texts, "label": labels}, features=FEATURES)
    return ds.train_test_split(test_size=0.25, stratify_by_column="label", seed=0)
