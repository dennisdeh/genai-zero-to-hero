"""
A small hand-written evaluation set: finance and machine learning questions with short reference answers.

Each item has a question, a canonical reference answer and optional aliases (other acceptable spellings).
Short, unambiguous references make automatic metrics such as exact match meaningful. Real evaluation sets
should be larger (hundreds of items), drawn from the actual use case and versioned like code.
"""

EVAL_SET: list[dict] = [
    {
        "id": "fin-01",
        "category": "finance",
        "question": "What does the abbreviation ETF stand for?",
        "reference": "exchange-traded fund",
        "aliases": ["exchange traded fund"],
    },
    {
        "id": "fin-02",
        "category": "finance",
        "question": "What does the P/E ratio stand for?",
        "reference": "price-to-earnings ratio",
        "aliases": ["price to earnings ratio", "price/earnings ratio"],
    },
    {
        "id": "fin-03",
        "category": "finance",
        "question": "What is the name of Denmark's central bank?",
        "reference": "Danmarks Nationalbank",
        "aliases": ["Nationalbanken"],
    },
    {
        "id": "fin-04",
        "category": "finance",
        "question": "Which international regulatory framework for bank capital was agreed after the 2008 financial "
        "crisis by the Basel Committee?",
        "reference": "Basel III",
        "aliases": [],
    },
    {
        "id": "fin-05",
        "category": "finance",
        "question": "An investment grows from 100 to 121 over two years. What is the annual compound growth rate "
        "in percent?",
        "reference": "10%",
        "aliases": ["10", "10 percent"],
    },
    {
        "id": "fin-06",
        "category": "finance",
        "question": "In which year were euro banknotes and coins introduced?",
        "reference": "2002",
        "aliases": [],
    },
    {
        "id": "ml-01",
        "category": "ml",
        "question": "Which activation function is defined as max(0, x)?",
        "reference": "ReLU",
        "aliases": ["rectified linear unit"],
    },
    {
        "id": "ml-02",
        "category": "ml",
        "question": "Which algorithm computes the gradients of a neural network's loss with respect to its weights?",
        "reference": "backpropagation",
        "aliases": ["back-propagation", "backprop"],
    },
    {
        "id": "ml-03",
        "category": "ml",
        "question": "Which metric is the harmonic mean of precision and recall?",
        "reference": "F1 score",
        "aliases": ["F1", "F-score", "F1-score"],
    },
    {
        "id": "ml-04",
        "category": "ml",
        "question": "What does RAG stand for in the context of large language models?",
        "reference": "retrieval-augmented generation",
        "aliases": ["retrieval augmented generation"],
    },
    {
        "id": "ml-05",
        "category": "ml",
        "question": "What is the name of the 2017 paper that introduced the transformer architecture?",
        "reference": "Attention Is All You Need",
        "aliases": [],
    },
    {
        "id": "ml-06",
        "category": "ml",
        "question": "In k-means clustering, what does k denote?",
        "reference": "the number of clusters",
        "aliases": ["number of clusters"],
    },
]
