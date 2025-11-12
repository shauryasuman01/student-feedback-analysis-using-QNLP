import re
from typing import List, Tuple

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def load_data(path: str) -> pd.DataFrame:
    """Load CSV data into a pandas DataFrame."""
    df = pd.read_csv(path)
    return df


def basic_clean(text: str) -> str:
    """Perform a minimal, deterministic cleaning of a text string.

    - lowercases
    - removes non-alphanumeric characters (keeps spaces)
    - collapses repeated whitespace
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_texts(texts: List[str]) -> List[str]:
    """Clean a list of texts using `basic_clean`."""
    return [basic_clean(t) for t in texts]


def vectorize_texts(texts: List[str], max_features: int = 5000) -> Tuple:
    """Vectorize texts with CountVectorizer and return (X, vectorizer)."""
    vec = CountVectorizer(max_features=max_features)
    X = vec.fit_transform(texts)
    return X, vec
