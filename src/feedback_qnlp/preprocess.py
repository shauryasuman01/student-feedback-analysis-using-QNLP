import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer


def load_data(path):
    df = pd.read_csv(path)
    return df


def basic_clean(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_texts(texts):
    return [basic_clean(t) for t in texts]


def vectorize_texts(texts, max_features=5000):
    vec = CountVectorizer(max_features=max_features)
    X = vec.fit_transform(texts)
    return X, vec
