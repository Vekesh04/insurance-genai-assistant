from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class EmbeddingBackend:
    name = "base"

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def transform(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class TfidfEmbeddingBackend(EmbeddingBackend):
    """Offline fallback vector embedding backend requiring no model download."""

    name = "tfidf-local"

    def __init__(self, max_features: int = 4096):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=max_features,
            sublinear_tf=True,
        )

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.vectorizer.fit_transform(texts).toarray().astype("float32")

    def transform(self, texts: list[str]) -> np.ndarray:
        return self.vectorizer.transform(texts).toarray().astype("float32")


class SentenceTransformerEmbeddingBackend(EmbeddingBackend):
    name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.name = model_name

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.transform(texts)

    def transform(self, texts: list[str]) -> np.ndarray:
        arr = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(arr, dtype="float32")


def create_embedding_backend(prefer_semantic: bool = True) -> EmbeddingBackend:
    if prefer_semantic:
        try:
            return SentenceTransformerEmbeddingBackend()
        except Exception:
            pass
    return TfidfEmbeddingBackend()
