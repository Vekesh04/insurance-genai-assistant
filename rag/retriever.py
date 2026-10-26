from __future__ import annotations

import re
from dataclasses import dataclass, asdict

import numpy as np

from rag.bm25 import BM25Index, tokenize
from rag.vector_store import load_vector_store


@dataclass
class RetrievalResult:
    chunk_id: str
    product_name: str
    policy_type: str
    section_number: str
    section_title: str
    text: str
    source_file: str
    vector_score: float
    bm25_score: float
    combined_score: float

    def to_dict(self):
        return asdict(self)


def _minmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo, hi = float(values.min()), float(values.max())
    if abs(hi - lo) < 1e-12:
        return np.ones_like(values) if hi > 0 else np.zeros_like(values)
    return (values - lo) / (hi - lo)


class HybridPolicyRetriever:
    def __init__(self):
        self.chunks, self.vectors, self.backend, self.info = load_vector_store()
        self.bm25 = BM25Index([c.text for c in self.chunks])

    def search(self, query: str, top_k: int = 5, product_name: str | None = None) -> list[RetrievalResult]:
        qvec = self.backend.transform([query]).astype("float32")
        qnorm = np.linalg.norm(qvec, axis=1, keepdims=True)
        qnorm[qnorm == 0] = 1.0
        qvec = qvec / qnorm
        vector_scores = (self.vectors @ qvec[0]).astype(float)
        bm25_scores = np.asarray(self.bm25.scores(query), dtype=float)

        combined = 0.62 * _minmax(vector_scores) + 0.38 * _minmax(bm25_scores)
        qtokens = set(tokenize(query))

        for i, chunk in enumerate(self.chunks):
            # Metadata filtering when product is explicitly supplied.
            if product_name and product_name.lower() not in chunk.product_name.lower():
                combined[i] = -1
                continue
            # Reranking boosts exact product mentions and section-title relevance.
            if chunk.product_name.lower() in query.lower():
                combined[i] += 0.22
            title_tokens = set(tokenize(chunk.section_title))
            overlap = len(qtokens & title_tokens)
            if overlap:
                combined[i] += min(0.18, 0.06 * overlap)

        order = np.argsort(-combined)
        results = []
        for idx in order:
            if combined[idx] < 0:
                continue
            c = self.chunks[int(idx)]
            results.append(
                RetrievalResult(
                    chunk_id=c.chunk_id,
                    product_name=c.product_name,
                    policy_type=c.policy_type,
                    section_number=c.section_number,
                    section_title=c.section_title,
                    text=c.text,
                    source_file=c.source_file,
                    vector_score=float(vector_scores[idx]),
                    bm25_score=float(bm25_scores[idx]),
                    combined_score=float(combined[idx]),
                )
            )
            if len(results) >= top_k:
                break
        return results
