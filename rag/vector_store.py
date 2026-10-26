from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from config import BASE_DIR
from rag.document_loader import PolicyChunk, load_all_policy_chunks
from rag.embeddings import TfidfEmbeddingBackend, create_embedding_backend

STORE_DIR = BASE_DIR / "vector_store"
META_PATH = STORE_DIR / "policy_chunks.json"
VECTORS_PATH = STORE_DIR / "policy_vectors.npy"
BACKEND_PATH = STORE_DIR / "embedding_backend.pkl"
INFO_PATH = STORE_DIR / "index_info.json"


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def build_vector_store(prefer_semantic: bool = True) -> dict:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    chunks = load_all_policy_chunks()
    texts = [c.text for c in chunks]
    backend = create_embedding_backend(prefer_semantic=prefer_semantic)
    vectors = _normalize(backend.fit_transform(texts))

    META_PATH.write_text(
        json.dumps([c.to_dict() for c in chunks], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    np.save(VECTORS_PATH, vectors)

    # Only sklearn TF-IDF needs its fitted vocabulary persisted. SentenceTransformer
    # can be re-instantiated from its model name.
    if isinstance(backend, TfidfEmbeddingBackend):
        with BACKEND_PATH.open("wb") as f:
            pickle.dump(backend, f)
    elif BACKEND_PATH.exists():
        BACKEND_PATH.unlink()

    info = {"chunks": len(chunks), "dimensions": int(vectors.shape[1]), "embedding_backend": backend.name}
    INFO_PATH.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def index_exists() -> bool:
    return META_PATH.exists() and VECTORS_PATH.exists() and INFO_PATH.exists()


def load_vector_store():
    if not index_exists():
        build_vector_store(prefer_semantic=True)

    info = json.loads(INFO_PATH.read_text(encoding="utf-8"))
    chunks = [PolicyChunk(**d) for d in json.loads(META_PATH.read_text(encoding="utf-8"))]
    vectors = np.load(VECTORS_PATH)

    backend_name = info["embedding_backend"]
    if backend_name == "tfidf-local":
        with BACKEND_PATH.open("rb") as f:
            backend = pickle.load(f)
    else:
        from rag.embeddings import SentenceTransformerEmbeddingBackend
        backend = SentenceTransformerEmbeddingBackend(backend_name)
    return chunks, vectors, backend, info


if __name__ == "__main__":
    print(json.dumps(build_vector_store(prefer_semantic=True), indent=2))
