import json
from rag.vector_store import build_vector_store

if __name__ == "__main__":
    print(json.dumps(build_vector_store(prefer_semantic=True), indent=2))
