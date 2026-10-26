"""One-command setup for the local demo after dependencies are installed."""
from database.create_database import create_database
from rag.vector_store import build_vector_store

if __name__ == "__main__":
    print("1/2 Creating SQLite database...")
    create_database()
    print("2/2 Building policy retrieval index...")
    info = build_vector_store(prefer_semantic=True)
    print("Setup complete:", info)
