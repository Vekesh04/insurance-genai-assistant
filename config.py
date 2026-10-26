from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
POLICY_DOCS_DIR = DATA_DIR / "policy_documents"
DATABASE_DIR = BASE_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "insurance.db"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

CSV_TABLES = {
    "customers.csv": "customers",
    "policies.csv": "policies",
    "claims.csv": "claims",
    "claim_documents.csv": "claim_documents",
    "hospitals.csv": "hospitals",
    "customer_interactions.csv": "customer_interactions",
}

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b").strip()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")

TOP_K = int(os.getenv("RAG_TOP_K", "4"))
MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "1000"))
