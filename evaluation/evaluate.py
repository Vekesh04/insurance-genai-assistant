from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from config import DATA_DIR
from rag.rag_service import PolicyRAGService


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def token_recall(expected: str, answer: str) -> float:
    exp = set(normalize(expected).split())
    got = set(normalize(answer).split())
    if not exp:
        return 1.0
    return len(exp & got) / len(exp)


def load_eval_questions() -> pd.DataFrame:
    path = DATA_DIR / "evaluation_questions.csv"
    df = pd.read_csv(path)
    # Supplied file uses numeric column names; normalize without changing source data.
    if list(df.columns)[:5] == ["0", "1", "2", "3", "4"]:
        df = df.rename(columns={"0": "evaluation_id", "1": "product_name", "2": "question", "3": "expected_answer", "4": "topic"})
    return df


def run_evaluation() -> dict:
    service = PolicyRAGService()
    rows = []
    for _, row in load_eval_questions().iterrows():
        result = service.answer(str(row["question"]), product_name=str(row["product_name"]), use_llm=False)
        score = token_recall(str(row["expected_answer"]), result["answer"])
        rows.append({
            "evaluation_id": row["evaluation_id"],
            "product_name": row["product_name"],
            "question": row["question"],
            "expected_answer": row["expected_answer"],
            "generated_answer": result["answer"],
            "token_recall": round(score, 3),
            "passed": score >= 0.35,
        })
    report = {
        "total": len(rows),
        "passed": sum(1 for r in rows if r["passed"]),
        "average_token_recall": round(sum(r["token_recall"] for r in rows) / max(len(rows), 1), 3),
        "results": rows,
    }
    out = Path(__file__).resolve().parent / "evaluation_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run_evaluation(), indent=2))
