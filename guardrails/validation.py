from __future__ import annotations

import re
from dataclasses import dataclass

from config import MAX_QUESTION_LENGTH


@dataclass
class GuardrailResult:
    allowed: bool
    message: str | None = None


BLOCKED_PATTERNS = [
    r"ignore (all|any|the) previous instructions",
    r"reveal (the )?(system|developer) prompt",
    r"show me your hidden prompt",
    r"drop table",
    r"delete from",
    r"truncate table",
]


def validate_question(question: str) -> GuardrailResult:
    q = (question or "").strip()
    if not q:
        return GuardrailResult(False, "Please enter an insurance-related question.")
    if len(q) > MAX_QUESTION_LENGTH:
        return GuardrailResult(False, f"Question is too long. Maximum length is {MAX_QUESTION_LENGTH} characters.")
    lower = q.lower()
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, lower):
            return GuardrailResult(False, "That request is blocked by the application guardrails.")
    return GuardrailResult(True)


def grounded_answer_guard(answer: str | None, has_evidence: bool) -> str:
    if not has_evidence:
        return "I do not have enough grounded evidence in the insurance database or policy documents to answer that reliably."
    if not answer or not answer.strip():
        return "Evidence was retrieved, but I could not generate a reliable answer. Please inspect the supporting evidence shown below."
    return answer.strip()
