from __future__ import annotations

from rag.retriever import HybridPolicyRetriever
from services.llm_service import LLMService


class PolicyRAGService:
    """Grounded policy question answering over the supplied policy documents."""

    def __init__(self):
        self.retriever = HybridPolicyRetriever()
        self.llm = LLMService()

    def search(self, question: str, top_k: int = 5, product_name: str | None = None) -> list[dict]:
        return [r.to_dict() for r in self.retriever.search(question, top_k=top_k, product_name=product_name)]

    def answer(self, question: str, top_k: int = 4, product_name: str | None = None, use_llm: bool = True) -> dict:
        hits = self.retriever.search(question, top_k=top_k, product_name=product_name)
        if not hits:
            return {
                "answer": "I could not find supporting policy evidence for that question.",
                "sources": [],
                "mode": "no_evidence",
            }

        context = "\n\n".join(
            f"SOURCE {i+1} | {h.product_name} | Section {h.section_number} {h.section_title}\n{h.text}"
            for i, h in enumerate(hits)
        )
        prompt = f"""QUESTION:
{question}

POLICY EVIDENCE:
{context}

Answer using ONLY the policy evidence above. If the evidence is insufficient, say so. Cite the relevant section in square brackets. Never guarantee claim approval.
"""
        answer = self.llm.generate(
            prompt,
            system="You are a grounded insurance policy assistant. Use only retrieved policy evidence.",
        ) if use_llm else None
        mode = f"{self.llm.provider}_llm" if answer else "extractive_fallback"

        if not answer:
            q = question.lower()
            best = hits[0]
            # Intent-aware extractive fallback keeps the demo useful without an LLM.
            # Example: for a maternity waiting-period question, prefer the MATERNITY
            # section over a generic waiting-period section when both were retrieved.
            if "maternity" in q:
                best = next((h for h in hits if "maternity" in h.section_title.lower()), best)
            body = best.text.split(". ", 1)[-1].strip()
            body_lower = body.lower()

            if "dental" in q and "unless caused by accident" in body_lower:
                answer = "Dental treatment is generally excluded unless it is caused by an accident."
            elif ("drunk" in q or "intoxic" in q) and "intoxication" in body_lower:
                answer = "No. The policy exclusions include driving under intoxication."
            elif q.startswith(("is ", "are ", "does ", "can ")) and "not covered" in body_lower:
                answer = "No. " + body
            elif q.startswith(("is ", "are ", "does ", "can ")) and "unless" in body_lower and "exclusion" in best.section_title.lower():
                answer = "Generally excluded, subject to the stated exception: " + body
            else:
                answer = body
            answer += f" [{best.product_name}, Section {best.section_number}: {best.section_title}]"

        return {
            "answer": answer,
            "mode": mode,
            "embedding_backend": self.retriever.info["embedding_backend"],
            "sources": [
                {
                    "product_name": h.product_name,
                    "section_number": h.section_number,
                    "section_title": h.section_title,
                    "source_file": h.source_file,
                    "score": round(h.combined_score, 4),
                    "text": h.text,
                }
                for h in hits
            ],
        }
