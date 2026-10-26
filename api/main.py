from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from config import DATABASE_PATH
from graph.workflow import ask_assistant
from rag.rag_service import PolicyRAGService
from tools.claims_tool import get_claim, get_claim_summary
from tools.customer_tool import get_customer
from tools.document_tool import get_claim_documents, get_document_issues
from tools.hospital_tool import get_hospital, search_hospitals
from tools.interaction_tool import get_claim_interactions
from tools.policy_tool import get_customer_policies, get_policy

app = FastAPI(
    title="Enterprise Insurance Claims & Policy Intelligence Assistant",
    description="Python + SQLite + Hybrid RAG + LangGraph + Guardrails",
    version="1.0.0",
)

_rag_service: PolicyRAGService | None = None


def get_rag_service() -> PolicyRAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = PolicyRAGService()
    return _rag_service


def ensure_database():
    if not DATABASE_PATH.exists():
        raise HTTPException(status_code=503, detail="Database not found. Run: python -m database.create_database")


class AssistantQuestion(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class RAGQuestion(BaseModel):
    question: str = Field(min_length=2)
    product_name: str | None = None
    top_k: int = Field(default=4, ge=1, le=10)
    use_llm: bool = True


@app.get("/")
def root():
    return {
        "message": "Insurance Gen-AI Assistant API is running",
        "database_ready": DATABASE_PATH.exists(),
        "docs": "/docs",
        "main_endpoint": "/assistant/ask",
    }


@app.get("/health")
def health():
    return {"status": "ok", "database_ready": DATABASE_PATH.exists()}


@app.post("/assistant/ask")
def assistant_ask(payload: AssistantQuestion):
    ensure_database()
    return ask_assistant(payload.question)


@app.post("/rag/ask")
def rag_ask(payload: RAGQuestion):
    return get_rag_service().answer(payload.question, top_k=payload.top_k, product_name=payload.product_name, use_llm=payload.use_llm)


@app.get("/rag/search")
def rag_search(q: str, product_name: str | None = None, top_k: int = Query(default=5, ge=1, le=10)):
    return {"query": q, "results": get_rag_service().search(q, top_k=top_k, product_name=product_name)}


@app.get("/claims/{claim_id}")
def claim_details(claim_id: str):
    ensure_database(); result = get_claim(claim_id)
    if not result: raise HTTPException(status_code=404, detail="Claim not found")
    return result


@app.get("/claims/{claim_id}/summary")
def claim_summary(claim_id: str):
    ensure_database(); result = get_claim_summary(claim_id)
    if not result: raise HTTPException(status_code=404, detail="Claim not found")
    result["document_analysis"] = get_document_issues(claim_id)
    result["interactions"] = get_claim_interactions(claim_id, limit=10)
    return result


@app.get("/claims/{claim_id}/documents")
def claim_documents(claim_id: str):
    ensure_database(); docs = get_claim_documents(claim_id)
    if not docs: raise HTTPException(status_code=404, detail="No claim documents found")
    return {"claim_id": claim_id.upper(), "documents": docs}


@app.get("/claims/{claim_id}/document-issues")
def claim_document_issues(claim_id: str):
    ensure_database(); return get_document_issues(claim_id)


@app.get("/policies/{policy_id}")
def policy_details(policy_id: str):
    ensure_database(); result = get_policy(policy_id)
    if not result: raise HTTPException(status_code=404, detail="Policy not found")
    return result


@app.get("/customers/{customer_id}")
def customer_details(customer_id: str):
    ensure_database(); result = get_customer(customer_id)
    if not result: raise HTTPException(status_code=404, detail="Customer not found")
    result["policies"] = get_customer_policies(customer_id); return result


@app.get("/hospitals/{hospital_id}")
def hospital_details(hospital_id: str):
    ensure_database(); result = get_hospital(hospital_id)
    if not result: raise HTTPException(status_code=404, detail="Hospital not found")
    return result


@app.get("/hospitals")
def hospital_search(city: str | None = None, network_status: str | None = None, limit: int = Query(default=20, ge=1, le=100)):
    ensure_database(); return {"results": search_hospitals(city, network_status, limit)}
