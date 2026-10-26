from __future__ import annotations

import json
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from graph.router import route_question
from guardrails.validation import grounded_answer_guard, validate_question
from rag.rag_service import PolicyRAGService
from services.llm_service import LLMService
from tools.claims_tool import get_claim_summary
from tools.customer_tool import get_customer
from tools.document_tool import get_document_issues
from tools.hospital_tool import get_hospital
from tools.interaction_tool import get_claim_interactions
from tools.policy_tool import get_policy, get_customer_policies


class AssistantState(TypedDict, total=False):
    question: str
    allowed: bool
    route: str
    route_reason: str
    entities: dict[str, Any]
    sql_result: dict[str, Any] | list[Any] | None
    rag_result: dict[str, Any] | None
    answer: str
    evidence: list[dict[str, Any]]
    error: str | None


_llm = LLMService()
_rag: PolicyRAGService | None = None


def _rag_service() -> PolicyRAGService:
    global _rag
    if _rag is None:
        _rag = PolicyRAGService()
    return _rag


def safety_node(state: AssistantState) -> AssistantState:
    result = validate_question(state.get("question", ""))
    return {"allowed": result.allowed, "error": result.message}


def route_node(state: AssistantState) -> AssistantState:
    decision = route_question(state["question"])
    return {"route": decision.route, "route_reason": decision.reason, "entities": decision.entities}


def blocked_node(state: AssistantState) -> AssistantState:
    return {"answer": state.get("error") or "Request blocked by guardrails.", "evidence": []}


def _sql_lookup(question: str, entities: dict) -> dict:
    claim_id = entities.get("claim_id")
    policy_id = entities.get("policy_id")
    customer_id = entities.get("customer_id")
    hospital_id = entities.get("hospital_id")

    if claim_id:
        summary = get_claim_summary(claim_id)
        if not summary:
            return {"type": "claim", "found": False, "claim_id": claim_id}
        return {
            "type": "claim",
            "found": True,
            **summary,
            "document_analysis": get_document_issues(claim_id),
            "interactions": get_claim_interactions(claim_id, limit=5),
        }
    if policy_id:
        policy = get_policy(policy_id)
        return {"type": "policy", "found": bool(policy), "policy": policy}
    if customer_id:
        customer = get_customer(customer_id)
        return {
            "type": "customer",
            "found": bool(customer),
            "customer": customer,
            "policies": get_customer_policies(customer_id) if customer else [],
        }
    if hospital_id:
        hospital = get_hospital(hospital_id)
        return {"type": "hospital", "found": bool(hospital), "hospital": hospital}
    return {"type": "unknown", "found": False, "message": "Please include a claim, policy, customer, or hospital ID for this database question."}


def _deterministic_sql_answer(result: dict) -> str:
    if not result.get("found"):
        return result.get("message") or "I could not find that record in the SQL database."
    if result["type"] == "claim":
        c = result["claim"]
        docs = result.get("document_analysis", {})
        missing = docs.get("missing") or []
        pending = docs.get("pending") or []
        parts = [
            f"Claim {c['claim_id']} is currently '{c.get('claim_status')}'.",
            f"Claim amount: ₹{float(c.get('claim_amount') or 0):,.0f}; approved amount: ₹{float(c.get('approved_amount') or 0):,.0f}.",
        ]
        if missing:
            parts.append("Missing documents: " + ", ".join(missing) + ".")
        if pending:
            parts.append("Pending documents: " + ", ".join(pending) + ".")
        if c.get("manual_review_flag"):
            parts.append(f"Manual review flag is set; fraud risk score is {float(c.get('fraud_risk_score') or 0):.3f}.")
        if c.get("rejection_or_adjustment_reason"):
            parts.append("Recorded reason: " + str(c["rejection_or_adjustment_reason"]) + ".")
        return " ".join(parts)
    if result["type"] == "policy":
        p = result["policy"]
        return f"Policy {p['policy_id']} is {p.get('policy_status')} for product {p.get('product_name')}, with sum insured ₹{float(p.get('sum_insured') or 0):,.0f}."
    if result["type"] == "customer":
        c = result["customer"]
        return f"Customer {c['customer_id']} is {c.get('customer_name')} from {c.get('city')}, with KYC status {c.get('kyc_status')}. I found {len(result.get('policies', []))} policy record(s)."
    if result["type"] == "hospital":
        h = result["hospital"]
        return f"{h.get('hospital_name')} ({h.get('hospital_id')}) is marked '{h.get('network_status')}' in {h.get('city')}; specialty: {h.get('specialty')}."
    return "Database result retrieved."


def sql_node(state: AssistantState) -> AssistantState:
    result = _sql_lookup(state["question"], state.get("entities", {}))
    answer = _deterministic_sql_answer(result)
    # Optional LLM only rewrites supplied evidence; it does not invent database facts.
    if result.get("found"):
        prompt = f"Question: {state['question']}\n\nDATABASE EVIDENCE:\n{json.dumps(result, default=str)[:12000]}\n\nWrite a concise answer using only this evidence. Do not invent fields or guarantee claim approval."
        llm_answer = _llm.generate(prompt, system="You are a grounded insurance operations assistant.")
        if llm_answer:
            answer = llm_answer
    return {"sql_result": result, "answer": grounded_answer_guard(answer, bool(result.get('found'))), "evidence": [{"source": "sql", "data": result}]}


def rag_node(state: AssistantState) -> AssistantState:
    result = _rag_service().answer(state["question"], use_llm=True)
    sources = result.get("sources", [])
    return {"rag_result": result, "answer": grounded_answer_guard(result.get("answer"), bool(sources)), "evidence": sources}


def combined_node(state: AssistantState) -> AssistantState:
    entities = state.get("entities", {})
    claim_id = entities.get("claim_id")
    sql_result = _sql_lookup(state["question"], entities)
    if not claim_id or not sql_result.get("found"):
        answer = _deterministic_sql_answer(sql_result)
        return {"sql_result": sql_result, "answer": answer, "evidence": [{"source": "sql", "data": sql_result}]}

    policy = sql_result.get("policy") or {}
    product_name = policy.get("product_name")
    rag_result = _rag_service().answer(
        state["question"] + f" Product: {product_name or ''}",
        product_name=product_name,
        use_llm=False,
    )
    docs = sql_result.get("document_analysis", {})
    claim = sql_result.get("claim", {})
    policy_sources = rag_result.get("sources", [])

    fallback_parts = [
        f"Claim {claim_id} is '{claim.get('claim_status')}' for product {product_name or 'unknown'}.",
        f"Claim amount is ₹{float(claim.get('claim_amount') or 0):,.0f}; approved amount is ₹{float(claim.get('approved_amount') or 0):,.0f}.",
    ]
    if docs.get("missing"):
        fallback_parts.append("Missing documents: " + ", ".join(docs["missing"]) + ".")
    if docs.get("pending"):
        fallback_parts.append("Pending documents: " + ", ".join(docs["pending"]) + ".")
    if policy_sources:
        top = policy_sources[0]
        fallback_parts.append(f"Relevant policy evidence was found in {top['product_name']}, Section {top['section_number']}: {top['section_title']}.")
    fallback_parts.append("This is an evidence summary, not a guarantee of claim approval.")
    answer = " ".join(fallback_parts)

    evidence_payload = {
        "claim_database": sql_result,
        "policy_sources": policy_sources,
    }
    prompt = f"""Question: {state['question']}

STRUCTURED CLAIM EVIDENCE:
{json.dumps(sql_result, default=str)[:10000]}

POLICY EVIDENCE:
{json.dumps(policy_sources, default=str)[:10000]}

Analyze the claim using ONLY the evidence above. Mention missing/pending documents, relevant policy rule, claim status, and any recorded manual-review/rejection reason. If evidence is insufficient, say so. Never promise or guarantee approval. Cite policy sections in square brackets.
"""
    llm_answer = _llm.generate(prompt, system="You are a careful insurance claim analyst. Ground every conclusion in supplied evidence.")
    if llm_answer:
        answer = llm_answer

    return {
        "sql_result": sql_result,
        "rag_result": rag_result,
        "answer": grounded_answer_guard(answer, bool(policy_sources) or bool(sql_result.get("found"))),
        "evidence": [{"source": "combined", "data": evidence_payload}],
    }


def general_node(state: AssistantState) -> AssistantState:
    answer = (
        "I can help with insurance claims, customers, policies, hospitals, claim documents, "
        "and policy wording. For database questions, include an ID such as CLM300001 or POL200000."
    )
    return {"answer": answer, "evidence": []}


def _after_safety(state: AssistantState) -> str:
    return "router" if state.get("allowed") else "blocked"


def _after_router(state: AssistantState) -> str:
    return state.get("route", "general")


def build_graph():
    graph = StateGraph(AssistantState)
    graph.add_node("safety", safety_node)
    graph.add_node("router", route_node)
    graph.add_node("blocked", blocked_node)
    graph.add_node("sql", sql_node)
    graph.add_node("rag", rag_node)
    graph.add_node("combined", combined_node)
    graph.add_node("general", general_node)

    graph.add_edge(START, "safety")
    graph.add_conditional_edges("safety", _after_safety, {"router": "router", "blocked": "blocked"})
    graph.add_conditional_edges(
        "router",
        _after_router,
        {"sql": "sql", "rag": "rag", "combined": "combined", "general": "general"},
    )
    for node in ("blocked", "sql", "rag", "combined", "general"):
        graph.add_edge(node, END)
    return graph.compile()


assistant_graph = build_graph()


def ask_assistant(question: str) -> dict:
    result = assistant_graph.invoke({"question": question})
    return {
        "question": question,
        "route": result.get("route", "blocked" if not result.get("allowed", True) else "general"),
        "route_reason": result.get("route_reason"),
        "entities": result.get("entities", {}),
        "answer": result.get("answer", ""),
        "evidence": result.get("evidence", []),
        "sql_result": result.get("sql_result"),
        "rag_result": result.get("rag_result"),
    }
