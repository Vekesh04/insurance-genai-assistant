"""Tool registry used by the LangGraph workflow.

These are ordinary Python functions, which makes the behavior easy to explain in a
viva. The graph decides which tool to call instead of sending every question to an LLM.
"""
from tools.claims_tool import get_claim, get_claim_summary
from tools.customer_tool import get_customer
from tools.document_tool import get_claim_documents, get_document_issues
from tools.hospital_tool import get_hospital, search_hospitals
from tools.interaction_tool import get_claim_interactions
from tools.policy_tool import get_policy, get_customer_policies

TOOLS = {
    "get_claim": get_claim,
    "get_claim_summary": get_claim_summary,
    "get_customer": get_customer,
    "get_claim_documents": get_claim_documents,
    "get_document_issues": get_document_issues,
    "get_hospital": get_hospital,
    "search_hospitals": search_hospitals,
    "get_claim_interactions": get_claim_interactions,
    "get_policy": get_policy,
    "get_customer_policies": get_customer_policies,
}
