"""Optional MCP extension.

This is not required to run the main application. Install `mcp` and run this file
only if your lecturer asks you to demonstrate the advanced MCP concept.
"""
try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:
    raise SystemExit("Install the optional MCP SDK first: pip install mcp") from exc

from tools.claims_tool import get_claim_summary
from tools.document_tool import get_document_issues
from rag.rag_service import PolicyRAGService

mcp = FastMCP("insurance-intelligence")


@mcp.tool()
def claim_summary(claim_id: str) -> dict:
    """Return structured claim, customer, policy, and hospital information."""
    return get_claim_summary(claim_id) or {"error": "claim not found"}


@mcp.tool()
def claim_document_issues(claim_id: str) -> dict:
    """Return missing, pending, rejected, and verified claim documents."""
    return get_document_issues(claim_id)


@mcp.tool()
def policy_search(question: str, product_name: str = "") -> dict:
    """Search policy documents using the local hybrid RAG index."""
    return PolicyRAGService().answer(question, product_name=product_name or None, use_llm=False)


if __name__ == "__main__":
    mcp.run()
