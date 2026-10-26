from rag.document_loader import load_all_policy_chunks
from rag.vector_store import build_vector_store
from rag.rag_service import PolicyRAGService


def test_policy_documents_are_chunked():
    chunks = load_all_policy_chunks()
    assert len(chunks) >= 100
    assert any(c.product_name == "Health Secure Plus" and c.section_title == "MATERNITY" for c in chunks)


def test_health_secure_maternity_retrieval():
    build_vector_store(prefer_semantic=False)
    service = PolicyRAGService()
    result = service.answer(
        "What is the maternity waiting period under Health Secure Plus?",
        product_name="Health Secure Plus",
        use_llm=False,
    )
    assert "24-month" in result["answer"] or "24 months" in result["answer"]
    assert result["sources"][0]["product_name"] == "Health Secure Plus"
