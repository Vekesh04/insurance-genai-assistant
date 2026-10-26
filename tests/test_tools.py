from tools.claims_tool import get_claim
from tools.document_tool import get_document_issues


def test_known_claim_exists():
    claim = get_claim("CLM300001")
    assert claim is not None
    assert claim["claim_id"] == "CLM300001"


def test_document_analysis_shape():
    result = get_document_issues("CLM300001")
    assert "missing" in result
    assert "pending" in result
    assert "rejected" in result
    assert "verified" in result
