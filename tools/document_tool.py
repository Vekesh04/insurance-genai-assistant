from database.connection import get_connection


def get_claim_documents(claim_id: str):
    claim_id = claim_id.strip().upper()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM claim_documents WHERE UPPER(claim_id) = ? ORDER BY document_type",
            (claim_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_document_issues(claim_id: str):
    documents = get_claim_documents(claim_id)
    if not documents:
        return {
            "claim_id": claim_id.upper(),
            "documents": [],
            "missing": [],
            "pending": [],
            "rejected": [],
            "verified": [],
        }

    result = {
        "claim_id": claim_id.upper(),
        "documents": documents,
        "missing": [],
        "pending": [],
        "rejected": [],
        "verified": [],
    }

    for doc in documents:
        upload = str(doc.get("upload_status") or "").strip().lower()
        verify = str(doc.get("verification_status") or "").strip().lower()
        name = doc.get("document_type")

        if upload == "missing":
            result["missing"].append(name)
        if "pending" in upload or "pending" in verify:
            result["pending"].append(name)
        if "reject" in upload or "reject" in verify:
            result["rejected"].append(name)
        if verify == "verified":
            result["verified"].append(name)

    return result
