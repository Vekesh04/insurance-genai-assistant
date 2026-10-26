from database.connection import get_connection


def get_claim(claim_id: str):
    claim_id = claim_id.strip().upper()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM claims WHERE UPPER(claim_id) = ? LIMIT 1",
            (claim_id,),
        ).fetchone()
    return dict(row) if row else None


def get_claim_summary(claim_id: str):
    claim = get_claim(claim_id)
    if not claim:
        return None

    with get_connection() as conn:
        policy = conn.execute(
            "SELECT * FROM policies WHERE policy_id = ? LIMIT 1",
            (claim["policy_id"],),
        ).fetchone()
        customer = conn.execute(
            "SELECT * FROM customers WHERE customer_id = ? LIMIT 1",
            (claim["customer_id"],),
        ).fetchone()
        hospital = None
        if claim.get("hospital_id"):
            hospital = conn.execute(
                "SELECT * FROM hospitals WHERE hospital_id = ? LIMIT 1",
                (claim["hospital_id"],),
            ).fetchone()

    return {
        "claim": claim,
        "policy": dict(policy) if policy else None,
        "customer": dict(customer) if customer else None,
        "hospital": dict(hospital) if hospital else None,
    }
