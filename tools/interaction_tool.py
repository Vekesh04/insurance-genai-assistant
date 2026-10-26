from database.connection import get_connection


def get_claim_interactions(claim_id: str, limit: int = 50):
    claim_id = claim_id.strip().upper()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM customer_interactions
            WHERE UPPER(claim_id) = ?
            ORDER BY interaction_date DESC
            LIMIT ?
            """,
            (claim_id, min(max(limit, 1), 100)),
        ).fetchall()
    return [dict(row) for row in rows]
