from database.connection import get_connection


def get_policy(policy_id: str):
    policy_id = policy_id.strip().upper()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM policies WHERE UPPER(policy_id) = ? LIMIT 1",
            (policy_id,),
        ).fetchone()
    return dict(row) if row else None


def get_customer_policies(customer_id: str):
    customer_id = customer_id.strip().upper()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM policies WHERE UPPER(customer_id) = ? ORDER BY policy_start_date DESC",
            (customer_id,),
        ).fetchall()
    return [dict(row) for row in rows]
