from database.connection import get_connection


def get_customer(customer_id: str):
    customer_id = customer_id.strip().upper()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM customers WHERE UPPER(customer_id) = ? LIMIT 1",
            (customer_id,),
        ).fetchone()
    return dict(row) if row else None
