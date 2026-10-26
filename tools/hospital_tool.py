from database.connection import get_connection


def get_hospital(hospital_id: str):
    hospital_id = hospital_id.strip().upper()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM hospitals WHERE UPPER(hospital_id) = ? LIMIT 1",
            (hospital_id,),
        ).fetchone()
    return dict(row) if row else None


def search_hospitals(city: str | None = None, network_status: str | None = None, limit: int = 20):
    sql = "SELECT * FROM hospitals WHERE 1=1"
    params = []
    if city:
        sql += " AND LOWER(city) = LOWER(?)"
        params.append(city.strip())
    if network_status:
        sql += " AND LOWER(network_status) = LOWER(?)"
        params.append(network_status.strip())
    sql += " ORDER BY hospital_name LIMIT ?"
    params.append(min(max(limit, 1), 100))

    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]
