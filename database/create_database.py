from pathlib import Path
import sqlite3
import pandas as pd

from config import DATA_DIR, DATABASE_PATH, CSV_TABLES


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Keep genuine missing values as SQL NULL rather than inventing data.
    return df.where(pd.notna(df), None)


def create_database() -> Path:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DATABASE_PATH.exists():
        DATABASE_PATH.unlink()

    conn = sqlite3.connect(DATABASE_PATH)
    try:
        for csv_name, table_name in CSV_TABLES.items():
            csv_path = DATA_DIR / csv_name
            if not csv_path.exists():
                compressed_path = DATA_DIR / (csv_name + ".gz")
                if compressed_path.exists():
                    csv_path = compressed_path
            if not csv_path.exists():
                compressed_path = DATA_DIR / (csv_name + ".gz")
                if compressed_path.exists():
                    csv_path = compressed_path
            print(f"Loading {csv_name} -> {table_name} ...")
            # Chunked loading keeps memory usage low for the ~600k document rows.
            first = True
            for chunk in pd.read_csv(csv_path, chunksize=50_000):
                chunk = _clean_dataframe(chunk)
                chunk.to_sql(
                    table_name,
                    conn,
                    if_exists="replace" if first else "append",
                    index=False,
                )
                first = False

        conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_customers_customer_id
                ON customers(customer_id);
            CREATE INDEX IF NOT EXISTS idx_policies_policy_id
                ON policies(policy_id);
            CREATE INDEX IF NOT EXISTS idx_policies_customer_id
                ON policies(customer_id);
            CREATE INDEX IF NOT EXISTS idx_claims_claim_id
                ON claims(claim_id);
            CREATE INDEX IF NOT EXISTS idx_claims_customer_id
                ON claims(customer_id);
            CREATE INDEX IF NOT EXISTS idx_claims_policy_id
                ON claims(policy_id);
            CREATE INDEX IF NOT EXISTS idx_claims_hospital_id
                ON claims(hospital_id);
            CREATE INDEX IF NOT EXISTS idx_claim_documents_claim_id
                ON claim_documents(claim_id);
            CREATE INDEX IF NOT EXISTS idx_hospitals_hospital_id
                ON hospitals(hospital_id);
            CREATE INDEX IF NOT EXISTS idx_interactions_customer_id
                ON customer_interactions(customer_id);
            CREATE INDEX IF NOT EXISTS idx_interactions_claim_id
                ON customer_interactions(claim_id);
            """
        )
        conn.commit()
    finally:
        conn.close()

    print(f"Database created: {DATABASE_PATH}")
    return DATABASE_PATH


if __name__ == "__main__":
    create_database()
