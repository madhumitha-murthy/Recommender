"""
Fast data loader using PostgreSQL COPY command via psycopg2.
Much faster than SQLAlchemy's to_sql for large datasets.
"""

import io
import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
TRAIN_FILE = DATA_DIR / "train_ver2.csv"
CHUNK_SIZE = 200_000  # smaller chunks for memory

DB_PARAMS = {
    "dbname": "nbp_recommender",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": 5432,
}

PRODUCT_COLUMNS = {
    "ind_ahor_fin_ult1": "has_saving_account",
    "ind_aval_fin_ult1": "has_guarantees",
    "ind_cco_fin_ult1": "has_current_account",
    "ind_cder_fin_ult1": "has_derivada_account",
    "ind_cno_fin_ult1": "has_payroll_account",
    "ind_ctju_fin_ult1": "has_junior_account",
    "ind_ctma_fin_ult1": "has_mas_particular",
    "ind_ctop_fin_ult1": "has_particular_account",
    "ind_ctpp_fin_ult1": "has_particular_plus",
    "ind_deco_fin_ult1": "has_short_term_deposit",
    "ind_deme_fin_ult1": "has_medium_term_deposit",
    "ind_dela_fin_ult1": "has_long_term_deposit",
    "ind_ecue_fin_ult1": "has_e_account",
    "ind_fond_fin_ult1": "has_funds",
    "ind_hip_fin_ult1": "has_mortgage",
    "ind_plan_fin_ult1": "has_pensions",
    "ind_pres_fin_ult1": "has_loans",
    "ind_reca_fin_ult1": "has_taxes",
    "ind_tjcr_fin_ult1": "has_credit_card",
    "ind_valo_fin_ult1": "has_securities",
    "ind_viv_fin_ult1": "has_home_account",
    "ind_nomina_ult1": "has_payroll",
    "ind_nom_pens_ult1": "has_pension_nom",
    "ind_recibo_ult1": "has_direct_debit",
}

DEMOGRAPHIC_COLUMNS = {
    "fecha_dato": "snapshot_date",
    "ncodpers": "customer_id",
    "ind_empleado": "employee_index",
    "pais_residencia": "country",
    "sexo": "sex",
    "age": "age",
    "fecha_alta": "first_contract_date",
    "ind_nuevo": "is_new_customer",
    "antiguedad": "seniority_months",
    "indrel": "primary_flag",
    "ult_fec_cli_1t": "last_primary_date",
    "indrel_1mes": "customer_type",
    "tiprel_1mes": "relation_type",
    "indresi": "is_resident",
    "indext": "is_foreigner",
    "conyuemp": "is_spouse_employee",
    "canal_entrada": "join_channel",
    "indfall": "is_deceased",
    "tipodom": "address_type",
    "cod_prov": "province_code",
    "nomprov": "province_name",
    "ind_actividad_cliente": "is_active",
    "renta": "household_income",
    "segmento": "segment",
}

ALL_COLUMNS = {**DEMOGRAPHIC_COLUMNS, **PRODUCT_COLUMNS}

# DB column order (must match schema)
DB_COLUMNS = [
    "snapshot_date", "customer_id", "employee_index", "country", "sex", "age",
    "first_contract_date", "is_new_customer", "seniority_months", "primary_flag",
    "last_primary_date", "customer_type", "relation_type", "is_resident",
    "is_foreigner", "is_spouse_employee", "join_channel", "is_deceased",
    "address_type", "province_code", "province_name", "is_active",
    "household_income", "segment",
    "has_saving_account", "has_guarantees", "has_current_account",
    "has_derivada_account", "has_payroll_account", "has_junior_account",
    "has_mas_particular", "has_particular_account", "has_particular_plus",
    "has_short_term_deposit", "has_medium_term_deposit", "has_long_term_deposit",
    "has_e_account", "has_funds", "has_mortgage", "has_pensions",
    "has_loans", "has_taxes", "has_credit_card", "has_securities",
    "has_home_account", "has_payroll", "has_pension_nom", "has_direct_debit",
]


def clean_chunk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=ALL_COLUMNS)
    keep_cols = [c for c in DB_COLUMNS if c in df.columns]
    df = df[keep_cols]

    # Parse dates
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], format="%Y-%m-%d")
    df["first_contract_date"] = pd.to_datetime(df["first_contract_date"], format="%Y-%m-%d", errors="coerce")
    df["last_primary_date"] = pd.to_datetime(df["last_primary_date"], errors="coerce")

    # Clean numerics
    df["age"] = pd.to_numeric(df["age"].astype(str).str.strip(), errors="coerce").clip(lower=18, upper=100)
    df["age"] = pd.array(df["age"], dtype=pd.Int64Dtype())
    df["seniority_months"] = pd.to_numeric(df["seniority_months"].astype(str).str.strip(), errors="coerce")
    df["seniority_months"] = pd.array(df["seniority_months"], dtype=pd.Int64Dtype())
    df["household_income"] = pd.to_numeric(df["household_income"], errors="coerce")

    # Product columns: NaN -> 0
    product_cols = list(PRODUCT_COLUMNS.values())
    for col in product_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Numeric flags — cast to nullable int
    for col in ["is_new_customer", "is_active", "primary_flag", "address_type", "province_code"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = pd.array(df[col], dtype=pd.Int64Dtype())

    # Clean strings
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": None, "": None, "NA": None, "None": None})

    # Ensure all DB columns exist
    for col in DB_COLUMNS:
        if col not in df.columns:
            df[col] = None

    return df[DB_COLUMNS]


def copy_to_postgres(df: pd.DataFrame, conn, table: str):
    """Use PostgreSQL COPY for fast bulk insert."""
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, sep="\t", na_rep="\\N")
    buf.seek(0)
    cur = conn.cursor()
    cur.copy_from(buf, table, sep="\t", null="\\N", columns=DB_COLUMNS)
    conn.commit()
    cur.close()


def main():
    conn = psycopg2.connect(**DB_PARAMS)
    logger.info(f"Loading {TRAIN_FILE} ({TRAIN_FILE.stat().st_size / 1e9:.1f} GB)")

    total_rows = 0
    for i, chunk in enumerate(pd.read_csv(TRAIN_FILE, chunksize=CHUNK_SIZE, low_memory=False)):
        cleaned = clean_chunk(chunk)
        copy_to_postgres(cleaned, conn, "customer_snapshots")
        total_rows += len(cleaned)
        logger.info(f"  Chunk {i+1}: {len(cleaned):,} rows (total: {total_rows:,})")

    logger.info(f"Loaded {total_rows:,} rows into customer_snapshots")

    # Derive product changes
    logger.info("Deriving product_changes...")
    cur = conn.cursor()
    product_cols = list(PRODUCT_COLUMNS.values())
    for col in product_cols:
        code = col.replace("has_", "")
        logger.info(f"  {code}...")
        cur.execute(f"""
            INSERT INTO product_changes (customer_id, product_code, change_date, change_type)
            SELECT
                customer_id,
                '{code}' AS product_code,
                snapshot_date,
                CASE
                    WHEN prev_val = 0 AND curr_val = 1 THEN 'added'
                    WHEN prev_val = 1 AND curr_val = 0 THEN 'dropped'
                END AS change_type
            FROM (
                SELECT
                    customer_id,
                    snapshot_date,
                    {col} AS curr_val,
                    LAG({col}) OVER (
                        PARTITION BY customer_id ORDER BY snapshot_date
                    ) AS prev_val
                FROM customer_snapshots
            ) sub
            WHERE (prev_val = 0 AND curr_val = 1)
               OR (prev_val = 1 AND curr_val = 0)
            ON CONFLICT DO NOTHING
        """)
        conn.commit()

    cur.execute("SELECT COUNT(*) FROM product_changes")
    n_changes = cur.fetchone()[0]
    logger.info(f"Product changes: {n_changes:,}")

    # Simulate campaigns
    logger.info("Simulating campaigns...")
    from src.data.load_to_postgres import simulate_campaigns
    from sqlalchemy import create_engine
    engine = create_engine(f"postgresql://postgres:postgres@localhost:5432/nbp_recommender")
    simulate_campaigns(engine)

    # Summary
    cur.execute("SELECT COUNT(DISTINCT customer_id) FROM customer_snapshots")
    n_cust = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM customer_snapshots")
    n_snap = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM campaign_simulation")
    n_camp = cur.fetchone()[0]

    logger.info("=" * 60)
    logger.info("DATA LOAD COMPLETE")
    logger.info(f"  Customers:         {n_cust:>12,}")
    logger.info(f"  Monthly snapshots: {n_snap:>12,}")
    logger.info(f"  Product changes:   {n_changes:>12,}")
    logger.info(f"  Campaign records:  {n_camp:>12,}")
    logger.info("=" * 60)

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
