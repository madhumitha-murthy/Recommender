"""
Load Santander Product Recommendation data into PostgreSQL.

Handles:
- Reading the 2.1GB CSV in chunks (memory-efficient)
- Cleaning column names, types, and missing values
- Loading into customer_snapshots table
- Deriving product_changes table (month-over-month diffs)
- Simulating campaign treatment/control for uplift modelling
"""

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
TRAIN_FILE = DATA_DIR / "train_ver2.csv"
CHUNK_SIZE = 500_000

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/nbp_recommender",
)

# Column mapping: original CSV name -> our clean name
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


def clean_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and rename a chunk of raw Santander data."""
    # Rename columns
    df = df.rename(columns=ALL_COLUMNS)

    # Keep only columns we need
    keep_cols = list(ALL_COLUMNS.values())
    df = df[[c for c in keep_cols if c in df.columns]]

    # Parse dates
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], format="%Y-%m-%d")
    df["first_contract_date"] = pd.to_datetime(
        df["first_contract_date"], format="%Y-%m-%d", errors="coerce"
    )
    df["last_primary_date"] = pd.to_datetime(
        df["last_primary_date"], errors="coerce"
    )

    # Clean age (comes as string with spaces)
    df["age"] = pd.to_numeric(df["age"].astype(str).str.strip(), errors="coerce")
    df["age"] = df["age"].clip(lower=18, upper=100)

    # Clean seniority
    df["seniority_months"] = pd.to_numeric(
        df["seniority_months"].astype(str).str.strip(), errors="coerce"
    )

    # Clean income
    df["household_income"] = pd.to_numeric(df["household_income"], errors="coerce")

    # Clean product columns: fill NaN with 0, cast to int
    product_cols = list(PRODUCT_COLUMNS.values())
    for col in product_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Clean categorical flags
    for col in ["is_new_customer", "is_active", "primary_flag", "address_type", "province_code"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": None, "": None, "NA": None})

    return df


def load_snapshots(engine) -> None:
    """Load customer_snapshots from CSV in chunks."""
    logger.info(f"Loading {TRAIN_FILE} ({TRAIN_FILE.stat().st_size / 1e9:.1f} GB)")

    total_rows = 0
    for i, chunk in enumerate(pd.read_csv(TRAIN_FILE, chunksize=CHUNK_SIZE, low_memory=False)):
        cleaned = clean_chunk(chunk)
        cleaned.to_sql(
            "customer_snapshots",
            engine,
            if_exists="append",
            index=False,
            method="multi",
        )
        total_rows += len(cleaned)
        logger.info(f"  Chunk {i+1}: loaded {len(cleaned):,} rows (total: {total_rows:,})")

    logger.info(f"Finished loading {total_rows:,} rows into customer_snapshots")


def derive_product_changes(engine) -> None:
    """
    Derive product_changes by comparing consecutive monthly snapshots.
    If a product flag goes 0->1, it's 'added'. If 1->0, it's 'dropped'.
    """
    logger.info("Deriving product_changes from monthly snapshots...")

    product_cols = list(PRODUCT_COLUMNS.values())
    # Map clean column name -> product_code (strip 'has_' prefix)
    col_to_code = {col: col.replace("has_", "") for col in product_cols}

    query = text("""
        INSERT INTO product_changes (customer_id, product_code, change_date, change_type)
        SELECT
            curr.customer_id,
            :product_code,
            curr.snapshot_date,
            CASE
                WHEN prev_val = 0 AND curr_val = 1 THEN 'added'
                WHEN prev_val = 1 AND curr_val = 0 THEN 'dropped'
            END AS change_type
        FROM (
            SELECT
                customer_id,
                snapshot_date,
                {curr_col} AS curr_val,
                LAG({curr_col}) OVER (
                    PARTITION BY customer_id ORDER BY snapshot_date
                ) AS prev_val
            FROM customer_snapshots
        ) sub
        CROSS JOIN LATERAL (
            SELECT curr.customer_id, curr.snapshot_date
        ) curr(customer_id, snapshot_date)
        WHERE (prev_val = 0 AND curr_val = 1) OR (prev_val = 1 AND curr_val = 0)
        ON CONFLICT DO NOTHING
    """)

    # Simpler approach: do it per product in Python to avoid dynamic SQL issues
    with engine.connect() as conn:
        for col, code in col_to_code.items():
            logger.info(f"  Processing {code}...")
            conn.execute(text(f"""
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
            """))
            conn.commit()

    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM product_changes")).scalar()
        logger.info(f"Finished: {count:,} product change events derived")


def simulate_campaigns(engine, seed: int = 42) -> None:
    """
    Simulate campaign treatment/control assignments for uplift modelling.

    Strategy:
    - For each month (excluding first and last), pick customers who DON'T yet
      hold a product but are eligible (based on age/income/seniority).
    - Randomly assign 50% to treatment (recommended), 50% to control.
    - Outcome = 1 if the customer added the product in the NEXT month snapshot.
    - For treatment group, slightly boost outcome probability (simulating that
      the recommendation had a causal effect on some customers).

    This is transparent simulation — documented in README as simulated, not real.
    """
    logger.info("Simulating campaign treatment/control assignments...")
    rng = np.random.default_rng(seed)

    with engine.connect() as conn:
        # Get available months
        months = pd.read_sql(
            "SELECT DISTINCT snapshot_date FROM customer_snapshots ORDER BY snapshot_date",
            conn,
        )["snapshot_date"].tolist()

    # Use months 3 through second-to-last for campaigns
    # (need previous month for features and next month for outcome)
    campaign_months = months[3:-1]
    product_cols = list(PRODUCT_COLUMNS.values())
    col_to_code = {col: col.replace("has_", "") for col in product_cols}

    # Pick top 5 most commonly added products for campaigns
    with engine.connect() as conn:
        top_products = pd.read_sql("""
            SELECT product_code, COUNT(*) as n_adds
            FROM product_changes
            WHERE change_type = 'added'
            GROUP BY product_code
            ORDER BY n_adds DESC
            LIMIT 5
        """, conn)["product_code"].tolist()

    logger.info(f"  Campaign products: {top_products}")

    all_campaigns = []

    for month in campaign_months:
        next_month_idx = months.index(month) + 1
        if next_month_idx >= len(months):
            continue
        next_month = months[next_month_idx]

        for product_code in top_products:
            col_name = f"has_{product_code}"
            if col_name not in product_cols:
                continue

            with engine.connect() as conn:
                # Get customers who DON'T hold this product in current month
                # and are active — these are campaign-eligible
                eligible = pd.read_sql(f"""
                    SELECT customer_id
                    FROM customer_snapshots
                    WHERE snapshot_date = '{month}'
                      AND {col_name} = 0
                      AND is_active = 1
                      AND age >= 21
                    ORDER BY RANDOM()
                    LIMIT 2000
                """, conn)

                if eligible.empty:
                    continue

                # Check who actually added the product next month
                added_next = pd.read_sql(f"""
                    SELECT customer_id
                    FROM customer_snapshots
                    WHERE snapshot_date = '{next_month}'
                      AND {col_name} = 1
                      AND customer_id IN (
                          SELECT customer_id FROM customer_snapshots
                          WHERE snapshot_date = '{month}' AND {col_name} = 0
                      )
                """, conn)
                added_set = set(added_next["customer_id"].tolist())

            # Assign treatment/control
            n = len(eligible)
            treatment_flags = rng.binomial(1, 0.5, size=n)

            # Outcome: did they add the product?
            outcomes = eligible["customer_id"].isin(added_set).astype(int).values

            # For treatment group, simulate a small causal uplift:
            # Some customers in treatment who didn't naturally convert, now convert
            # This gives the uplift model something real to find
            uplift_boost = 0.008  # 0.8% incremental conversion
            for j in range(n):
                if treatment_flags[j] == 1 and outcomes[j] == 0:
                    if rng.random() < uplift_boost:
                        outcomes[j] = 1

            campaign_df = pd.DataFrame({
                "customer_id": eligible["customer_id"].values,
                "product_code": product_code,
                "campaign_month": month,
                "treatment_flag": treatment_flags,
                "outcome": outcomes,
            })
            all_campaigns.append(campaign_df)

    if all_campaigns:
        campaigns = pd.concat(all_campaigns, ignore_index=True)
        # Drop duplicates (same customer-product-month)
        campaigns = campaigns.drop_duplicates(
            subset=["customer_id", "product_code", "campaign_month"], keep="first"
        )
        campaigns.to_sql(
            "campaign_simulation",
            create_engine(DB_URL),
            if_exists="append",
            index=False,
            method="multi",
        )
        logger.info(f"Finished: {len(campaigns):,} campaign records simulated")
        logger.info(f"  Treatment: {campaigns['treatment_flag'].sum():,}")
        logger.info(f"  Control: {(1 - campaigns['treatment_flag']).sum():,.0f}")
        logger.info(f"  Conversions: {campaigns['outcome'].sum():,}")
    else:
        logger.warning("No campaign records generated")


def main():
    engine = create_engine(DB_URL)

    # Create schema
    logger.info("Creating database schema...")
    schema_path = Path(__file__).parent / "schema.sql"
    with engine.connect() as conn:
        conn.execute(text(schema_path.read_text()))
        conn.commit()
    logger.info("Schema created")

    # Load snapshots
    load_snapshots(engine)

    # Derive product changes
    derive_product_changes(engine)

    # Simulate campaigns
    simulate_campaigns(engine)

    # Summary stats
    with engine.connect() as conn:
        n_customers = conn.execute(
            text("SELECT COUNT(DISTINCT customer_id) FROM customer_snapshots")
        ).scalar()
        n_snapshots = conn.execute(
            text("SELECT COUNT(*) FROM customer_snapshots")
        ).scalar()
        n_changes = conn.execute(
            text("SELECT COUNT(*) FROM product_changes")
        ).scalar()
        n_campaigns = conn.execute(
            text("SELECT COUNT(*) FROM campaign_simulation")
        ).scalar()

    logger.info("=" * 60)
    logger.info("DATA LOAD COMPLETE")
    logger.info(f"  Customers:        {n_customers:>12,}")
    logger.info(f"  Monthly snapshots: {n_snapshots:>12,}")
    logger.info(f"  Product changes:   {n_changes:>12,}")
    logger.info(f"  Campaign records:  {n_campaigns:>12,}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
