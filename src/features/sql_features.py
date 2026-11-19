"""
SQL-first feature engineering for the recommendation + uplift models.

All features are computed inside PostgreSQL using window functions, CTEs,
and aggregations — not Pandas. This module executes the queries and returns
DataFrames ready for modelling.

Feature groups:
  1. RFM (Recency, Frequency, Monetary)
  2. Product affinity & portfolio breadth
  3. Transaction velocity & trends
  4. Customer lifetime value proxy
  5. Cohort-based behavioural features
  6. Product co-occurrence (for collaborative signal)
  7. Campaign features (for uplift modelling)
"""

import os
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/nbp_recommender",
)

# ---------------------------------------------------------------------------
# SQL Queries — each is a self-contained feature computation
# ---------------------------------------------------------------------------

# 1. RFM Features: Recency, Frequency, Monetary per customer at latest snapshot
QUERY_RFM = """
WITH latest AS (
    SELECT MAX(snapshot_date) AS max_date FROM customer_snapshots
),
customer_product_adds AS (
    SELECT
        customer_id,
        COUNT(*) AS total_products_added,
        MAX(change_date) AS last_product_add_date,
        MIN(change_date) AS first_product_add_date
    FROM product_changes
    WHERE change_type = 'added'
    GROUP BY customer_id
),
customer_latest AS (
    SELECT
        cs.customer_id,
        cs.age,
        cs.seniority_months,
        cs.household_income,
        cs.segment,
        cs.sex,
        cs.is_active,
        cs.join_channel,
        cs.province_code
    FROM customer_snapshots cs
    CROSS JOIN latest l
    WHERE cs.snapshot_date = l.max_date
)
SELECT
    cl.customer_id,
    -- Recency: days since last product addition
    COALESCE(
        (l.max_date - cpa.last_product_add_date),
        9999
    ) AS recency_days,
    -- Frequency: total products ever added
    COALESCE(cpa.total_products_added, 0) AS frequency_adds,
    -- Monetary: household income as proxy
    COALESCE(cl.household_income, 0) AS monetary_income,
    -- Tenure
    COALESCE(cl.seniority_months, 0) AS seniority_months,
    -- Days as customer
    COALESCE(
        (l.max_date - cpa.first_product_add_date),
        0
    ) AS days_since_first_add,
    -- Demographics
    cl.age,
    cl.sex,
    cl.is_active,
    cl.segment,
    cl.join_channel,
    cl.province_code
FROM customer_latest cl
CROSS JOIN latest l
LEFT JOIN customer_product_adds cpa ON cl.customer_id = cpa.customer_id
"""

# 2. Product portfolio: how many and which products does each customer hold?
QUERY_PRODUCT_PORTFOLIO = """
WITH latest AS (
    SELECT MAX(snapshot_date) AS max_date FROM customer_snapshots
)
SELECT
    cs.customer_id,
    -- Total products held
    (COALESCE(has_saving_account, 0) + COALESCE(has_guarantees, 0) +
     COALESCE(has_current_account, 0) + COALESCE(has_derivada_account, 0) +
     COALESCE(has_payroll_account, 0) + COALESCE(has_junior_account, 0) +
     COALESCE(has_mas_particular, 0) + COALESCE(has_particular_account, 0) +
     COALESCE(has_particular_plus, 0) + COALESCE(has_short_term_deposit, 0) +
     COALESCE(has_medium_term_deposit, 0) + COALESCE(has_long_term_deposit, 0) +
     COALESCE(has_e_account, 0) + COALESCE(has_funds, 0) +
     COALESCE(has_mortgage, 0) + COALESCE(has_pensions, 0) +
     COALESCE(has_loans, 0) + COALESCE(has_taxes, 0) +
     COALESCE(has_credit_card, 0) + COALESCE(has_securities, 0) +
     COALESCE(has_home_account, 0) + COALESCE(has_payroll, 0) +
     COALESCE(has_pension_nom, 0) + COALESCE(has_direct_debit, 0)
    ) AS total_products_held,
    -- Category counts
    (COALESCE(has_saving_account, 0) + COALESCE(has_current_account, 0) +
     COALESCE(has_derivada_account, 0) + COALESCE(has_payroll_account, 0) +
     COALESCE(has_particular_account, 0) + COALESCE(has_particular_plus, 0) +
     COALESCE(has_home_account, 0) + COALESCE(has_junior_account, 0) +
     COALESCE(has_mas_particular, 0)
    ) AS n_deposit_products,
    (COALESCE(has_funds, 0) + COALESCE(has_securities, 0) +
     COALESCE(has_pensions, 0) + COALESCE(has_pension_nom, 0)
    ) AS n_wealth_products,
    (COALESCE(has_mortgage, 0) + COALESCE(has_loans, 0)
    ) AS n_loan_products,
    COALESCE(has_credit_card, 0) AS has_credit_card,
    -- Individual product flags (for the model to use as features)
    COALESCE(has_saving_account, 0) AS has_saving_account,
    COALESCE(has_current_account, 0) AS has_current_account,
    COALESCE(has_payroll_account, 0) AS has_payroll_account,
    COALESCE(has_e_account, 0) AS has_e_account,
    COALESCE(has_funds, 0) AS has_funds,
    COALESCE(has_mortgage, 0) AS has_mortgage,
    COALESCE(has_pensions, 0) AS has_pensions,
    COALESCE(has_loans, 0) AS has_loans,
    COALESCE(has_direct_debit, 0) AS has_direct_debit,
    COALESCE(has_securities, 0) AS has_securities,
    COALESCE(has_payroll, 0) AS has_payroll
FROM customer_snapshots cs
CROSS JOIN latest l
WHERE cs.snapshot_date = l.max_date
"""

# 3. Product velocity: how fast is the customer adding products? (3-month trend)
QUERY_PRODUCT_VELOCITY = """
WITH monthly_adds AS (
    SELECT
        customer_id,
        DATE_TRUNC('month', change_date) AS month,
        COUNT(*) AS adds_in_month
    FROM product_changes
    WHERE change_type = 'added'
    GROUP BY customer_id, DATE_TRUNC('month', change_date)
),
velocity AS (
    SELECT
        customer_id,
        AVG(adds_in_month) AS avg_monthly_adds,
        MAX(adds_in_month) AS max_monthly_adds,
        COUNT(DISTINCT month) AS active_months,
        -- Recent vs older: compare last 3 months to earlier
        AVG(CASE WHEN month >= (SELECT MAX(month) - INTERVAL '3 months' FROM monthly_adds)
                 THEN adds_in_month END) AS recent_3m_avg_adds,
        AVG(CASE WHEN month < (SELECT MAX(month) - INTERVAL '3 months' FROM monthly_adds)
                 THEN adds_in_month END) AS older_avg_adds
    FROM monthly_adds
    GROUP BY customer_id
)
SELECT
    customer_id,
    COALESCE(avg_monthly_adds, 0) AS avg_monthly_adds,
    COALESCE(max_monthly_adds, 0) AS max_monthly_adds,
    COALESCE(active_months, 0) AS active_months,
    COALESCE(recent_3m_avg_adds, 0) AS recent_3m_avg_adds,
    COALESCE(older_avg_adds, 0) AS older_avg_adds,
    -- Velocity trend: positive = accelerating, negative = decelerating
    COALESCE(recent_3m_avg_adds, 0) - COALESCE(older_avg_adds, 0) AS velocity_trend
FROM velocity
"""

# 4. Customer lifetime value proxy
QUERY_CLV = """
WITH latest AS (
    SELECT MAX(snapshot_date) AS max_date FROM customer_snapshots
),
product_counts AS (
    SELECT
        cs.customer_id,
        cs.seniority_months,
        cs.household_income,
        (COALESCE(has_saving_account, 0) + COALESCE(has_guarantees, 0) +
         COALESCE(has_current_account, 0) + COALESCE(has_derivada_account, 0) +
         COALESCE(has_payroll_account, 0) + COALESCE(has_junior_account, 0) +
         COALESCE(has_mas_particular, 0) + COALESCE(has_particular_account, 0) +
         COALESCE(has_particular_plus, 0) + COALESCE(has_short_term_deposit, 0) +
         COALESCE(has_medium_term_deposit, 0) + COALESCE(has_long_term_deposit, 0) +
         COALESCE(has_e_account, 0) + COALESCE(has_funds, 0) +
         COALESCE(has_mortgage, 0) + COALESCE(has_pensions, 0) +
         COALESCE(has_loans, 0) + COALESCE(has_taxes, 0) +
         COALESCE(has_credit_card, 0) + COALESCE(has_securities, 0) +
         COALESCE(has_home_account, 0) + COALESCE(has_payroll, 0) +
         COALESCE(has_pension_nom, 0) + COALESCE(has_direct_debit, 0)
        ) AS total_products
    FROM customer_snapshots cs
    CROSS JOIN latest l
    WHERE cs.snapshot_date = l.max_date
)
SELECT
    customer_id,
    -- Simple CLV proxy: income * tenure * product depth
    CASE
        WHEN seniority_months > 0 AND household_income > 0
        THEN ROUND(
            (household_income / 12.0) * (seniority_months / 12.0) * (1 + total_products * 0.1),
            2
        )
        ELSE 0
    END AS clv_proxy,
    -- Income tier
    NTILE(5) OVER (ORDER BY COALESCE(household_income, 0)) AS income_quintile,
    -- Product depth tier
    NTILE(4) OVER (ORDER BY total_products) AS product_depth_quartile
FROM product_counts
"""

# 5. Cohort features: how does this customer compare to their join cohort?
QUERY_COHORT = """
WITH latest AS (
    SELECT MAX(snapshot_date) AS max_date FROM customer_snapshots
),
customer_products AS (
    SELECT
        cs.customer_id,
        DATE_TRUNC('quarter', cs.first_contract_date) AS join_cohort,
        (COALESCE(has_saving_account, 0) + COALESCE(has_guarantees, 0) +
         COALESCE(has_current_account, 0) + COALESCE(has_derivada_account, 0) +
         COALESCE(has_payroll_account, 0) + COALESCE(has_junior_account, 0) +
         COALESCE(has_mas_particular, 0) + COALESCE(has_particular_account, 0) +
         COALESCE(has_particular_plus, 0) + COALESCE(has_short_term_deposit, 0) +
         COALESCE(has_medium_term_deposit, 0) + COALESCE(has_long_term_deposit, 0) +
         COALESCE(has_e_account, 0) + COALESCE(has_funds, 0) +
         COALESCE(has_mortgage, 0) + COALESCE(has_pensions, 0) +
         COALESCE(has_loans, 0) + COALESCE(has_taxes, 0) +
         COALESCE(has_credit_card, 0) + COALESCE(has_securities, 0) +
         COALESCE(has_home_account, 0) + COALESCE(has_payroll, 0) +
         COALESCE(has_pension_nom, 0) + COALESCE(has_direct_debit, 0)
        ) AS total_products,
        cs.household_income
    FROM customer_snapshots cs
    CROSS JOIN latest l
    WHERE cs.snapshot_date = l.max_date
      AND cs.first_contract_date IS NOT NULL
),
cohort_stats AS (
    SELECT
        join_cohort,
        AVG(total_products) AS cohort_avg_products,
        AVG(household_income) AS cohort_avg_income,
        STDDEV(total_products) AS cohort_std_products,
        COUNT(*) AS cohort_size
    FROM customer_products
    GROUP BY join_cohort
)
SELECT
    cp.customer_id,
    cp.join_cohort,
    cs.cohort_size,
    -- How many more/fewer products vs cohort average
    ROUND(cp.total_products - cs.cohort_avg_products, 2) AS products_vs_cohort,
    -- Z-score within cohort
    CASE
        WHEN cs.cohort_std_products > 0
        THEN ROUND((cp.total_products - cs.cohort_avg_products) / cs.cohort_std_products, 2)
        ELSE 0
    END AS product_zscore_in_cohort,
    -- Income vs cohort
    CASE
        WHEN cs.cohort_avg_income > 0
        THEN ROUND(cp.household_income / cs.cohort_avg_income, 2)
        ELSE NULL
    END AS income_ratio_vs_cohort
FROM customer_products cp
JOIN cohort_stats cs ON cp.join_cohort = cs.join_cohort
"""

# 6. Product co-occurrence: which products are commonly held together?
# Returns a customer-level feature: how many "natural pairs" does the customer hold?
QUERY_COOCCURRENCE = """
WITH latest AS (
    SELECT MAX(snapshot_date) AS max_date FROM customer_snapshots
),
customer_latest AS (
    SELECT *
    FROM customer_snapshots cs
    CROSS JOIN latest l
    WHERE cs.snapshot_date = l.max_date
)
SELECT
    customer_id,
    -- Common product pairs (domain knowledge + data-driven)
    (COALESCE(has_current_account, 0) * COALESCE(has_direct_debit, 0)) AS pair_current_debit,
    (COALESCE(has_payroll_account, 0) * COALESCE(has_payroll, 0)) AS pair_payroll_both,
    (COALESCE(has_saving_account, 0) * COALESCE(has_credit_card, 0)) AS pair_savings_card,
    (COALESCE(has_mortgage, 0) * COALESCE(has_home_account, 0)) AS pair_mortgage_home,
    (COALESCE(has_pensions, 0) * COALESCE(has_pension_nom, 0)) AS pair_pensions_both,
    (COALESCE(has_funds, 0) * COALESCE(has_securities, 0)) AS pair_wealth_both,
    -- Total natural pairs held (0 to 6)
    (COALESCE(has_current_account, 0) * COALESCE(has_direct_debit, 0)) +
    (COALESCE(has_payroll_account, 0) * COALESCE(has_payroll, 0)) +
    (COALESCE(has_saving_account, 0) * COALESCE(has_credit_card, 0)) +
    (COALESCE(has_mortgage, 0) * COALESCE(has_home_account, 0)) +
    (COALESCE(has_pensions, 0) * COALESCE(has_pension_nom, 0)) +
    (COALESCE(has_funds, 0) * COALESCE(has_securities, 0))
    AS total_natural_pairs
FROM customer_latest
"""

# 7. Campaign features for uplift modelling
QUERY_CAMPAIGN_FEATURES = """
SELECT
    c.customer_id,
    c.product_code,
    c.campaign_month,
    c.treatment_flag,
    c.outcome
FROM campaign_simulation c
"""

# ---------------------------------------------------------------------------
# Feature extraction functions
# ---------------------------------------------------------------------------

def get_engine():
    return create_engine(DB_URL)


def run_query(query: str, engine=None) -> pd.DataFrame:
    """Execute a SQL query and return a DataFrame."""
    if engine is None:
        engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


def build_customer_features(engine=None) -> pd.DataFrame:
    """
    Build the full customer feature matrix by running all SQL queries
    and joining the results.
    """
    if engine is None:
        engine = get_engine()

    logger.info("Building customer features from SQL...")

    logger.info("  1/6 RFM features...")
    rfm = run_query(QUERY_RFM, engine)

    logger.info("  2/6 Product portfolio...")
    portfolio = run_query(QUERY_PRODUCT_PORTFOLIO, engine)

    logger.info("  3/6 Product velocity...")
    velocity = run_query(QUERY_PRODUCT_VELOCITY, engine)

    logger.info("  4/6 CLV proxy...")
    clv = run_query(QUERY_CLV, engine)

    logger.info("  5/6 Cohort features...")
    cohort = run_query(QUERY_COHORT, engine)

    logger.info("  6/6 Co-occurrence features...")
    cooccurrence = run_query(QUERY_COOCCURRENCE, engine)

    # Join all features on customer_id
    features = rfm
    for df in [portfolio, velocity, clv, cohort, cooccurrence]:
        # Drop duplicate columns before merge
        overlap = set(features.columns) & set(df.columns) - {"customer_id"}
        df = df.drop(columns=list(overlap), errors="ignore")
        features = features.merge(df, on="customer_id", how="left")

    logger.info(f"  Final feature matrix: {features.shape[0]:,} customers x {features.shape[1]} features")
    return features


def build_campaign_dataset(engine=None) -> pd.DataFrame:
    """
    Build the dataset for uplift modelling: campaign records joined with
    customer features at the time of the campaign.
    """
    if engine is None:
        engine = get_engine()

    logger.info("Building campaign dataset for uplift modelling...")

    campaigns = run_query(QUERY_CAMPAIGN_FEATURES, engine)
    customer_features = build_customer_features(engine)

    # Merge campaign records with customer features
    dataset = campaigns.merge(customer_features, on="customer_id", how="inner")

    logger.info(f"  Campaign dataset: {len(dataset):,} records")
    logger.info(f"  Treatment: {dataset['treatment_flag'].sum():,}")
    logger.info(f"  Control: {(1 - dataset['treatment_flag']).sum():,.0f}")
    logger.info(f"  Conversion rate (treatment): {dataset[dataset['treatment_flag']==1]['outcome'].mean():.4f}")
    logger.info(f"  Conversion rate (control): {dataset[dataset['treatment_flag']==0]['outcome'].mean():.4f}")

    return dataset


if __name__ == "__main__":
    engine = get_engine()

    # Build and save features
    features = build_customer_features(engine)
    out_path = Path(__file__).resolve().parents[2] / "data" / "processed"
    out_path.mkdir(parents=True, exist_ok=True)

    features.to_parquet(out_path / "customer_features.parquet", index=False)
    logger.info(f"Saved to {out_path / 'customer_features.parquet'}")

    # Build and save campaign dataset
    campaign_ds = build_campaign_dataset(engine)
    campaign_ds.to_parquet(out_path / "campaign_dataset.parquet", index=False)
    logger.info(f"Saved to {out_path / 'campaign_dataset.parquet'}")
