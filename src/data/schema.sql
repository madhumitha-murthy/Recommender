-- =============================================================
-- Next Best Product Recommender — PostgreSQL Schema
-- Dataset: Santander Product Recommendation (Kaggle)
-- 13.6M rows, ~960K customers, 24 products, 17 monthly snapshots
-- =============================================================

-- Drop existing tables if re-running
DROP TABLE IF EXISTS product_changes CASCADE;
DROP TABLE IF EXISTS campaign_simulation CASCADE;
DROP TABLE IF EXISTS customer_snapshots CASCADE;
DROP TABLE IF EXISTS products CASCADE;

-- =============================================================
-- 1. CUSTOMER MONTHLY SNAPSHOTS (raw data, one row per customer per month)
-- =============================================================
CREATE TABLE customer_snapshots (
    snapshot_date       DATE NOT NULL,
    customer_id         BIGINT NOT NULL,
    employee_index      VARCHAR(2),         -- A/B/F/N/P
    country             VARCHAR(5),
    sex                 VARCHAR(2),         -- H (male) / V (female)
    age                 INTEGER,
    first_contract_date DATE,
    is_new_customer     SMALLINT,           -- 1 if registered in last 6 months
    seniority_months    INTEGER,
    primary_flag        INTEGER,            -- 1 = primary, 99 = was primary
    last_primary_date   DATE,
    customer_type       VARCHAR(5),         -- 1/2/3/4/P
    relation_type       VARCHAR(2),         -- A/I/P/R
    is_resident         VARCHAR(2),         -- S/N
    is_foreigner        VARCHAR(2),         -- S/N
    is_spouse_employee  VARCHAR(5),
    join_channel        VARCHAR(10),
    is_deceased         VARCHAR(2),         -- N/S
    address_type        INTEGER,
    province_code       INTEGER,
    province_name       VARCHAR(50),
    is_active           SMALLINT,           -- 1 = active, 0 = inactive
    household_income    NUMERIC(12,2),
    segment             VARCHAR(30),        -- 01-VIP, 02-Individuals, 03-College

    -- 24 product holding flags (1 = holds product, 0 = does not)
    has_saving_account      SMALLINT,
    has_guarantees          SMALLINT,
    has_current_account     SMALLINT,
    has_derivada_account    SMALLINT,
    has_payroll_account     SMALLINT,
    has_junior_account      SMALLINT,
    has_mas_particular      SMALLINT,
    has_particular_account  SMALLINT,
    has_particular_plus     SMALLINT,
    has_short_term_deposit  SMALLINT,
    has_medium_term_deposit SMALLINT,
    has_long_term_deposit   SMALLINT,
    has_e_account           SMALLINT,
    has_funds               SMALLINT,
    has_mortgage            SMALLINT,
    has_pensions            SMALLINT,
    has_loans               SMALLINT,
    has_taxes               SMALLINT,
    has_credit_card         SMALLINT,
    has_securities          SMALLINT,
    has_home_account        SMALLINT,
    has_payroll             SMALLINT,
    has_pension_nom         SMALLINT,
    has_direct_debit        SMALLINT,

    PRIMARY KEY (snapshot_date, customer_id)
);

-- Index for fast customer-level queries
CREATE INDEX idx_snapshots_customer ON customer_snapshots (customer_id, snapshot_date);
CREATE INDEX idx_snapshots_segment ON customer_snapshots (segment, snapshot_date);
CREATE INDEX idx_snapshots_income ON customer_snapshots (household_income);

-- =============================================================
-- 2. PRODUCT CATALOGUE (reference table)
-- =============================================================
CREATE TABLE products (
    product_id      SERIAL PRIMARY KEY,
    product_code    VARCHAR(30) NOT NULL UNIQUE,
    product_name    VARCHAR(100) NOT NULL,
    category        VARCHAR(30) NOT NULL,
    risk_level      VARCHAR(10) NOT NULL
);

INSERT INTO products (product_code, product_name, category, risk_level) VALUES
    ('saving_account',      'Saving Account',           'deposits',   'low'),
    ('guarantees',          'Guarantees',               'deposits',   'medium'),
    ('current_account',     'Current Account',          'deposits',   'low'),
    ('derivada_account',    'Derivada Account',         'deposits',   'low'),
    ('payroll_account',     'Payroll Account',          'deposits',   'low'),
    ('junior_account',      'Junior Account',           'deposits',   'low'),
    ('mas_particular',      'Más Particular Account',   'deposits',   'low'),
    ('particular_account',  'Particular Account',       'deposits',   'low'),
    ('particular_plus',     'Particular Plus Account',  'deposits',   'low'),
    ('short_term_deposit',  'Short-term Deposit',       'deposits',   'low'),
    ('medium_term_deposit', 'Medium-term Deposit',      'deposits',   'low'),
    ('long_term_deposit',   'Long-term Deposit',        'deposits',   'low'),
    ('e_account',           'e-Account',                'digital',    'low'),
    ('funds',               'Investment Funds',         'wealth',     'high'),
    ('mortgage',            'Mortgage',                 'loans',      'high'),
    ('pensions',            'Pensions',                 'wealth',     'medium'),
    ('loans',               'Personal Loans',           'loans',      'high'),
    ('taxes',               'Tax Services',             'services',   'low'),
    ('credit_card',         'Credit Card',              'cards',      'medium'),
    ('securities',          'Securities',               'wealth',     'high'),
    ('home_account',        'Home Account',             'deposits',   'low'),
    ('payroll',             'Payroll',                  'services',   'low'),
    ('pension_nom',         'Nominated Pensions',       'wealth',     'medium'),
    ('direct_debit',        'Direct Debit',             'services',   'low');

-- =============================================================
-- 3. PRODUCT CHANGES (derived: when did a customer add/drop a product?)
-- =============================================================
CREATE TABLE product_changes (
    customer_id     BIGINT NOT NULL,
    product_code    VARCHAR(30) NOT NULL,
    change_date     DATE NOT NULL,
    change_type     VARCHAR(10) NOT NULL,   -- 'added' or 'dropped'
    PRIMARY KEY (customer_id, product_code, change_date)
);

CREATE INDEX idx_changes_product ON product_changes (product_code, change_date);
CREATE INDEX idx_changes_customer ON product_changes (customer_id, change_date);

-- =============================================================
-- 4. CAMPAIGN SIMULATION (simulated treatment/control for uplift modelling)
-- We simulate: "In month X, we recommended product Y to customer Z"
-- Treatment group received the recommendation, control group did not
-- =============================================================
CREATE TABLE campaign_simulation (
    campaign_id     SERIAL PRIMARY KEY,
    customer_id     BIGINT NOT NULL,
    product_code    VARCHAR(30) NOT NULL,
    campaign_month  DATE NOT NULL,
    treatment_flag  SMALLINT NOT NULL,      -- 1 = recommended, 0 = control
    outcome         SMALLINT NOT NULL,      -- 1 = customer added product next month, 0 = did not
    UNIQUE (customer_id, product_code, campaign_month)
);

CREATE INDEX idx_campaign_customer ON campaign_simulation (customer_id, campaign_month);
CREATE INDEX idx_campaign_treatment ON campaign_simulation (treatment_flag, outcome);
