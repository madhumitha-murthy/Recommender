# Next Best Product: Uplift-Aware Recommendation Copilot for Banking

A recommendation system that goes beyond "predict who will buy" to answer **"who will buy because of our recommendation?"** — using causal uplift modelling, hybrid collaborative filtering, and a RAG-powered agentic copilot for relationship managers.

## The Problem

Banks waste relationship manager (RM) time recommending products to customers who would have bought anyway ("sure things") or who will never buy ("lost causes"). The real value is in **persuadables** — customers who only convert when an RM reaches out.

This system identifies persuadables using **counterfactual uplift modelling**, generates personalised RM pitches using **RAG over product docs**, and serves it all through a **LangGraph agentic copilot**.

## Architecture

```
Kaggle Santander Dataset (13.6M rows, 960K customers, 24 products)
  |
  v
PostgreSQL + SQL Feature Engineering
  ├── RFM segmentation (window functions, NTILE)
  ├── Product velocity (LAG, 3-month trends)
  ├── CLV proxy scores
  ├── Cohort z-scores
  └── Product co-occurrence (self-join + lift)
  |
  v
Hybrid Recommendation Engine
  ├── Collaborative Filtering (ALS via implicit library)
  └── LightGBM Ranker (CF scores as features + customer features)
        |  Optuna hyperparameter tuning (50 trials, NDCG@5)
        v
Causal Uplift Model (T-Learner)
  ├── P(buy | recommended) - P(buy | not recommended) = Uplift Score
  └── Segments: Persuadables | Sure Things | Sleeping Dogs | Lost Causes
        |
        v
LangGraph Agentic Copilot (ReAct pattern)
  ├── Tool 1: Customer lookup (SQL)
  ├── Tool 2: Recommendation + uplift scoring
  └── Tool 3: RAG pitch generation (FAISS + sentence-transformers)
        |
        v
Streamlit Dashboard (6 panels) + FastAPI Scoring API
  |
  v
MLflow Experiment Tracking + KS-Test Drift Monitoring + Docker + CI/CD
```

## Key Components

### 1. Real Banking Data (Kaggle Santander)
- 13.6M customer-product snapshots across 17 months
- 960K unique customers, 24 banking products
- Month-over-month product change detection (adds/removals)
- Simulated treatment/control groups with realistic uplift signal

### 2. SQL-First Feature Engineering
All features computed inside PostgreSQL — not Pandas:
- **RFM scores** — Recency, Frequency, Monetary via window functions
- **Product portfolio** — Holdings count, binary product flags
- **Product velocity** — 3-month add/remove trends using LAG
- **CLV proxy** — Income-weighted product engagement, NTILE ranking
- **Cohort features** — Z-scores within join-date cohorts
- **Co-occurrence** — Product pair lift scores via self-join
- **Campaign features** — Treatment group, conversion flags

### 3. Hybrid Recommendation Engine
- **ALS Collaborative Filtering** — Matrix factorisation with exponential decay weighting on user-product interactions
- **LightGBM Ranker** — CF scores fed as features alongside customer demographics; Optuna-tuned (50 trials) optimising NDCG@5
- **SHAP explanations** — Feature importance for model interpretability

### 4. Causal Uplift Modelling (T-Learner)
- Separate LightGBM models for treatment and control groups
- Uplift = P(convert | recommended) − P(convert | not recommended)
- Customer segmentation into the classic uplift quadrant:
  - **Persuadables** (high uplift) — Target these with RM outreach
  - **Sure Things** (buy anyway) — Don't waste RM time
  - **Sleeping Dogs** (negative uplift) — Avoid; recommendation hurts
  - **Lost Causes** (low probability either way) — Skip
- Evaluation: uplift curves, AUUC, Qini coefficient
- **KS-test drift monitoring** — Detect distribution shift in uplift scores

### 5. LangGraph Agentic Copilot
- ReAct agent with 3 tools (customer lookup, scoring, RAG pitch)
- RAG over 5 product documentation files (FAISS + sentence-transformers)
- LLM generates personalised talking points per customer-product pair
- Groq Llama 3.3 for fast inference

### 6. Streamlit Dashboard
- Uplift quadrant scatter plot (interactive)
- Segment performance summary table
- Product penetration heatmap
- Business impact metrics (RM hours saved, incremental conversions, revenue)
- Uplift score distribution + drift monitoring
- Individual customer lookup

## Tech Stack

| Layer | Technology |
|---|---|
| Data | Kaggle Santander (13.6M rows) + PostgreSQL 16 |
| Feature Engineering | SQL (window functions, CTEs, LAG, NTILE) |
| ML Models | LightGBM (ranker + uplift), implicit (ALS) |
| Hyperparameter Tuning | Optuna (50 trials, NDCG@5) |
| Agent Framework | LangGraph, LangChain |
| LLM | Groq Llama 3.3 |
| RAG | FAISS, sentence-transformers |
| API | FastAPI |
| Dashboard | Streamlit, Plotly |
| Experiment Tracking | MLflow |
| Deployment | Docker Compose (PostgreSQL + MLflow) |
| CI/CD | GitHub Actions (pytest + ruff) |

## Project Structure

```
next-best-product-recommender/
├── src/
│   ├── data/
│   │   ├── schema.sql                # PostgreSQL schema (4 tables + indexes)
│   │   └── load_to_postgres.py       # Load Santander CSV → PostgreSQL (chunked)
│   ├── features/
│   │   └── sql_features.py           # 7 SQL feature queries → DataFrames
│   ├── models/
│   │   ├── recommender.py            # ALS CF + LGBMRanker + Optuna + SHAP
│   │   └── uplift.py                 # T-Learner uplift + drift monitoring
│   ├── agents/
│   │   ├── copilot.py                # LangGraph ReAct copilot
│   │   ├── tools.py                  # 3 agent tools
│   │   └── product_docs/             # 5 product docs for RAG
│   ├── api/
│   │   └── serve.py                  # FastAPI scoring endpoint
│   └── dashboard/
│       └── app.py                    # Streamlit dashboard (6 panels)
├── tests/
│   ├── test_sql_features.py          # SQL feature query tests
│   ├── test_recommender.py           # ALS + ranker tests
│   ├── test_uplift.py                # T-Learner + drift tests
│   ├── test_api.py                   # FastAPI endpoint tests
│   └── test_agents.py               # Copilot + tool tests
├── scripts/
│   ├── train_pipeline.py             # End-to-end training orchestration
│   └── setup_db.sh                   # PostgreSQL setup
├── configs/
│   ├── config.yaml                   # Hyperparameters, DB, MLflow config
│   └── products.yaml                 # 12 products with eligibility + cross-sell
├── data/raw/                         # Kaggle CSVs (gitignored)
├── docker-compose.yml                # PostgreSQL 16 + MLflow
├── requirements.txt
├── .github/workflows/ci.yml          # pytest + ruff on push/PR
├── .env.example
├── .gitignore
└── README.md
```

## Quick Start

```bash
# Clone
git clone https://github.com/madhumitha-murthy/next-best-product-recommender.git
cd next-best-product-recommender

# Start PostgreSQL + MLflow
docker-compose up -d

# Load Santander data into PostgreSQL
python src/data/load_to_postgres.py

# Train models (SQL features → recommender → uplift → MLflow)
python scripts/train_pipeline.py

# Start API
uvicorn src.api.serve:app --reload

# Start dashboard
streamlit run src/dashboard/app.py

# Run tests
pytest tests/ -v
```

## Design Decisions

- **Why uplift over propensity?** — Propensity models predict P(buy). Uplift models predict the *incremental* effect of the recommendation. This avoids wasting RM time on sure things.
- **Why T-Learner?** — Simple, interpretable, and works well with moderate treatment/control sample sizes. X-Learner would be better with imbalanced groups but adds complexity.
- **Why ALS as features (not standalone)?** — Pure CF has cold-start issues. Feeding CF scores into LightGBM lets the ranker learn when to trust collaborative signals vs customer demographics.
- **Why SQL for features?** — Demonstrates production-ready feature engineering. In banking, features live in databases, not notebooks.
- **Why Santander data?** — Real banking data with realistic sparsity, missing values, and temporal structure. More credible than synthetic data.
