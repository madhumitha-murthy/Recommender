"""
Agent tools for the recommendation copilot.

Tool 1: customer_lookup — Query customer profile + features from PostgreSQL
Tool 2: recommend_and_score — Run recommendation + uplift model → ranked products with segments
Tool 3: generate_pitch — RAG over product docs → personalised RM talking points
"""

import os
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.tools import tool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PRODUCT_DOCS_DIR = Path(__file__).parent / "product_docs"

# Lazy-loaded globals
_customer_features: pd.DataFrame | None = None
_als = None
_ranker = None
_uplift_model = None
_vector_store = None


def _load_customer_features() -> pd.DataFrame:
    global _customer_features
    if _customer_features is None:
        _customer_features = pd.read_parquet(DATA_DIR / "customer_features.parquet")
    return _customer_features


def _load_models():
    global _als, _ranker, _uplift_model
    if _als is None:
        from src.models.recommender import CollaborativeFilter
        _als = CollaborativeFilter.load(MODEL_DIR)
    if _ranker is None:
        with open(MODEL_DIR / "lgbm_ranker.pkl", "rb") as f:
            _ranker = pickle.load(f)
    if _uplift_model is None:
        from src.models.uplift import TLearnerUplift
        _uplift_model = TLearnerUplift.load(MODEL_DIR)
    return _als, _ranker, _uplift_model


def _load_vector_store():
    """Load or build FAISS vector store from product docs."""
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import DirectoryLoader, TextLoader

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    index_path = MODEL_DIR / "product_faiss_index"
    if index_path.exists():
        _vector_store = FAISS.load_local(
            str(index_path), embeddings, allow_dangerous_deserialization=True
        )
    else:
        loader = DirectoryLoader(str(PRODUCT_DOCS_DIR), glob="*.md", loader_cls=TextLoader)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        _vector_store = FAISS.from_documents(chunks, embeddings)
        index_path.mkdir(parents=True, exist_ok=True)
        _vector_store.save_local(str(index_path))

    return _vector_store


# ---------------------------------------------------------------------------
# Tool 1: Customer Lookup
# ---------------------------------------------------------------------------

@tool
def customer_lookup(customer_id: int) -> dict[str, Any]:
    """
    Look up a customer's profile and features from the database.
    Returns demographics, product holdings, RFM scores, CLV, and segment info.
    """
    features = _load_customer_features()
    cust = features[features["customer_id"] == customer_id]

    if cust.empty:
        return {"error": f"Customer {customer_id} not found"}

    row = cust.iloc[0]
    profile = {
        "customer_id": int(row.get("customer_id", 0)),
        "age": int(row.get("age", 0)) if pd.notna(row.get("age")) else None,
        "segment": row.get("segment", "Unknown"),
        "seniority_months": int(row.get("seniority_months", 0)) if pd.notna(row.get("seniority_months")) else 0,
        "household_income": float(row.get("monetary_income", 0)),
        "is_active": int(row.get("is_active", 0)) if pd.notna(row.get("is_active")) else 0,
        "total_products_held": int(row.get("total_products_held", 0)),
        "n_deposit_products": int(row.get("n_deposit_products", 0)),
        "n_wealth_products": int(row.get("n_wealth_products", 0)),
        "n_loan_products": int(row.get("n_loan_products", 0)),
        "has_credit_card": int(row.get("has_credit_card", 0)),
        "recency_days": int(row.get("recency_days", 9999)),
        "frequency_adds": int(row.get("frequency_adds", 0)),
        "clv_proxy": float(row.get("clv_proxy", 0)),
        "velocity_trend": float(row.get("velocity_trend", 0)),
    }
    return profile


# ---------------------------------------------------------------------------
# Tool 2: Recommend and Score
# ---------------------------------------------------------------------------

@tool
def recommend_and_score(customer_id: int, top_k: int = 5) -> dict[str, Any]:
    """
    Run the recommendation model and uplift model for a customer.
    Returns top-K product recommendations with scores and uplift segment.
    """
    features = _load_customer_features()
    als, ranker, uplift_model = _load_models()

    from src.models.recommender import recommend

    # Get recommendations
    recs = recommend(customer_id, features, als, ranker, k=top_k)

    if recs.empty:
        return {"error": "No recommendations available (customer may hold all products)"}

    # Get uplift segment for this customer
    cust_features = features[features["customer_id"] == customer_id]
    if not cust_features.empty:
        uplift_score = float(uplift_model.predict_uplift(cust_features)[0])
        segments = uplift_model.segment_customers(cust_features)
        segment = segments.iloc[0]["segment"]
    else:
        uplift_score = 0.0
        segment = "Unknown"

    return {
        "customer_id": customer_id,
        "uplift_segment": segment,
        "uplift_score": round(uplift_score, 4),
        "recommendations": [
            {
                "rank": int(row["rank"]),
                "product": row["product_code"],
                "score": round(float(row["score"]), 4),
            }
            for _, row in recs.iterrows()
        ],
        "action": {
            "Persuadable": "PROCEED — this customer is worth RM outreach",
            "Sure Thing": "SKIP — customer will likely buy without outreach",
            "Sleeping Dog": "AVOID — outreach may hurt conversion",
            "Lost Cause": "LOW PRIORITY — low conversion probability",
        }.get(segment, "REVIEW"),
    }


# ---------------------------------------------------------------------------
# Tool 3: Generate Pitch (RAG over product docs)
# ---------------------------------------------------------------------------

@tool
def generate_pitch(customer_id: int, product_code: str) -> dict[str, Any]:
    """
    Generate a personalised RM talking-point pitch for recommending a
    specific product to a customer. Uses RAG over product documentation
    to ground the pitch in actual product features.
    """
    # Get customer profile
    profile = customer_lookup.invoke({"customer_id": customer_id})
    if "error" in profile:
        return profile

    # Get relevant product docs via RAG
    vector_store = _load_vector_store()
    query = f"{product_code} features benefits eligibility"
    docs = vector_store.similarity_search(query, k=3)
    product_context = "\n".join(doc.page_content for doc in docs)

    # Build prompt
    prompt = f"""You are a relationship manager copilot at a bank.
Generate a concise, personalised pitch (3-4 bullet points) for recommending
the "{product_code}" product to this customer.

CUSTOMER PROFILE:
- Age: {profile.get('age', 'Unknown')}
- Segment: {profile.get('segment', 'Unknown')}
- Monthly income: S${profile.get('household_income', 0):,.0f}
- Tenure: {profile.get('seniority_months', 0)} months
- Products held: {profile.get('total_products_held', 0)}
- Has credit card: {'Yes' if profile.get('has_credit_card') else 'No'}
- CLV: S${profile.get('clv_proxy', 0):,.0f}

PRODUCT INFORMATION:
{product_context}

RULES:
- Be specific to this customer's profile
- Reference actual product features from the documentation
- Include one objection handler
- Keep it under 100 words
"""

    return {
        "customer_id": customer_id,
        "product": product_code,
        "prompt": prompt,
        "product_context": product_context[:500],
        "customer_summary": f"Age {profile.get('age')}, {profile.get('segment')}, "
                           f"{profile.get('total_products_held')} products, "
                           f"S${profile.get('household_income', 0):,.0f} income",
    }
