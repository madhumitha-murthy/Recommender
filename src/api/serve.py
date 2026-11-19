"""FastAPI scoring endpoint for Next Best Product recommendations."""

import os
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import psycopg2

app = FastAPI(
    title="Next Best Product Recommender API",
    description="Scoring API for personalised banking product recommendations with uplift-aware ranking.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class RecommendRequest(BaseModel):
    customer_id: int = Field(..., description="Unique customer identifier (ncodpers)")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of recommendations to return")


class ProductRecommendation(BaseModel):
    product_code: str
    product_name: str
    score: float = Field(..., description="LGBMRanker relevance score")
    uplift: float = Field(..., description="Estimated incremental conversion probability")
    segment: str = Field(..., description="Uplift segment: Persuadable / Sure Thing / Lost Cause / Sleeping Dog")


class RecommendResponse(BaseModel):
    customer_id: int
    recommendations: list[ProductRecommendation]


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool


# ---------------------------------------------------------------------------
# Product code → name mapping (matches configs/products.yaml)
# ---------------------------------------------------------------------------

PRODUCT_NAMES = {
    "ind_ahor_fin_ult1": "Savings Account",
    "ind_aval_fin_ult1": "Guarantees",
    "ind_cco_fin_ult1": "Current Account",
    "ind_cder_fin_ult1": "Derivada Account",
    "ind_cno_fin_ult1": "Payroll Account",
    "ind_ctju_fin_ult1": "Junior Account",
    "ind_ctma_fin_ult1": "Mas Particular Account",
    "ind_ctop_fin_ult1": "Particular Account",
    "ind_ctpp_fin_ult1": "Particular Plus Account",
    "ind_deco_fin_ult1": "Short-term Deposits",
    "ind_deme_fin_ult1": "Medium-term Deposits",
    "ind_dela_fin_ult1": "Long-term Deposits",
    "ind_ecue_fin_ult1": "e-Account",
    "ind_fond_fin_ult1": "Funds",
    "ind_hip_fin_ult1": "Mortgage",
    "ind_plan_fin_ult1": "Pensions",
    "ind_pres_fin_ult1": "Loans",
    "ind_reca_fin_ult1": "Taxes",
    "ind_tjcr_fin_ult1": "Credit Card",
    "ind_valo_fin_ult1": "Securities",
    "ind_viv_fin_ult1": "Home Account",
    "ind_nomina_ult1": "Payroll",
    "ind_nom_pens_ult1": "Pensions (nom)",
    "ind_recibo_ult1": "Direct Debit",
}

# ---------------------------------------------------------------------------
# Model singletons (loaded once at startup)
# ---------------------------------------------------------------------------

_ranker = None
_uplift_model = None
_cf_model = None


def _get_db_conn():
    return psycopg2.connect(os.getenv("DATABASE_URL", "postgresql://localhost:5432/nbp"))


def _load_models():
    global _ranker, _uplift_model, _cf_model
    model_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models")

    ranker_path = os.path.join(model_dir, "lgbm_ranker.pkl")
    uplift_path = os.path.join(model_dir, "uplift_model.pkl")
    cf_path = os.path.join(model_dir, "cf_model.pkl")

    if os.path.exists(ranker_path):
        _ranker = joblib.load(ranker_path)
    if os.path.exists(uplift_path):
        _uplift_model = joblib.load(uplift_path)
    if os.path.exists(cf_path):
        _cf_model = joblib.load(cf_path)


@app.on_event("startup")
def startup():
    _load_models()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        models_loaded=_ranker is not None,
    )


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    if _ranker is None or _uplift_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Run training pipeline first.")

    try:
        conn = _get_db_conn()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database connection failed: {e}")

    try:
        # Fetch customer features
        query = """
        SELECT * FROM customer_snapshots
        WHERE ncodpers = %s
        ORDER BY fecha_dato DESC
        LIMIT 1
        """
        df = pd.read_sql(query, conn, params=(req.customer_id,))

        if df.empty:
            raise HTTPException(status_code=404, detail=f"Customer {req.customer_id} not found")

        # Build candidate rows (one per product the customer does NOT already hold)
        product_cols = [c for c in df.columns if c.startswith("ind_") and c.endswith("_ult1")]
        held = [c for c in product_cols if df[c].iloc[0] == 1]
        candidates = [c for c in product_cols if c not in held]

        if not candidates:
            return RecommendResponse(customer_id=req.customer_id, recommendations=[])

        # Score each candidate with the ranker
        feature_cols = [c for c in df.columns if c not in ["ncodpers", "fecha_dato"] + product_cols]
        base_features = df[feature_cols].iloc[0].to_dict()

        rows = []
        for prod in candidates:
            row = {**base_features, "candidate_product": prod}
            rows.append(row)

        score_df = pd.DataFrame(rows)

        # Encode candidate product as category ordinal
        score_df["candidate_product"] = score_df["candidate_product"].astype("category").cat.codes
        scores = _ranker.predict(score_df)

        # Uplift scores
        uplift_scores = _uplift_model.predict(score_df) if hasattr(_uplift_model, "predict") else [0.0] * len(candidates)

        # Build response
        results = []
        for i, prod in enumerate(candidates):
            uplift_val = float(uplift_scores[i]) if i < len(uplift_scores) else 0.0

            if uplift_val > 0.05:
                segment = "Persuadable"
            elif uplift_val > 0:
                segment = "Sure Thing"
            elif uplift_val > -0.05:
                segment = "Lost Cause"
            else:
                segment = "Sleeping Dog"

            results.append(ProductRecommendation(
                product_code=prod,
                product_name=PRODUCT_NAMES.get(prod, prod),
                score=float(scores[i]),
                uplift=uplift_val,
                segment=segment,
            ))

        # Rank by combined score: ranker score + uplift bonus for Persuadables
        results.sort(key=lambda r: r.score + (r.uplift * 2 if r.segment == "Persuadable" else 0), reverse=True)

        return RecommendResponse(
            customer_id=req.customer_id,
            recommendations=results[:req.top_k],
        )
    finally:
        conn.close()


@app.post("/recommend/batch")
def recommend_batch(customer_ids: list[int], top_k: int = 3):
    """Score multiple customers in one call (for RM daily briefs)."""
    results = {}
    for cid in customer_ids[:50]:  # cap at 50 per batch
        try:
            resp = recommend(RecommendRequest(customer_id=cid, top_k=top_k))
            results[cid] = resp.recommendations
        except HTTPException:
            results[cid] = []
    return results
