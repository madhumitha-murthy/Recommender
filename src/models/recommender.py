"""
Hybrid recommendation model: LightGBM ranker with ALS collaborative
filtering scores as a feature.

Approach:
  1. ALS matrix factorisation on the customer-product interaction matrix
     → produces a "collaborative score" per customer-product pair
  2. LightGBM ranker combines the ALS score with customer features
     (RFM, portfolio, velocity, CLV, cohort, co-occurrence)
     → final ranked product list per customer
  3. Optuna hyperparameter tuning with cross-validation
  4. SHAP explanations for feature importance

This is ONE model done deeply, not three shallow models.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import optuna
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PRODUCT_CODES = [
    "saving_account", "guarantees", "current_account", "derivada_account",
    "payroll_account", "junior_account", "mas_particular", "particular_account",
    "particular_plus", "short_term_deposit", "medium_term_deposit",
    "long_term_deposit", "e_account", "funds", "mortgage", "pensions",
    "loans", "taxes", "credit_card", "securities", "home_account",
    "payroll", "pension_nom", "direct_debit",
]

MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models"


# ---------------------------------------------------------------------------
# 1. ALS Collaborative Filtering
# ---------------------------------------------------------------------------

class CollaborativeFilter:
    """
    ALS matrix factorisation on customer-product interaction matrix.
    Produces a score per customer-product pair that captures latent
    co-purchase patterns ("customers like you bought X").
    """

    def __init__(self, factors: int = 64, iterations: int = 15, regularization: float = 0.1):
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.model: Optional[AlternatingLeastSquares] = None
        self.customer_encoder = LabelEncoder()
        self.product_encoder = LabelEncoder()

    def fit(self, interactions: pd.DataFrame) -> "CollaborativeFilter":
        """
        Fit ALS on a customer-product interaction DataFrame.

        Args:
            interactions: DataFrame with columns [customer_id, product_code, weight]
                          weight = 1 for holding, could be higher for recent adds
        """
        logger.info(f"Fitting ALS (factors={self.factors}, iterations={self.iterations})...")

        customer_ids = self.customer_encoder.fit_transform(interactions["customer_id"])
        product_ids = self.product_encoder.fit_transform(interactions["product_code"])
        weights = interactions["weight"].values.astype(np.float32)

        # Sparse matrix: customers x products
        n_customers = len(self.customer_encoder.classes_)
        n_products = len(self.product_encoder.classes_)
        interaction_matrix = csr_matrix(
            (weights, (customer_ids, product_ids)),
            shape=(n_customers, n_products),
        )

        self.model = AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization,
            random_state=42,
        )
        self.model.fit(interaction_matrix)

        logger.info(f"ALS fitted: {n_customers:,} customers x {n_products} products")
        return self

    def score(self, customer_id: int, product_code: str) -> float:
        """Get ALS score for a single customer-product pair."""
        try:
            cust_idx = self.customer_encoder.transform([customer_id])[0]
            prod_idx = self.product_encoder.transform([product_code])[0]
            user_factors = self.model.user_factors[cust_idx]
            item_factors = self.model.item_factors[prod_idx]
            return float(np.dot(user_factors, item_factors))
        except (ValueError, IndexError):
            return 0.0

    def score_batch(self, pairs: pd.DataFrame) -> np.ndarray:
        """
        Score a batch of customer-product pairs.

        Args:
            pairs: DataFrame with columns [customer_id, product_code]
        Returns:
            Array of ALS scores
        """
        scores = np.zeros(len(pairs))
        for i, (_, row) in enumerate(pairs.iterrows()):
            scores[i] = self.score(row["customer_id"], row["product_code"])
        return scores

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "als_model.pkl", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "CollaborativeFilter":
        with open(path / "als_model.pkl", "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# 2. Build training data for the ranker
# ---------------------------------------------------------------------------

def build_interaction_matrix(product_changes: pd.DataFrame) -> pd.DataFrame:
    """
    Build customer-product interaction matrix from product change events.
    Weight recent additions higher than older ones.
    """
    adds = product_changes[product_changes["change_type"] == "added"].copy()
    if adds.empty:
        raise ValueError("No product additions found in product_changes")

    adds["change_date"] = pd.to_datetime(adds["change_date"])
    max_date = adds["change_date"].max()
    # Weight: more recent = higher weight (exponential decay)
    adds["days_ago"] = (max_date - adds["change_date"]).dt.days
    adds["weight"] = np.exp(-adds["days_ago"] / 365.0)  # half-life ~1 year
    adds["weight"] = adds["weight"].clip(lower=0.1)

    interactions = (
        adds.groupby(["customer_id", "product_code"])["weight"]
        .sum()
        .reset_index()
    )
    return interactions


def build_ranker_training_data(
    customer_features: pd.DataFrame,
    product_changes: pd.DataFrame,
    als: CollaborativeFilter,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Build pairwise training data for the LightGBM ranker.

    For each customer:
      - Positive examples: products the customer added (label=1)
      - Negative examples: products the customer did NOT add (label=0)
      - Features: customer features + ALS score for this pair

    Returns:
        (features_df, labels, group_sizes) for LGBMRanker
    """
    logger.info("Building ranker training data...")

    adds = product_changes[product_changes["change_type"] == "added"].copy()
    adds["change_date"] = pd.to_datetime(adds["change_date"])
    # Use last 6 months of data for training labels
    max_date = adds["change_date"].max()
    cutoff = max_date - pd.DateOffset(months=6)
    recent_adds = adds[adds["change_date"] >= cutoff]

    # Get set of (customer, product) pairs that were added
    positive_pairs = set(
        zip(recent_adds["customer_id"], recent_adds["product_code"])
    )

    # Build candidate set: for each customer, score all products
    customers = customer_features["customer_id"].unique()

    rows = []
    labels = []
    groups = []

    for cust_id in customers:
        cust_rows = []
        cust_labels = []

        cust_features = customer_features[
            customer_features["customer_id"] == cust_id
        ].iloc[0].to_dict()

        for product_code in PRODUCT_CODES:
            # Skip products already held (we only recommend NEW products)
            held_col = f"has_{product_code}"
            if held_col in cust_features and cust_features[held_col] == 1:
                continue

            # ALS score
            als_score = als.score(cust_id, product_code)

            # Product index
            prod_idx = PRODUCT_CODES.index(product_code)

            row = {
                **{k: v for k, v in cust_features.items() if k != "customer_id"},
                "als_score": als_score,
                "product_idx": prod_idx,
            }
            cust_rows.append(row)

            label = 1 if (cust_id, product_code) in positive_pairs else 0
            cust_labels.append(label)

        if cust_rows:
            rows.extend(cust_rows)
            labels.extend(cust_labels)
            groups.append(len(cust_rows))

    features_df = pd.DataFrame(rows)
    labels_arr = np.array(labels)
    groups_arr = np.array(groups)

    logger.info(
        f"Ranker data: {len(features_df):,} pairs, "
        f"{labels_arr.sum():,} positives, "
        f"{len(groups_arr):,} customer groups"
    )
    return features_df, labels_arr, groups_arr


# ---------------------------------------------------------------------------
# 3. LightGBM Ranker with Optuna tuning
# ---------------------------------------------------------------------------

def _get_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select only numeric columns and fill NaN."""
    numeric = df.select_dtypes(include=[np.number]).copy()
    numeric = numeric.fillna(0)
    return numeric


def _ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 5) -> float:
    """Compute NDCG@K for a single query group."""
    order = np.argsort(-y_score)[:k]
    gains = y_true[order]
    discounts = np.log2(np.arange(1, len(gains) + 1) + 1)
    dcg = np.sum(gains / discounts)

    ideal_order = np.argsort(-y_true)[:k]
    ideal_gains = y_true[ideal_order]
    idcg = np.sum(ideal_gains / discounts[: len(ideal_gains)])

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ranker(
    model: lgb.LGBMRanker,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    k_values: list[int] = [3, 5, 10],
) -> dict[str, float]:
    """Evaluate ranker with NDCG@K and Precision@K."""
    scores = model.predict(X)

    ndcg_scores = {f"ndcg@{k}": [] for k in k_values}
    precision_scores = {f"precision@{k}": [] for k in k_values}

    idx = 0
    for group_size in groups:
        group_y = y[idx: idx + group_size]
        group_scores = scores[idx: idx + group_size]
        idx += group_size

        for k in k_values:
            ndcg_scores[f"ndcg@{k}"].append(_ndcg_at_k(group_y, group_scores, k))

            top_k_idx = np.argsort(-group_scores)[:k]
            hits = group_y[top_k_idx].sum()
            precision_scores[f"precision@{k}"].append(hits / min(k, group_size))

    metrics = {}
    for key in ndcg_scores:
        metrics[key] = float(np.mean(ndcg_scores[key]))
    for key in precision_scores:
        metrics[key] = float(np.mean(precision_scores[key]))

    return metrics


def tune_and_train(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    groups_val: np.ndarray,
    n_trials: int = 50,
) -> tuple[lgb.LGBMRanker, dict]:
    """
    Optuna hyperparameter tuning for LGBMRanker.
    Returns the best model and its metrics.
    """
    logger.info(f"Starting Optuna tuning ({n_trials} trials)...")

    X_train_num = _get_numeric_features(X_train)
    X_val_num = _get_numeric_features(X_val)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            verbose=-1,
            random_state=42,
            **params,
        )
        model.fit(
            X_train_num, y_train, group=groups_train,
            eval_set=[(X_val_num, y_val)],
            eval_group=[groups_val],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        metrics = evaluate_ranker(model, X_val_num, y_val, groups_val)
        return metrics["ndcg@5"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"Best trial: NDCG@5 = {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    # Retrain with best params
    best_model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        verbose=-1,
        random_state=42,
        **study.best_params,
    )
    best_model.fit(
        X_train_num, y_train, group=groups_train,
        eval_set=[(X_val_num, y_val)],
        eval_group=[groups_val],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    # Final evaluation
    train_metrics = evaluate_ranker(best_model, X_train_num, y_train, groups_train)
    val_metrics = evaluate_ranker(best_model, X_val_num, y_val, groups_val)

    logger.info("Train metrics: " + ", ".join(f"{k}={v:.4f}" for k, v in train_metrics.items()))
    logger.info("Val metrics: " + ", ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))

    return best_model, {"train": train_metrics, "val": val_metrics, "best_params": study.best_params}


# ---------------------------------------------------------------------------
# 4. SHAP explanations
# ---------------------------------------------------------------------------

def compute_shap_importance(model: lgb.LGBMRanker, X: pd.DataFrame) -> pd.DataFrame:
    """Compute SHAP feature importance."""
    try:
        import shap
        X_num = _get_numeric_features(X)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_num.sample(min(5000, len(X_num)), random_state=42))
        importance = pd.DataFrame({
            "feature": X_num.columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
        return importance
    except ImportError:
        logger.warning("shap not installed, using built-in feature importance")
        importance = pd.DataFrame({
            "feature": model.feature_name_,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        return importance


# ---------------------------------------------------------------------------
# 5. Full training pipeline
# ---------------------------------------------------------------------------

def train_recommender(
    customer_features: pd.DataFrame,
    product_changes: pd.DataFrame,
    n_optuna_trials: int = 50,
) -> tuple[CollaborativeFilter, lgb.LGBMRanker, dict]:
    """
    Full training pipeline:
      1. Fit ALS on interaction matrix
      2. Build ranker training data with ALS scores as features
      3. Train/val split by customer group
      4. Optuna tune LGBMRanker
      5. Log to MLflow
    """
    # 1. ALS
    interactions = build_interaction_matrix(product_changes)
    als = CollaborativeFilter(factors=64, iterations=15)
    als.fit(interactions)

    # 2. Build ranker data
    X, y, groups = build_ranker_training_data(customer_features, product_changes, als)

    # 3. Train/val split by customer groups (80/20)
    n_groups = len(groups)
    split_idx = int(n_groups * 0.8)

    train_end = sum(groups[:split_idx])
    X_train, X_val = X.iloc[:train_end], X.iloc[train_end:]
    y_train, y_val = y[:train_end], y[train_end:]
    groups_train, groups_val = groups[:split_idx], groups[split_idx:]

    # 4. Tune and train
    with mlflow.start_run(run_name="recommender_training"):
        best_model, metrics = tune_and_train(
            X_train, y_train, groups_train,
            X_val, y_val, groups_val,
            n_trials=n_optuna_trials,
        )

        # Log to MLflow
        mlflow.log_params(metrics["best_params"])
        for split in ["train", "val"]:
            for metric_name, value in metrics[split].items():
                safe_name = metric_name.replace("@", "_at_")
                mlflow.log_metric(f"{split}_{safe_name}", value)

        # SHAP importance
        importance = compute_shap_importance(best_model, X_val)
        mlflow.log_text(importance.to_string(), "feature_importance.txt")

        # Save models
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        als.save(MODEL_DIR)
        with open(MODEL_DIR / "lgbm_ranker.pkl", "wb") as f:
            pickle.dump(best_model, f)
        mlflow.log_artifacts(str(MODEL_DIR), "models")

        logger.info("Models saved and logged to MLflow")

    return als, best_model, metrics


# ---------------------------------------------------------------------------
# 6. Inference: recommend top-K products for a customer
# ---------------------------------------------------------------------------

def recommend(
    customer_id: int,
    customer_features: pd.DataFrame,
    als: CollaborativeFilter,
    ranker: lgb.LGBMRanker,
    k: int = 5,
) -> pd.DataFrame:
    """
    Recommend top-K new products for a customer.

    Returns:
        DataFrame with columns [product_code, score, rank]
    """
    cust = customer_features[customer_features["customer_id"] == customer_id]
    if cust.empty:
        return pd.DataFrame(columns=["product_code", "score", "rank"])

    cust_dict = cust.iloc[0].to_dict()

    candidates = []
    for product_code in PRODUCT_CODES:
        held_col = f"has_{product_code}"
        if held_col in cust_dict and cust_dict[held_col] == 1:
            continue

        als_score = als.score(customer_id, product_code)
        prod_idx = PRODUCT_CODES.index(product_code)

        row = {
            **{k_: v for k_, v in cust_dict.items() if k_ != "customer_id"},
            "als_score": als_score,
            "product_idx": prod_idx,
        }
        candidates.append({"product_code": product_code, "features": row})

    if not candidates:
        return pd.DataFrame(columns=["product_code", "score", "rank"])

    features_df = pd.DataFrame([c["features"] for c in candidates])
    features_num = _get_numeric_features(features_df)
    scores = ranker.predict(features_num)

    results = pd.DataFrame({
        "product_code": [c["product_code"] for c in candidates],
        "score": scores,
    })
    results = results.sort_values("score", ascending=False).head(k).reset_index(drop=True)
    results["rank"] = range(1, len(results) + 1)

    return results
