"""
Causal uplift modelling using the T-Learner approach.

Instead of asking "Who is likely to buy?" (propensity), we ask
"Who is likely to buy BECAUSE of our recommendation?" (uplift).

T-Learner approach:
  - Train Model_T on treatment group: P(buy | recommended)
  - Train Model_C on control group: P(buy | not recommended)
  - Uplift = Model_T(x) - Model_C(x)

Customer segments (uplift quadrant):
  - Persuadables: high uplift → TARGET (RM time creates value)
  - Sure Things: high propensity, low uplift → SKIP (buy anyway)
  - Sleeping Dogs: negative uplift → AVOID (recommendation hurts)
  - Lost Causes: low propensity, low uplift → SKIP (nothing works)
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import ks_2samp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models"

# Feature columns to exclude from modelling
EXCLUDE_COLS = [
    "customer_id", "product_code", "campaign_month",
    "treatment_flag", "outcome", "snapshot_date",
    "join_cohort", "sex", "segment", "join_channel",
    "province_name", "first_contract_date", "last_primary_date",
    "employee_index", "country", "customer_type", "relation_type",
    "is_resident", "is_foreigner", "is_spouse_employee", "is_deceased",
]


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select numeric features and fill NaN."""
    drop = [c for c in EXCLUDE_COLS if c in df.columns]
    X = df.drop(columns=drop, errors="ignore")
    X = X.select_dtypes(include=[np.number]).fillna(0)
    return X


# ---------------------------------------------------------------------------
# T-Learner
# ---------------------------------------------------------------------------

class TLearnerUplift:
    """
    T-Learner: separate models for treatment and control groups.
    Uplift = P(outcome=1 | treatment) - P(outcome=1 | control)
    """

    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05, max_depth: int = 5):
        self.params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "verbose": -1,
            "random_state": 42,
            "is_unbalance": True,
        }
        self.model_treatment: lgb.LGBMClassifier | None = None
        self.model_control: lgb.LGBMClassifier | None = None
        self.feature_names: list[str] = []

    def fit(self, df: pd.DataFrame) -> "TLearnerUplift":
        """
        Fit T-Learner on campaign dataset.

        Args:
            df: DataFrame with columns including treatment_flag, outcome,
                and customer features.
        """
        X = _prepare_features(df)
        self.feature_names = list(X.columns)
        y = df["outcome"].values
        treatment = df["treatment_flag"].values

        # Split into treatment and control
        treat_mask = treatment == 1
        X_treat, y_treat = X[treat_mask], y[treat_mask]
        X_ctrl, y_ctrl = X[~treat_mask], y[~treat_mask]

        logger.info(
            f"Treatment group: {len(X_treat):,} samples, "
            f"{y_treat.sum():,} positives ({y_treat.mean():.4f} rate)"
        )
        logger.info(
            f"Control group: {len(X_ctrl):,} samples, "
            f"{y_ctrl.sum():,} positives ({y_ctrl.mean():.4f} rate)"
        )

        # Train treatment model
        logger.info("Training treatment model...")
        self.model_treatment = lgb.LGBMClassifier(**self.params)
        self.model_treatment.fit(X_treat, y_treat)

        # Train control model
        logger.info("Training control model...")
        self.model_control = lgb.LGBMClassifier(**self.params)
        self.model_control.fit(X_ctrl, y_ctrl)

        logger.info("T-Learner fitted")
        return self

    def predict_uplift(self, df: pd.DataFrame) -> np.ndarray:
        """Predict uplift score for each customer."""
        X = _prepare_features(df)
        # Ensure same columns as training
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]

        p_treat = self.model_treatment.predict_proba(X)[:, 1]
        p_ctrl = self.model_control.predict_proba(X)[:, 1]
        return p_treat - p_ctrl

    def predict_propensity(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Predict propensity under treatment and control."""
        X = _prepare_features(df)
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]

        return {
            "p_treatment": self.model_treatment.predict_proba(X)[:, 1],
            "p_control": self.model_control.predict_proba(X)[:, 1],
        }

    def segment_customers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign each customer to an uplift quadrant.

        Returns DataFrame with columns:
          - uplift_score
          - p_treatment, p_control
          - segment: Persuadable / Sure Thing / Sleeping Dog / Lost Cause
        """
        uplift = self.predict_uplift(df)
        propensity = self.predict_propensity(df)

        result = pd.DataFrame({
            "customer_id": df["customer_id"].values if "customer_id" in df.columns else range(len(df)),
            "uplift_score": uplift,
            "p_treatment": propensity["p_treatment"],
            "p_control": propensity["p_control"],
        })

        # Thresholds: median uplift and median treatment propensity
        uplift_threshold = np.median(uplift)
        propensity_threshold = np.median(propensity["p_treatment"])

        conditions = [
            (result["uplift_score"] >= uplift_threshold) & (result["p_treatment"] >= propensity_threshold),
            (result["uplift_score"] < uplift_threshold) & (result["p_treatment"] >= propensity_threshold),
            (result["uplift_score"] < 0),
            (result["uplift_score"] >= uplift_threshold) & (result["p_treatment"] < propensity_threshold),
        ]
        choices = ["Persuadable", "Sure Thing", "Sleeping Dog", "Lost Cause"]
        # Default for anything not caught
        result["segment"] = np.select(conditions, choices, default="Lost Cause")

        return result

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "uplift_model.pkl", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "TLearnerUplift":
        with open(path / "uplift_model.pkl", "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# Evaluation metrics for uplift models
# ---------------------------------------------------------------------------

def uplift_curve(y_true: np.ndarray, uplift_scores: np.ndarray, treatment: np.ndarray) -> dict:
    """
    Compute the uplift curve (cumulative incremental gain).

    Sort by predicted uplift descending, then compute cumulative
    difference in conversion rates between treatment and control
    at each percentile.
    """
    df = pd.DataFrame({
        "y": y_true,
        "uplift": uplift_scores,
        "treatment": treatment,
    })
    df = df.sort_values("uplift", ascending=False).reset_index(drop=True)

    n = len(df)
    percentiles = []
    incremental_gains = []

    for pct in range(5, 101, 5):
        cutoff = int(n * pct / 100)
        subset = df.iloc[:cutoff]

        treat_sub = subset[subset["treatment"] == 1]
        ctrl_sub = subset[subset["treatment"] == 0]

        if len(treat_sub) > 0 and len(ctrl_sub) > 0:
            treat_rate = treat_sub["y"].mean()
            ctrl_rate = ctrl_sub["y"].mean()
            incremental_gain = (treat_rate - ctrl_rate) * cutoff
        else:
            incremental_gain = 0

        percentiles.append(pct)
        incremental_gains.append(incremental_gain)

    return {"percentiles": percentiles, "incremental_gains": incremental_gains}


def auuc(y_true: np.ndarray, uplift_scores: np.ndarray, treatment: np.ndarray) -> float:
    """
    Area Under the Uplift Curve (AUUC).
    Higher = better targeting of persuadables.
    """
    curve = uplift_curve(y_true, uplift_scores, treatment)
    gains = curve["incremental_gains"]
    # Normalise by number of data points
    n = len(y_true)
    area = sum(gains) / (n * len(gains)) if n > 0 else 0
    return float(area)


def qini_coefficient(y_true: np.ndarray, uplift_scores: np.ndarray, treatment: np.ndarray) -> float:
    """
    Qini coefficient: AUUC of model minus AUUC of random targeting.
    Positive = model is better than random.
    """
    model_auuc = auuc(y_true, uplift_scores, treatment)
    random_auuc = auuc(y_true, np.random.default_rng(42).permutation(uplift_scores), treatment)
    return model_auuc - random_auuc


def segment_summary(segments: pd.DataFrame, y_true: np.ndarray, treatment: np.ndarray) -> pd.DataFrame:
    """
    Summary statistics per uplift segment.
    Shows conversion rates, uplift, and recommended action.
    """
    segments = segments.copy()
    segments["outcome"] = y_true
    segments["treatment_flag"] = treatment

    summary_rows = []
    for seg_name in ["Persuadable", "Sure Thing", "Sleeping Dog", "Lost Cause"]:
        seg = segments[segments["segment"] == seg_name]
        if len(seg) == 0:
            continue

        treat = seg[seg["treatment_flag"] == 1]
        ctrl = seg[seg["treatment_flag"] == 0]

        treat_rate = treat["outcome"].mean() if len(treat) > 0 else 0
        ctrl_rate = ctrl["outcome"].mean() if len(ctrl) > 0 else 0

        summary_rows.append({
            "segment": seg_name,
            "count": len(seg),
            "pct_of_total": len(seg) / len(segments) * 100,
            "treatment_conversion": treat_rate,
            "control_conversion": ctrl_rate,
            "observed_uplift": treat_rate - ctrl_rate,
            "avg_predicted_uplift": seg["uplift_score"].mean(),
            "action": {
                "Persuadable": "TARGET — RM outreach creates value",
                "Sure Thing": "SKIP — will buy anyway, save RM time",
                "Sleeping Dog": "AVOID — recommendation hurts conversion",
                "Lost Cause": "SKIP — low probability either way",
            }.get(seg_name, ""),
        })

    return pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# Drift monitoring
# ---------------------------------------------------------------------------

def check_uplift_drift(
    baseline_scores: np.ndarray,
    current_scores: np.ndarray,
    threshold: float = 0.05,
) -> dict:
    """
    KS-test for distribution shift in uplift scores.
    If p-value < threshold, the model may need retraining.
    """
    statistic, p_value = ks_2samp(baseline_scores, current_scores)
    is_drifted = p_value < threshold

    result = {
        "ks_statistic": float(statistic),
        "p_value": float(p_value),
        "threshold": threshold,
        "is_drifted": is_drifted,
        "recommendation": "Retrain model" if is_drifted else "Model is stable",
    }

    if is_drifted:
        logger.warning(f"DRIFT DETECTED: KS={statistic:.4f}, p={p_value:.6f}")
    else:
        logger.info(f"No drift: KS={statistic:.4f}, p={p_value:.6f}")

    return result


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------

def train_uplift_model(
    campaign_dataset: pd.DataFrame,
    test_size: float = 0.3,
) -> tuple[TLearnerUplift, dict]:
    """
    Train T-Learner uplift model and evaluate.
    """
    # Temporal split: use earlier months for training, later for testing
    months = sorted(campaign_dataset["campaign_month"].unique())
    split_month = months[int(len(months) * (1 - test_size))]

    train = campaign_dataset[campaign_dataset["campaign_month"] <= split_month]
    test = campaign_dataset[campaign_dataset["campaign_month"] > split_month]

    logger.info(f"Train: {len(train):,} records (up to {split_month})")
    logger.info(f"Test: {len(test):,} records (after {split_month})")

    with mlflow.start_run(run_name="uplift_training"):
        # Fit
        model = TLearnerUplift()
        model.fit(train)

        # Evaluate on test set
        test_uplift = model.predict_uplift(test)
        test_segments = model.segment_customers(test)

        # Metrics
        test_auuc = auuc(test["outcome"].values, test_uplift, test["treatment_flag"].values)
        test_qini = qini_coefficient(test["outcome"].values, test_uplift, test["treatment_flag"].values)
        test_curve = uplift_curve(test["outcome"].values, test_uplift, test["treatment_flag"].values)

        # Segment summary
        summary = segment_summary(test_segments, test["outcome"].values, test["treatment_flag"].values)
        logger.info(f"\nSegment Summary:\n{summary.to_string(index=False)}")

        # Log to MLflow
        mlflow.log_metric("test_auuc", test_auuc)
        mlflow.log_metric("test_qini", test_qini)
        mlflow.log_metric("n_persuadables", int((test_segments["segment"] == "Persuadable").sum()))
        mlflow.log_metric("n_sure_things", int((test_segments["segment"] == "Sure Thing").sum()))
        mlflow.log_metric("n_sleeping_dogs", int((test_segments["segment"] == "Sleeping Dog").sum()))
        mlflow.log_metric("n_lost_causes", int((test_segments["segment"] == "Lost Cause").sum()))

        mlflow.log_text(summary.to_string(), "segment_summary.txt")

        # Save
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.save(MODEL_DIR)
        mlflow.log_artifacts(str(MODEL_DIR), "models")

        metrics = {
            "auuc": test_auuc,
            "qini": test_qini,
            "uplift_curve": test_curve,
            "segment_summary": summary.to_dict("records"),
        }

        logger.info(f"AUUC: {test_auuc:.4f}, Qini: {test_qini:.4f}")

    return model, metrics
