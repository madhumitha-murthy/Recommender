"""Tests for the T-Learner uplift model."""

import pytest
import numpy as np
import pandas as pd

from src.models.uplift import (
    TLearnerUplift,
    check_uplift_drift,
    uplift_curve,
    auuc,
    _prepare_features,
)


@pytest.fixture
def sample_campaign_data():
    """Generate synthetic campaign data matching expected schema."""
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "customer_id": range(n),
        "income": np.random.normal(5000, 2000, n).clip(1000),
        "age": np.random.randint(21, 65, n),
        "seniority_months": np.random.randint(1, 60, n),
        "total_products_held": np.random.randint(0, 8, n),
        "recency_days": np.random.randint(0, 365, n),
        "frequency_adds": np.random.randint(0, 10, n),
        "clv_proxy": np.random.uniform(0, 50000, n),
        "treatment_flag": np.random.binomial(1, 0.5, n),
    })
    # Outcome with uplift effect for high-income customers
    base_prob = 0.1 + 0.03 * (df["total_products_held"] / 8)
    uplift_effect = 0.12 * (df["income"] > 5000).astype(float)
    prob = base_prob + df["treatment_flag"] * uplift_effect
    df["outcome"] = np.random.binomial(1, np.clip(prob, 0, 1))
    return df


class TestPrepareFeatures:
    def test_drops_excluded_columns(self, sample_campaign_data):
        result = _prepare_features(sample_campaign_data)
        assert "customer_id" not in result.columns
        assert "treatment_flag" not in result.columns
        assert "outcome" not in result.columns

    def test_only_numeric(self, sample_campaign_data):
        result = _prepare_features(sample_campaign_data)
        assert all(np.issubdtype(result[c].dtype, np.number) for c in result.columns)

    def test_no_nans(self, sample_campaign_data):
        df = sample_campaign_data.copy()
        df.loc[0, "income"] = np.nan
        result = _prepare_features(df)
        assert result.isna().sum().sum() == 0


class TestTLearnerUplift:
    def test_fit(self, sample_campaign_data):
        model = TLearnerUplift()
        model.fit(sample_campaign_data)
        assert model.model_treatment is not None
        assert model.model_control is not None

    def test_predict_uplift_shape(self, sample_campaign_data):
        model = TLearnerUplift()
        model.fit(sample_campaign_data)
        uplift = model.predict_uplift(sample_campaign_data)
        assert len(uplift) == len(sample_campaign_data)

    def test_predict_uplift_bounded(self, sample_campaign_data):
        model = TLearnerUplift()
        model.fit(sample_campaign_data)
        uplift = model.predict_uplift(sample_campaign_data)
        assert uplift.min() >= -1.0
        assert uplift.max() <= 1.0

    def test_predict_propensity(self, sample_campaign_data):
        model = TLearnerUplift()
        model.fit(sample_campaign_data)
        prop = model.predict_propensity(sample_campaign_data)
        assert "p_treatment" in prop
        assert "p_control" in prop
        assert len(prop["p_treatment"]) == len(sample_campaign_data)
        assert (prop["p_treatment"] >= 0).all()
        assert (prop["p_treatment"] <= 1).all()

    def test_segment_customers(self, sample_campaign_data):
        model = TLearnerUplift()
        model.fit(sample_campaign_data)
        segments = model.segment_customers(sample_campaign_data)
        assert "segment" in segments.columns
        assert "uplift_score" in segments.columns
        valid = {"Persuadable", "Sure Thing", "Sleeping Dog", "Lost Cause"}
        assert set(segments["segment"].unique()).issubset(valid)

    def test_at_least_two_segments(self, sample_campaign_data):
        model = TLearnerUplift()
        model.fit(sample_campaign_data)
        segments = model.segment_customers(sample_campaign_data)
        assert segments["segment"].nunique() >= 2

    def test_feature_names_stored(self, sample_campaign_data):
        model = TLearnerUplift()
        model.fit(sample_campaign_data)
        assert len(model.feature_names) > 0
        assert "income" in model.feature_names


class TestDriftMonitoring:
    def test_no_drift_same_data(self):
        scores = np.random.normal(0.05, 0.1, 200)
        result = check_uplift_drift(scores, scores, threshold=0.05)
        assert result["is_drifted"] == False

    def test_drift_on_shifted_data(self):
        ref = np.random.normal(0.05, 0.1, 200)
        shifted = ref + 0.5
        result = check_uplift_drift(ref, shifted, threshold=0.05)
        assert result["is_drifted"] == True

    def test_result_has_expected_keys(self):
        ref = np.random.normal(0, 0.1, 100)
        result = check_uplift_drift(ref, ref)
        assert "ks_statistic" in result
        assert "p_value" in result
        assert "is_drifted" in result
        assert "recommendation" in result


class TestUpliftCurve:
    def test_returns_expected_structure(self):
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 10)
        uplift = np.random.normal(0, 0.1, 100)
        treatment = np.random.binomial(1, 0.5, 100)
        result = uplift_curve(y, uplift, treatment)
        assert "percentiles" in result
        assert "incremental_gains" in result
        assert len(result["percentiles"]) == 20  # 5% increments to 100%


class TestAUUC:
    def test_returns_float(self):
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 10)
        uplift = np.random.normal(0, 0.1, 100)
        treatment = np.random.binomial(1, 0.5, 100)
        result = auuc(y, uplift, treatment)
        assert isinstance(result, float)
