"""Tests for the hybrid recommender (ALS + LGBMRanker)."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from src.models.recommender import (
    CollaborativeFilter,
    build_interaction_matrix,
    build_ranker_training_data,
    _ndcg_at_k,
    PRODUCT_CODES,
)


@pytest.fixture
def sample_product_changes():
    """Product change events for building interactions."""
    return pd.DataFrame({
        "customer_id": [1, 1, 1, 2, 2, 3, 3, 3, 3],
        "product_code": [
            "saving_account", "credit_card", "payroll",
            "saving_account", "funds",
            "saving_account", "credit_card", "mortgage", "funds",
        ],
        "change_type": ["added"] * 9,
        "change_date": pd.to_datetime([
            "2016-01-28", "2016-02-28", "2016-03-28",
            "2016-01-28", "2016-04-28",
            "2016-01-28", "2016-02-28", "2016-03-28", "2016-04-28",
        ]),
    })


class TestBuildInteractionMatrix:
    def test_returns_dataframe(self, sample_product_changes):
        result = build_interaction_matrix(sample_product_changes)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, sample_product_changes):
        result = build_interaction_matrix(sample_product_changes)
        assert "customer_id" in result.columns
        assert "product_code" in result.columns
        assert "weight" in result.columns

    def test_weights_are_positive(self, sample_product_changes):
        result = build_interaction_matrix(sample_product_changes)
        assert (result["weight"] > 0).all()

    def test_recent_items_weighted_higher(self, sample_product_changes):
        result = build_interaction_matrix(sample_product_changes)
        # Customer 3's funds (most recent) should have higher weight than saving_account (oldest)
        cust3 = result[result["customer_id"] == 3]
        funds_w = cust3[cust3["product_code"] == "funds"]["weight"].values[0]
        savings_w = cust3[cust3["product_code"] == "saving_account"]["weight"].values[0]
        assert funds_w >= savings_w

    def test_raises_on_no_additions(self):
        empty = pd.DataFrame({
            "customer_id": [1],
            "product_code": ["savings"],
            "change_type": ["removed"],
            "change_date": pd.to_datetime(["2016-01-28"]),
        })
        with pytest.raises(ValueError, match="No product additions"):
            build_interaction_matrix(empty)


class TestCollaborativeFilter:
    def test_fit_and_score(self, sample_product_changes):
        interactions = build_interaction_matrix(sample_product_changes)
        cf = CollaborativeFilter(factors=8, iterations=5)
        cf.fit(interactions)

        # Customer 1 holds saving_account — score should be non-zero
        score = cf.score(1, "saving_account")
        assert isinstance(score, float)

    def test_unknown_customer_returns_zero(self, sample_product_changes):
        interactions = build_interaction_matrix(sample_product_changes)
        cf = CollaborativeFilter(factors=8, iterations=5)
        cf.fit(interactions)

        score = cf.score(99999, "saving_account")
        assert score == 0.0

    def test_unknown_product_returns_zero(self, sample_product_changes):
        interactions = build_interaction_matrix(sample_product_changes)
        cf = CollaborativeFilter(factors=8, iterations=5)
        cf.fit(interactions)

        score = cf.score(1, "nonexistent_product")
        assert score == 0.0

    def test_score_batch(self, sample_product_changes):
        interactions = build_interaction_matrix(sample_product_changes)
        cf = CollaborativeFilter(factors=8, iterations=5)
        cf.fit(interactions)

        pairs = pd.DataFrame({
            "customer_id": [1, 2, 3],
            "product_code": ["funds", "credit_card", "saving_account"],
        })
        scores = cf.score_batch(pairs)
        assert len(scores) == 3
        assert all(np.isfinite(scores))


class TestNDCG:
    def test_perfect_ranking(self):
        y_true = np.array([1, 1, 0, 0, 0])
        y_score = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert _ndcg_at_k(y_true, y_score, k=5) == pytest.approx(1.0)

    def test_all_zeros(self):
        y_true = np.array([0, 0, 0])
        y_score = np.array([3.0, 2.0, 1.0])
        assert _ndcg_at_k(y_true, y_score, k=3) == 0.0

    def test_k_larger_than_array(self):
        y_true = np.array([1, 0])
        y_score = np.array([2.0, 1.0])
        result = _ndcg_at_k(y_true, y_score, k=10)
        assert 0.0 <= result <= 1.0
