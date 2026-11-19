"""Tests for SQL feature engineering queries."""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from src.features.sql_features import (
    run_query,
    build_customer_features,
    build_campaign_dataset,
    QUERY_RFM,
    QUERY_PRODUCT_PORTFOLIO,
    QUERY_PRODUCT_VELOCITY,
    QUERY_CLV,
    QUERY_COHORT,
    QUERY_COOCCURRENCE,
    QUERY_CAMPAIGN_FEATURES,
)


class TestSQLQueries:
    """Verify SQL query strings are well-formed."""

    def test_rfm_query_has_customer_id(self):
        assert "customer_id" in QUERY_RFM

    def test_rfm_query_has_recency(self):
        assert "recency_days" in QUERY_RFM

    def test_portfolio_query_has_product_counts(self):
        assert "total_products_held" in QUERY_PRODUCT_PORTFOLIO

    def test_velocity_query_has_trend(self):
        assert "velocity_trend" in QUERY_PRODUCT_VELOCITY

    def test_clv_query_has_ntile(self):
        assert "NTILE" in QUERY_CLV

    def test_cohort_query_has_zscore(self):
        assert "product_zscore_in_cohort" in QUERY_COHORT

    def test_cooccurrence_query_has_pairs(self):
        assert "pair_savings_card" in QUERY_COOCCURRENCE

    def test_campaign_query_has_treatment(self):
        assert "treatment_flag" in QUERY_CAMPAIGN_FEATURES


class TestRunQuery:
    def test_calls_pd_read_sql(self):
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        expected_df = pd.DataFrame({"customer_id": [1, 2], "value": [10, 20]})
        with patch("src.features.sql_features.pd.read_sql", return_value=expected_df) as mock_read:
            result = run_query("SELECT 1", engine=mock_engine)
            mock_read.assert_called_once()
            assert len(result) == 2


class TestBuildCustomerFeatures:
    def test_merges_all_feature_groups(self):
        """build_customer_features should call run_query 6 times and merge."""
        base = pd.DataFrame({"customer_id": [1, 2, 3]})
        dfs = [
            base.assign(recency_days=10, frequency_adds=5, monetary_income=5000,
                       seniority_months=24, days_since_first_add=100, age=30,
                       sex="M", is_active=1, segment="01", join_channel="KAT", province_code="28"),
            base.assign(total_products_held=3, n_deposit_products=2, n_wealth_products=1,
                       n_loan_products=0, has_credit_card=1, has_saving_account=1,
                       has_current_account=1, has_payroll_account=0, has_e_account=0,
                       has_funds=0, has_mortgage=0, has_pensions=0, has_loans=0,
                       has_direct_debit=0, has_securities=0, has_payroll=0),
            base.assign(avg_monthly_adds=1, max_monthly_adds=2, active_months=5,
                       recent_3m_avg_adds=1, older_avg_adds=0.5, velocity_trend=0.5),
            base.assign(clv_proxy=5000.0, income_quintile=4, product_depth_quartile=3),
            base.assign(join_cohort="2015-Q1", cohort_size=100, products_vs_cohort=1.5,
                       product_zscore_in_cohort=0.8, income_ratio_vs_cohort=1.2),
            base.assign(pair_current_debit=1, pair_payroll_both=0, pair_savings_card=1,
                       pair_mortgage_home=0, pair_pensions_both=0, pair_wealth_both=0,
                       total_natural_pairs=2),
        ]

        call_count = [0]
        def mock_run_query(query, engine=None):
            idx = call_count[0]
            call_count[0] += 1
            return dfs[idx]

        with patch("src.features.sql_features.run_query", side_effect=mock_run_query):
            result = build_customer_features(engine=MagicMock())
            assert len(result) == 3
            assert "customer_id" in result.columns
            assert "recency_days" in result.columns
            assert "clv_proxy" in result.columns
            assert "velocity_trend" in result.columns


class TestBuildCampaignDataset:
    def test_merges_campaigns_with_features(self):
        campaigns = pd.DataFrame({
            "customer_id": [1, 2],
            "product_code": ["savings", "credit_card"],
            "campaign_month": ["2016-04", "2016-04"],
            "treatment_flag": [1, 0],
            "outcome": [1, 0],
        })
        features = pd.DataFrame({
            "customer_id": [1, 2, 3],
            "age": [30, 40, 50],
            "income": [5000, 8000, 3000],
        })

        with patch("src.features.sql_features.run_query", return_value=campaigns):
            with patch("src.features.sql_features.build_customer_features", return_value=features):
                result = build_campaign_dataset(engine=MagicMock())
                assert len(result) == 2
                assert "treatment_flag" in result.columns
                assert "age" in result.columns
