"""Tests for FastAPI scoring endpoint."""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.api.serve import app, PRODUCT_NAMES


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_health_shows_model_status(self, client):
        response = client.get("/health")
        data = response.json()
        assert "models_loaded" in data


class TestRecommendEndpoint:
    def test_returns_503_without_models(self, client):
        response = client.post("/recommend", json={"customer_id": 12345, "top_k": 3})
        assert response.status_code == 503

    def test_validates_top_k_min(self, client):
        response = client.post("/recommend", json={"customer_id": 1, "top_k": 0})
        assert response.status_code == 422

    def test_validates_top_k_max(self, client):
        response = client.post("/recommend", json={"customer_id": 1, "top_k": 11})
        assert response.status_code == 422

    def test_requires_customer_id(self, client):
        response = client.post("/recommend", json={"top_k": 3})
        assert response.status_code == 422


class TestProductNames:
    def test_product_mapping_has_24_products(self):
        assert len(PRODUCT_NAMES) == 24

    def test_credit_card_mapping(self):
        assert PRODUCT_NAMES["ind_tjcr_fin_ult1"] == "Credit Card"

    def test_savings_mapping(self):
        assert PRODUCT_NAMES["ind_ahor_fin_ult1"] == "Savings Account"

    def test_all_values_are_strings(self):
        for code, name in PRODUCT_NAMES.items():
            assert isinstance(code, str)
            assert isinstance(name, str)
            assert len(name) > 0
