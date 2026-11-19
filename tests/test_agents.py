"""Tests for the LangGraph agentic copilot and tools."""

import os
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from src.agents.copilot import SYSTEM_PROMPT, TOOLS, AgentState


class TestSystemPrompt:
    def test_mentions_uplift(self):
        assert "uplift" in SYSTEM_PROMPT.lower()

    def test_mentions_sleeping_dogs(self):
        assert "Sleeping Dog" in SYSTEM_PROMPT

    def test_mentions_all_tools(self):
        assert "customer_lookup" in SYSTEM_PROMPT
        assert "recommend_and_score" in SYSTEM_PROMPT
        assert "generate_pitch" in SYSTEM_PROMPT


class TestToolsList:
    def test_has_three_tools(self):
        assert len(TOOLS) == 3

    def test_tool_names(self):
        names = [t.name for t in TOOLS]
        assert "customer_lookup" in names
        assert "recommend_and_score" in names
        assert "generate_pitch" in names


class TestProductDocs:
    def test_all_product_docs_exist(self):
        docs_dir = os.path.join(
            os.path.dirname(__file__), "..", "src", "agents", "product_docs"
        )
        expected_files = [
            "credit_card.md",
            "savings_account.md",
            "funds.md",
            "direct_debit.md",
            "payroll.md",
        ]
        for f in expected_files:
            path = os.path.join(docs_dir, f)
            assert os.path.exists(path), f"Missing product doc: {f}"

    def test_product_docs_have_key_sections(self):
        docs_dir = os.path.join(
            os.path.dirname(__file__), "..", "src", "agents", "product_docs"
        )
        for f in os.listdir(docs_dir):
            if f.endswith(".md"):
                path = os.path.join(docs_dir, f)
                with open(path) as fh:
                    content = fh.read()
                    assert len(content) > 100, f"Product doc {f} seems too short"
                    assert "## Best For" in content, f"{f} missing Best For"

    def test_product_docs_have_objection_handlers(self):
        docs_dir = os.path.join(
            os.path.dirname(__file__), "..", "src", "agents", "product_docs"
        )
        for f in os.listdir(docs_dir):
            if f.endswith(".md"):
                path = os.path.join(docs_dir, f)
                with open(path) as fh:
                    content = fh.read()
                    assert "## Common Objections" in content, f"{f} missing objection handlers"


class TestCopilotGraphCreation:
    def test_create_copilot(self):
        from src.agents.copilot import create_copilot
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = MagicMock()
        graph = create_copilot(mock_llm)
        assert graph is not None


class TestCustomerLookupTool:
    def test_returns_profile_dict(self):
        features = pd.DataFrame({
            "customer_id": [12345],
            "age": [35],
            "segment": ["01 - TOP"],
            "seniority_months": [24],
            "monetary_income": [80000],
            "is_active": [1],
            "total_products_held": [5],
            "n_deposit_products": [3],
            "n_wealth_products": [1],
            "n_loan_products": [0],
            "has_credit_card": [1],
            "recency_days": [10],
            "frequency_adds": [3],
            "clv_proxy": [25000.0],
            "velocity_trend": [0.5],
        })
        with patch("src.agents.tools._load_customer_features", return_value=features):
            from src.agents.tools import customer_lookup
            result = customer_lookup.invoke({"customer_id": 12345})
            assert result["customer_id"] == 12345
            assert result["age"] == 35

    def test_handles_missing_customer(self):
        features = pd.DataFrame({
            "customer_id": [1],
            "age": [30],
        })
        with patch("src.agents.tools._load_customer_features", return_value=features):
            from src.agents.tools import customer_lookup
            result = customer_lookup.invoke({"customer_id": 99999})
            assert "error" in result
