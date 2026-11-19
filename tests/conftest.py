"""Shared test fixtures."""

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def sample_customer_snapshot():
    """Single customer snapshot row matching the schema."""
    return pd.DataFrame([{
        "ncodpers": 12345,
        "fecha_dato": "2016-05-28",
        "ind_empleado": "N",
        "pais_residencia": "ES",
        "sexo": "V",
        "age": 35,
        "fecha_alta": "2012-01-15",
        "ind_nuevo": 0,
        "antiguedad": 52,
        "indrel": 1,
        "ind_actividad_cliente": 1,
        "renta": 80000.0,
        "segmento": "01 - TOP",
        "ind_ahor_fin_ult1": 1,
        "ind_aval_fin_ult1": 0,
        "ind_cco_fin_ult1": 1,
        "ind_cder_fin_ult1": 0,
        "ind_cno_fin_ult1": 0,
        "ind_ctju_fin_ult1": 0,
        "ind_ctma_fin_ult1": 0,
        "ind_ctop_fin_ult1": 1,
        "ind_ctpp_fin_ult1": 0,
        "ind_deco_fin_ult1": 0,
        "ind_deme_fin_ult1": 0,
        "ind_dela_fin_ult1": 0,
        "ind_ecue_fin_ult1": 1,
        "ind_fond_fin_ult1": 0,
        "ind_hip_fin_ult1": 0,
        "ind_plan_fin_ult1": 0,
        "ind_pres_fin_ult1": 0,
        "ind_reca_fin_ult1": 0,
        "ind_tjcr_fin_ult1": 1,
        "ind_valo_fin_ult1": 0,
        "ind_viv_fin_ult1": 0,
        "ind_nomina_ult1": 0,
        "ind_nom_pens_ult1": 0,
        "ind_recibo_ult1": 1,
    }])


@pytest.fixture
def sample_product_changes():
    """Sample product change events for uplift training."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "ncodpers": np.random.randint(1, 100, n),
        "product": np.random.choice(
            ["savings", "credit_card", "funds", "payroll", "direct_debit"], n
        ),
        "change_type": np.random.choice(["added", "removed"], n, p=[0.7, 0.3]),
        "month": np.random.choice(["2016-03", "2016-04", "2016-05"], n),
        "treatment_group": np.random.binomial(1, 0.5, n),
    })
