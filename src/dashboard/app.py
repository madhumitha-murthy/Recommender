"""
Streamlit dashboard for the Next Best Product recommendation system.

Panels:
  1. Uplift Quadrant — scatter plot of persuadables/sure things/sleeping dogs/lost causes
  2. Customer Segment Explorer — drill into any segment
  3. Product Penetration Heatmap — which segments hold which products
  4. Business Impact — estimated incremental revenue, RM time saved
  5. Drift Monitoring — KS-test alerts on uplift score distributions
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.uplift import TLearnerUplift, check_uplift_drift

MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

st.set_page_config(page_title="Next Best Product — Uplift Dashboard", layout="wide")


@st.cache_data
def load_data():
    features = pd.read_parquet(DATA_DIR / "customer_features.parquet")
    campaign = pd.read_parquet(DATA_DIR / "campaign_dataset.parquet")
    return features, campaign


@st.cache_resource
def load_model():
    return TLearnerUplift.load(MODEL_DIR)


def main():
    st.title("Next Best Product — Uplift Recommendation Dashboard")
    st.markdown("Targeting **persuadables** — customers who buy *because* of our recommendation.")

    try:
        features, campaign = load_data()
        model = load_model()
    except FileNotFoundError:
        st.error("Data or model files not found. Run the training pipeline first.")
        st.code("python scripts/train_pipeline.py")
        return

    # Score all campaign customers
    segments = model.segment_customers(campaign)
    segments["outcome"] = campaign["outcome"].values
    segments["treatment_flag"] = campaign["treatment_flag"].values
    if "product_code" in campaign.columns:
        segments["product_code"] = campaign["product_code"].values

    # Sidebar filters
    st.sidebar.header("Filters")
    selected_segments = st.sidebar.multiselect(
        "Uplift Segments",
        ["Persuadable", "Sure Thing", "Sleeping Dog", "Lost Cause"],
        default=["Persuadable", "Sure Thing", "Sleeping Dog", "Lost Cause"],
    )
    filtered = segments[segments["segment"].isin(selected_segments)]

    # ----- Panel 1: Uplift Quadrant -----
    st.header("1. Uplift Quadrant")
    col1, col2 = st.columns([2, 1])

    with col1:
        fig = px.scatter(
            filtered.sample(min(5000, len(filtered)), random_state=42),
            x="p_control",
            y="uplift_score",
            color="segment",
            color_discrete_map={
                "Persuadable": "#2ecc71",
                "Sure Thing": "#3498db",
                "Sleeping Dog": "#e74c3c",
                "Lost Cause": "#95a5a6",
            },
            opacity=0.6,
            title="Customer Uplift Quadrant",
            labels={
                "p_control": "P(buy | no recommendation)",
                "uplift_score": "Uplift Score (incremental effect)",
            },
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        seg_counts = segments["segment"].value_counts()
        st.metric("Persuadables", f"{seg_counts.get('Persuadable', 0):,}")
        st.metric("Sure Things", f"{seg_counts.get('Sure Thing', 0):,}")
        st.metric("Sleeping Dogs", f"{seg_counts.get('Sleeping Dog', 0):,}")
        st.metric("Lost Causes", f"{seg_counts.get('Lost Cause', 0):,}")

    # ----- Panel 2: Segment Summary Table -----
    st.header("2. Segment Performance")

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
            "Segment": seg_name,
            "Count": len(seg),
            "% of Total": f"{len(seg)/len(segments)*100:.1f}%",
            "Treatment Conv.": f"{treat_rate:.4f}",
            "Control Conv.": f"{ctrl_rate:.4f}",
            "Observed Uplift": f"{treat_rate - ctrl_rate:.4f}",
            "Avg Predicted Uplift": f"{seg['uplift_score'].mean():.4f}",
            "Action": {
                "Persuadable": "TARGET",
                "Sure Thing": "SKIP",
                "Sleeping Dog": "AVOID",
                "Lost Cause": "SKIP",
            }[seg_name],
        })

    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # ----- Panel 3: Product Penetration Heatmap -----
    st.header("3. Product Penetration by Segment")

    if "product_code" in segments.columns:
        pivot = segments.groupby(["segment", "product_code"])["outcome"].mean().reset_index()
        pivot_wide = pivot.pivot(index="product_code", columns="segment", values="outcome").fillna(0)

        fig_heat = px.imshow(
            pivot_wide.values,
            x=pivot_wide.columns.tolist(),
            y=pivot_wide.index.tolist(),
            color_continuous_scale="RdYlGn",
            title="Conversion Rate by Product x Segment",
            labels={"color": "Conv. Rate"},
        )
        fig_heat.update_layout(height=400)
        st.plotly_chart(fig_heat, use_container_width=True)

    # ----- Panel 4: Business Impact Estimation -----
    st.header("4. Estimated Business Impact")

    col1, col2, col3 = st.columns(3)

    # Current strategy: recommend to everyone
    total_customers = len(segments)
    persuadables = segments[segments["segment"] == "Persuadable"]
    sure_things = segments[segments["segment"] == "Sure Thing"]

    # RM time per recommendation (assume 15 min per customer)
    rm_time_per_customer = 15  # minutes
    total_rm_hours_all = total_customers * rm_time_per_customer / 60
    targeted_rm_hours = len(persuadables) * rm_time_per_customer / 60
    rm_hours_saved = total_rm_hours_all - targeted_rm_hours

    with col1:
        st.metric(
            "RM Hours Saved (vs recommend-all)",
            f"{rm_hours_saved:,.0f} hrs",
            delta=f"-{rm_hours_saved/total_rm_hours_all*100:.0f}% workload",
        )

    # Incremental conversions
    if len(persuadables) > 0:
        persuadable_uplift = persuadables["uplift_score"].mean()
        incremental_conversions = int(len(persuadables) * persuadable_uplift)
    else:
        incremental_conversions = 0

    with col2:
        st.metric(
            "Estimated Incremental Conversions",
            f"{incremental_conversions:,}",
            help="Conversions that wouldn't happen without RM outreach",
        )

    # Revenue (assume avg product value S$500/year)
    avg_product_value = 500
    incremental_revenue = incremental_conversions * avg_product_value

    with col3:
        st.metric(
            "Estimated Incremental Revenue",
            f"S${incremental_revenue:,.0f}",
            help="Based on avg product value of S$500/year",
        )

    # Efficiency comparison bar chart
    strategies = pd.DataFrame({
        "Strategy": ["Recommend All", "Target Persuadables Only"],
        "Customers Contacted": [total_customers, len(persuadables)],
        "Est. Incremental Conv.": [incremental_conversions, incremental_conversions],
        "RM Hours": [total_rm_hours_all, targeted_rm_hours],
    })
    strategies["Efficiency (Conv/Hour)"] = (
        strategies["Est. Incremental Conv."] / strategies["RM Hours"].clip(lower=1)
    )

    fig_eff = px.bar(
        strategies,
        x="Strategy",
        y="Efficiency (Conv/Hour)",
        color="Strategy",
        title="RM Efficiency: Recommend All vs Target Persuadables",
        color_discrete_sequence=["#95a5a6", "#2ecc71"],
    )
    st.plotly_chart(fig_eff, use_container_width=True)

    # ----- Panel 5: Uplift Score Distribution + Drift -----
    st.header("5. Uplift Score Distribution & Drift Monitoring")

    col1, col2 = st.columns(2)

    with col1:
        fig_dist = px.histogram(
            segments,
            x="uplift_score",
            color="segment",
            nbins=50,
            title="Uplift Score Distribution",
            color_discrete_map={
                "Persuadable": "#2ecc71",
                "Sure Thing": "#3498db",
                "Sleeping Dog": "#e74c3c",
                "Lost Cause": "#95a5a6",
            },
            barmode="overlay",
            opacity=0.7,
        )
        fig_dist.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        # Simulate drift check (compare first half vs second half as proxy)
        half = len(segments) // 2
        baseline = segments["uplift_score"].values[:half]
        current = segments["uplift_score"].values[half:]
        drift_result = check_uplift_drift(baseline, current)

        if drift_result["is_drifted"]:
            st.error(f"DRIFT DETECTED — KS statistic: {drift_result['ks_statistic']:.4f}, p-value: {drift_result['p_value']:.6f}")
            st.warning("Recommendation: Retrain the uplift model.")
        else:
            st.success(f"No drift — KS statistic: {drift_result['ks_statistic']:.4f}, p-value: {drift_result['p_value']:.6f}")
            st.info("Model is stable. No retraining needed.")

    # ----- Panel 6: Individual Customer Lookup -----
    st.header("6. Customer Lookup")
    customer_id = st.number_input("Enter Customer ID", min_value=0, step=1)

    if customer_id > 0:
        cust = segments[segments["customer_id"] == customer_id]
        if len(cust) > 0:
            row = cust.iloc[0]
            st.json({
                "customer_id": int(row["customer_id"]),
                "segment": row["segment"],
                "uplift_score": round(float(row["uplift_score"]), 4),
                "p_treatment": round(float(row["p_treatment"]), 4),
                "p_control": round(float(row["p_control"]), 4),
            })
        else:
            st.warning("Customer not found in campaign data.")


if __name__ == "__main__":
    main()
