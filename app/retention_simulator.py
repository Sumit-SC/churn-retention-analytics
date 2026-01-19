"""
Streamlit application for churn retention strategy simulation.
"""

import streamlit as st
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Churn Retention Strategy Simulator",
    page_icon="🎯",
    layout="wide"
)

MODELS_DIR = Path("models")
DB_PATH = Path("churn.duckdb")

@st.cache_data
def load_data_and_model():
    """Load data and trained model."""
    conn = duckdb.connect(str(DB_PATH))
    df = conn.execute("SELECT * FROM analytics.churn_features").df()
    conn.close()
    
    df_model = df[df["churn_label"].isin([0, 1])].copy()
    
    df_model["usage_decline_flag"] = ((df_model["usage_trend_30d"] < 0).fillna(False)).astype(int)
    df_model["high_support_flag"] = ((df_model["total_tickets"] >= 3).fillna(False)).astype(int)
    df_model["payment_issue_flag"] = ((df_model["total_payment_issues"] >= 1).fillna(False)).astype(int)
    
    numeric_features_rf = [
        "active_days",
        "avg_sessions",
        "avg_usage_minutes",
        "usage_trend_30d",
        "total_payment_issues",
        "failed_payments_30d",
        "total_tickets",
        "high_priority_tickets",
        "usage_decline_flag",
        "high_support_flag",
        "payment_issue_flag",
    ]
    
    categorical_features = ["plan", "region"]
    
    winsorize_features = ["total_payment_issues", "total_tickets"]
    for feat in winsorize_features:
        p1 = df_model[feat].quantile(0.01)
        p99 = df_model[feat].quantile(0.99)
        df_model[feat] = df_model[feat].clip(lower=p1, upper=p99)
    
    X_rf = df_model[numeric_features_rf + categorical_features].copy()
    
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    
    preprocessor_rf = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features_rf),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
    )
    
    preprocessor_rf.fit(X_rf)
    model = joblib.load(MODELS_DIR / "rf_model.joblib")
    
    X_processed = preprocessor_rf.transform(X_rf)
    churn_risk_scores = model.predict_proba(X_processed)[:, 1]
    
    df_model["churn_risk_score"] = churn_risk_scores
    
    return df_model, model, preprocessor_rf

df_model, model, preprocessor = load_data_and_model()

st.title("🎯 Churn Retention Strategy Simulator")

st.markdown("---")

st.header("1. Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", f"{len(df_model):,}")
col2.metric("Average Churn Risk", f"{df_model['churn_risk_score'].mean():.2%}")
col3.metric("Actual Churn Rate", f"{df_model['churn_label'].mean():.2%}")

st.markdown("---")

st.header("2. Risk Scoring")

col1, col2 = st.columns(2)

with col1:
    medium_risk_pct = (df_model["churn_risk_score"] > 0.3).mean() * 100
    high_risk_pct = (df_model["churn_risk_score"] > 0.6).mean() * 100
    
    st.subheader("Risk Distribution")
    st.metric("Medium Risk (>0.3)", f"{medium_risk_pct:.1f}%")
    st.metric("High Risk (>0.6)", f"{high_risk_pct:.1f}%")
    
    mean_risk = df_model["churn_risk_score"].mean()
    
    fig_hist = px.histogram(
        df_model,
        x="churn_risk_score",
        nbins=50,
        title="Churn Risk Score Distribution",
        labels={"churn_risk_score": "Churn Risk Score", "count": "Number of Customers"}
    )
    fig_hist.add_vline(
        x=mean_risk,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Mean: {mean_risk:.2%}",
        annotation_position="top"
    )
    fig_hist.update_layout(
        height=400,
        annotations=[
            dict(
                x=mean_risk,
                y=0.95,
                xref="x",
                yref="paper",
                text=f"Mean Risk: {mean_risk:.2%}",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-40
            )
        ]
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.caption("💡 **Tip:** Adjust the risk threshold in the sidebar to change which customers are eligible for targeting.")

with col2:
    st.subheader("Top 10 Highest-Risk Customers")
    top_risk = df_model.nlargest(10, "churn_risk_score")[
        ["customer_id", "plan", "region", "churn_risk_score", "total_tickets", "total_payment_issues"]
    ].copy()
    top_risk["churn_risk_score"] = top_risk["churn_risk_score"].apply(lambda x: f"{x:.2%}")
    st.dataframe(top_risk, use_container_width=True, hide_index=True)

st.markdown("---")

st.header("3. Retention Strategy Simulator")

st.sidebar.header("🎛️ Simulation Controls")

st.sidebar.subheader("Targeting")
risk_threshold = st.sidebar.slider(
    "Risk Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.6,
    step=0.05,
    help="Target customers above this risk score"
)

max_target_pct = st.sidebar.slider(
    "Max Customers to Target (%)",
    min_value=1,
    max_value=50,
    value=10,
    step=1,
    help="Maximum percentage of customer base to target"
)

st.sidebar.subheader("Retention Levers")

apply_discount = st.sidebar.checkbox("Apply Discount", value=False)
if apply_discount:
    discount_pct = st.sidebar.slider(
        "Discount (%)",
        min_value=5,
        max_value=30,
        value=15,
        step=5
    )
else:
    discount_pct = 0

apply_priority_support = st.sidebar.checkbox("Priority Support", value=False)
apply_feature_unlock = st.sidebar.checkbox("Feature Unlock", value=False)

targeted_customers = df_model[df_model["churn_risk_score"] >= risk_threshold].copy()
max_target_count = int(len(df_model) * max_target_pct / 100)
targeted_customers = targeted_customers.nlargest(max_target_count, "churn_risk_score")

st.subheader("Targeted Customers")
st.write(f"**{len(targeted_customers):,} customers** selected for retention intervention")

if len(targeted_customers) > 0:
    st.markdown("---")
    st.header("📌 Key Insights for Selected Cohort")
    
    usage_decline_pct = (targeted_customers["usage_decline_flag"] == 1).mean() * 100
    payment_issue_pct = (targeted_customers["payment_issue_flag"] == 1).mean() * 100
    high_support_pct = (targeted_customers["high_support_flag"] == 1).mean() * 100
    avg_risk_cohort = targeted_customers["churn_risk_score"].mean()
    avg_risk_overall = df_model["churn_risk_score"].mean()
    
    plan_churn_contribution = targeted_customers.groupby("plan").agg({
        "churn_risk_score": "sum"
    }).sort_values("churn_risk_score", ascending=False)
    total_expected_churn = targeted_customers["churn_risk_score"].sum()
    plan_churn_contribution["contribution_pct"] = (plan_churn_contribution["churn_risk_score"] / total_expected_churn * 100).round(1)
    top_plan = plan_churn_contribution.index[0]
    top_plan_pct = plan_churn_contribution.loc[top_plan, "contribution_pct"]
    
    dominant_driver = max(
        [("Usage Decline", usage_decline_pct), ("Payment Issues", payment_issue_pct), ("Support Load", high_support_pct)],
        key=lambda x: x[1]
    )
    
    insights = [
        f"**{dominant_driver[0]}** is the dominant churn signal: {dominant_driver[1]:.1f}% of selected customers are affected",
        f"**Plan '{top_plan}'** contributes {top_plan_pct:.1f}% of expected churn in this cohort",
        f"Average churn risk in cohort: **{avg_risk_cohort:.1%}** (vs {avg_risk_overall:.1%} overall)",
        f"**{usage_decline_pct:.1f}%** show usage decline, **{payment_issue_pct:.1f}%** have payment issues, **{high_support_pct:.1f}%** have high support load"
    ]
    
    if avg_risk_cohort > avg_risk_overall * 1.5:
        insights.append(f"Cohort risk is **{((avg_risk_cohort / avg_risk_overall - 1) * 100):.0f}% higher** than overall average — high-priority intervention segment")
    
    for insight in insights:
        st.markdown(f"• {insight}")
    
    st.markdown("")
    
    col1, col2 = st.columns(2)
    
    with col1:
        driver_data = pd.DataFrame({
            "Risk Driver": ["Usage Decline", "Payment Issues", "Support Load"],
            "% Affected": [usage_decline_pct, payment_issue_pct, high_support_pct]
        })
        
        fig_drivers = px.bar(
            driver_data,
            x="Risk Driver",
            y="% Affected",
            title="Primary Churn Signals in Selected Cohort",
            labels={"% Affected": "% of Customers Affected"},
            color="% Affected",
            color_continuous_scale="Reds"
        )
        fig_drivers.update_traces(
            hovertemplate="<b>%{x}</b><br>%{y:.1f}% of customers affected<extra></extra>",
            texttemplate="%{y:.1f}%",
            textposition="outside"
        )
        fig_drivers.update_layout(
            height=400,
            showlegend=False,
            yaxis_range=[0, max(driver_data["% Affected"]) * 1.2]
        )
        st.plotly_chart(fig_drivers, use_container_width=True)
    
    with col2:
        st.subheader("Plan Contribution to Expected Churn")
        plan_contrib_df = plan_churn_contribution.reset_index()
        plan_contrib_df["churn_risk_score"] = plan_contrib_df["churn_risk_score"].round(1)
        plan_contrib_df = plan_contrib_df.rename(columns={
            "plan": "Plan",
            "churn_risk_score": "Expected Churn",
            "contribution_pct": "Contribution %"
        })
        plan_contrib_df["Contribution %"] = plan_contrib_df["Contribution %"].apply(lambda x: f"{x:.1f}%")
        st.dataframe(plan_contrib_df[["Plan", "Expected Churn", "Contribution %"]], use_container_width=True, hide_index=True)

PLAN_REVENUE = {
    "Free": 0,
    "Basic": 20,
    "Pro": 50,
    "Enterprise": 200
}

PRIORITY_SUPPORT_COST = 5
FEATURE_UNLOCK_COST = 3

def simulate_retention(df_target, discount_pct, priority_support, feature_unlock):
    """Simulate retention impact using heuristic uplift assumptions."""
    df_sim = df_target.copy()
    
    base_churn_prob = df_sim["churn_risk_score"].values.copy()
    new_churn_prob = base_churn_prob.copy()
    
    if discount_pct > 0:
        reduction = discount_pct * 0.005
        new_churn_prob = np.maximum(new_churn_prob - reduction, 0.05)
    
    if priority_support:
        new_churn_prob = np.maximum(new_churn_prob - 0.10, 0.05)
    
    if feature_unlock:
        new_churn_prob = np.maximum(new_churn_prob - 0.07, 0.05)
    
    df_sim["churn_prob_before"] = base_churn_prob
    df_sim["churn_prob_after"] = new_churn_prob
    
    expected_churn_before = base_churn_prob.sum()
    expected_churn_after = new_churn_prob.sum()
    customers_saved = expected_churn_before - expected_churn_after
    
    df_sim["monthly_revenue"] = df_sim["plan"].map(PLAN_REVENUE)
    total_revenue = df_sim["monthly_revenue"].sum()
    
    discount_cost = (total_revenue * discount_pct / 100) if discount_pct > 0 else 0
    priority_cost = len(df_sim) * PRIORITY_SUPPORT_COST if priority_support else 0
    feature_cost = len(df_sim) * FEATURE_UNLOCK_COST if feature_unlock else 0
    total_cost = discount_cost + priority_cost + feature_cost
    
    retained_revenue = customers_saved * df_sim["monthly_revenue"].mean()
    net_roi = retained_revenue - total_cost
    
    return {
        "expected_churn_before": expected_churn_before,
        "expected_churn_after": expected_churn_after,
        "customers_saved": customers_saved,
        "total_cost": total_cost,
        "retained_revenue": retained_revenue,
        "net_roi": net_roi,
        "churn_rate_before": expected_churn_before / len(df_sim),
        "churn_rate_after": expected_churn_after / len(df_sim),
    }

if len(targeted_customers) > 0:
    results = simulate_retention(
        targeted_customers,
        discount_pct,
        apply_priority_support,
        apply_feature_unlock
    )
    
    st.markdown("---")
    st.header("4. Business Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Expected Churn (Before)", f"{results['expected_churn_before']:.0f}")
    col2.metric("Expected Churn (After)", f"{results['expected_churn_after']:.0f}")
    col3.metric("Customers Saved", f"{results['customers_saved']:.0f}")
    col4.metric("Churn Rate Reduction", f"{(results['churn_rate_before'] - results['churn_rate_after']):.2%}")
    
    st.markdown("")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Retention Cost", f"${results['total_cost']:,.0f}")
    col2.metric("Retained Revenue (Monthly)", f"${results['retained_revenue']:,.0f}")
    col3.metric("Net ROI", f"${results['net_roi']:,.0f}", delta=f"{(results['net_roi']/results['total_cost']*100):.1f}%" if results['total_cost'] > 0 else None)
    
    st.markdown("")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_churn = go.Figure()
        fig_churn.add_trace(go.Bar(
            x=["Before", "After"],
            y=[results["churn_rate_before"], results["churn_rate_after"]],
            marker_color=["#ef4444", "#10b981"],
            text=[f"{results['churn_rate_before']:.1%}", f"{results['churn_rate_after']:.1%}"],
            textposition="auto",
            hovertemplate="<b>%{x}</b><br>Churn Rate: %{y:.2%}<extra></extra>"
        ))
        fig_churn.update_layout(
            title="Churn Rate: Before vs After",
            yaxis_title="Churn Rate",
            height=400
        )
        st.plotly_chart(fig_churn, use_container_width=True)
    
    with col2:
        discount_cost = (targeted_customers["plan"].map(PLAN_REVENUE).sum() * discount_pct / 100) if discount_pct > 0 else 0
        priority_cost = len(targeted_customers) * PRIORITY_SUPPORT_COST if apply_priority_support else 0
        feature_cost = len(targeted_customers) * FEATURE_UNLOCK_COST if apply_feature_unlock else 0
        
        cost_categories = []
        cost_amounts = []
        revenue_categories = []
        revenue_amounts = []
        
        if discount_cost > 0:
            cost_categories.append("Discount")
            cost_amounts.append(discount_cost)
        if priority_cost > 0:
            cost_categories.append("Priority Support")
            cost_amounts.append(priority_cost)
        if feature_cost > 0:
            cost_categories.append("Feature Unlock")
            cost_amounts.append(feature_cost)
        
        if results["retained_revenue"] > 0:
            revenue_categories.append("Retained Revenue")
            revenue_amounts.append(results["retained_revenue"])
        
        fig_roi = go.Figure()
        
        if len(cost_categories) > 0:
            fig_roi.add_trace(go.Bar(
                name="Costs",
                x=cost_categories,
                y=cost_amounts,
                marker_color="#f59e0b",
                hovertemplate="<b>%{x}</b><br>Cost: $%{y:,.0f}<extra></extra>",
                text=[f"${x:,.0f}" for x in cost_amounts],
                textposition="outside"
            ))
        
        if len(revenue_categories) > 0:
            fig_roi.add_trace(go.Bar(
                name="Revenue",
                x=revenue_categories,
                y=revenue_amounts,
                marker_color="#3b82f6",
                hovertemplate="<b>%{x}</b><br>Revenue: $%{y:,.0f}<extra></extra>",
                text=[f"${x:,.0f}" for x in revenue_amounts],
                textposition="outside"
            ))
        fig_roi.update_layout(
            title="Cost Breakdown vs Retained Revenue",
            yaxis_title="Amount ($)",
            barmode="group",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_roi, use_container_width=True)
    
    st.markdown("---")
    st.subheader("💡 Understanding the Results")
    
    st.markdown(f"""
    **Why customers saved changes:** The number of customers saved depends on the size and risk profile of your selected cohort. 
    As you adjust the risk threshold or max target percentage, you're changing which customers are included in the intervention. 
    Higher-risk cohorts have more potential for improvement, but may also require higher intervention costs.
    
    **Why ROI can vary:** ROI depends on the balance between intervention costs and retained revenue. 
    - **Positive ROI** occurs when retained revenue exceeds total costs — typically when targeting high-value customers (Pro/Enterprise) with effective interventions
    - **Negative ROI** can occur when costs exceed benefits — often when targeting low-value customers (Free/Basic) or applying expensive interventions to large cohorts
    
    **This is a scenario comparison tool:** The simulator helps you compare different retention strategies and understand trade-offs. 
    The uplift assumptions are illustrative heuristics for relative comparison, not exact predictions. 
    Use this tool to explore "what-if" scenarios and identify promising intervention strategies for further testing.
    """)
    
    st.markdown("---")
    st.subheader("📋 Assumptions & Disclaimers")
    
    st.info("""
    **Model & Uplift Assumptions:**
    - Churn risk scores are predictions, not certainties
    - Uplift assumptions are illustrative heuristics:
      - Discount: Reduces churn risk by (discount % × 0.5%)
      - Priority Support: Reduces churn risk by 10%
      - Feature Unlock: Reduces churn risk by 7%
    - Revenue estimates based on average monthly revenue by plan tier
    - Costs are fixed estimates per customer
    
    **Use Cases:**
    - Scenario comparison and strategy planning
    - Relative impact assessment across interventions
    - Not intended for exact financial forecasting
    """)
else:
    st.warning("No customers match the current targeting criteria. Adjust risk threshold or max target percentage.")
