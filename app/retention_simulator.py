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
    page_title="Churn & Retention Analytics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

hide_streamlit_style = """
    <style>
    /* Hide footer */
    footer {visibility: hidden;}

    /* Hide deploy button */
    .stDeployButton {display: none;}

    /* Hide top decoration */
    #stDecoration {display: none;}

    /* Improve main content padding slightly */
    .main .block-container {
        padding-top: 1.5rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }

    /* Make expanders scrollable */
    .streamlit-expanderContent {
        max-height: 65vh;
        overflow-y: auto;
    }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Get project root (parent of app folder)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DB_PATH = PROJECT_ROOT / "churn.duckdb"

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
    
    if "signup_date" in df_model.columns:
        df_model["signup_date"] = pd.to_datetime(df_model["signup_date"])
    if "last_active_date" in df_model.columns:
        df_model["last_active_date"] = pd.to_datetime(df_model["last_active_date"])
    
    return df_model, model, preprocessor_rf

df_model, model, preprocessor = load_data_and_model()

PLAN_REVENUE = {
    "Free": 0,
    "Basic": 20,
    "Pro": 50,
    "Enterprise": 200
}

PRIORITY_SUPPORT_COST = 5
FEATURE_UNLOCK_COST = 3

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
    st.plotly_chart(fig_hist, width='stretch')
    
    st.caption("💡 **Tip:** Adjust the risk threshold in the sidebar to change which customers are eligible for targeting.")

with col2:
    st.subheader("Top 10 Highest-Risk Customers")
    top_risk = df_model.nlargest(10, "churn_risk_score")[
        ["customer_id", "plan", "region", "churn_risk_score", "total_tickets", "total_payment_issues"]
    ].copy()
    top_risk["churn_risk_score"] = top_risk["churn_risk_score"].apply(lambda x: f"{x:.2%}")
    st.dataframe(top_risk, width='stretch', hide_index=True)

st.markdown("---")

st.header("3. Retention Strategy Simulator")

with st.sidebar.expander("🎛️ Simulation Controls", expanded=True):
    st.caption("Configure filters, targeting, and retention strategies")
    
    # Filter Summary at top - will show current state
    summary_placeholder = st.empty()
    
    # Date Filters Section  
    st.markdown("**📅 Date Filters**")
    use_signup_filter = st.checkbox("Filter by Signup Date", value=False)
    if use_signup_filter and "signup_date" in df_model.columns:
        signup_min = df_model["signup_date"].min().date()
        signup_max = df_model["signup_date"].max().date()
        signup_date_range = st.date_input(
            "Signup Date Range",
            value=(signup_min, signup_max),
            min_value=signup_min,
            max_value=signup_max,
            help="Select customers who signed up within this date range"
        )
    else:
        signup_date_range = None
    
    use_last_active_filter = st.checkbox("Filter by Last Active Date", value=False)
    if use_last_active_filter and "last_active_date" in df_model.columns:
        last_active_valid = df_model[df_model["last_active_date"].notna()]
        if len(last_active_valid) > 0:
            last_active_min = last_active_valid["last_active_date"].min().date()
            last_active_max = last_active_valid["last_active_date"].max().date()
            last_active_date_range = st.date_input(
                "Last Active Date Range",
                value=(last_active_min, last_active_max),
                min_value=last_active_min,
                max_value=last_active_max,
                help="Select customers whose last activity was within this date range"
            )
        else:
            last_active_date_range = None
            st.warning("No last_active_date data available")
    else:
        last_active_date_range = None
    
    st.markdown("---")
    
    # Customer Segmentation Section
    st.markdown("**🎯 Customer Segmentation**")
    filter_plan = st.multiselect(
        "Filter by Plan",
        options=sorted(df_model["plan"].unique()),
        default=[],
        help="Select specific plan tiers to include (leave empty for all plans)"
    )
    
    filter_region = st.multiselect(
        "Filter by Region",
        options=sorted(df_model["region"].unique()),
        default=[],
        help="Select specific regions to include (leave empty for all regions)"
    )
    
    st.markdown("---")
    
    # Apply filters first to get df_filtered
    df_filtered = df_model.copy()
    
    if use_signup_filter and signup_date_range is not None:
        if isinstance(signup_date_range, tuple) and len(signup_date_range) == 2:
            df_filtered = df_filtered[
                (df_filtered["signup_date"].dt.date >= signup_date_range[0]) &
                (df_filtered["signup_date"].dt.date <= signup_date_range[1])
            ]
    
    if use_last_active_filter and last_active_date_range is not None:
        if isinstance(last_active_date_range, tuple) and len(last_active_date_range) == 2:
            df_filtered = df_filtered[
                (df_filtered["last_active_date"].notna()) &
                (df_filtered["last_active_date"].dt.date >= last_active_date_range[0]) &
                (df_filtered["last_active_date"].dt.date <= last_active_date_range[1])
            ]
    
    if filter_plan:
        df_filtered = df_filtered[df_filtered["plan"].isin(filter_plan)]
    
    if filter_region:
        df_filtered = df_filtered[df_filtered["region"].isin(filter_region)]
    
    # Risk Targeting Section
    st.markdown("**🎯 Risk Targeting**")
    if len(df_filtered) == 0:
        st.error("⚠️ No customers match current filters. Adjust filters to see results.")
        risk_threshold = 0.6
        max_target_pct = 10
    else:
        risk_min = float(df_filtered["churn_risk_score"].min())
        risk_max = float(df_filtered["churn_risk_score"].max())
        risk_default = min(0.6, risk_max)
        
        if risk_max > risk_min:
            # Use adaptive step size: smaller step for smaller ranges
            range_size = risk_max - risk_min
            if range_size < 0.01:
                step_size = 0.001
            elif range_size < 0.05:
                step_size = 0.01
            elif range_size < 0.2:
                step_size = 0.02
            else:
                step_size = 0.05
            
            risk_threshold = st.slider(
                "Risk Threshold",
                min_value=risk_min,
                max_value=risk_max,
                value=risk_default,
                step=step_size,
                help=f"Only customers with churn risk score ABOVE this threshold will be considered. Valid range: {risk_min:.2%} to {risk_max:.2%}"
            )
            st.caption(f"📊 Range: {risk_min:.2%} - {risk_max:.2%} | Current: **{risk_threshold:.2%}**")
        else:
            risk_threshold = risk_min
            st.info(f"All filtered customers have risk score: {risk_min:.2%}")
        
        eligible_above_threshold = len(df_filtered[df_filtered["churn_risk_score"] >= risk_threshold])
        max_target_pct_limit = min(50, int((eligible_above_threshold / len(df_filtered)) * 100) if len(df_filtered) > 0 else 0)
        
        if max_target_pct_limit > 0:
            max_target_pct = st.slider(
                "Max Customers to Target (%)",
                min_value=1,
                max_value=max_target_pct_limit,
                value=min(10, max_target_pct_limit),
                step=1,
                help=f"Maximum percentage of eligible customers ({len(df_filtered):,}) to target. Valid range: 1% to {max_target_pct_limit}%"
            )
            st.caption(f"📊 Max: {max_target_pct_limit}% ({eligible_above_threshold:,} eligible above threshold)")
        else:
            max_target_pct = 0
            st.warning(f"⚠️ No customers above risk threshold {risk_threshold:.2%}. Lower the threshold to see results.")
    
    st.markdown("---")
    
    # Retention Levers Section
    st.markdown("**💰 Retention Levers**")
    apply_discount = st.checkbox("Apply Discount", value=False, help="Offer percentage discount on monthly subscription")
    if apply_discount:
        discount_pct = st.slider(
            "Discount (%)",
            min_value=5,
            max_value=30,
            value=15,
            step=5,
            help="Discount percentage applied to monthly revenue. Higher discounts = higher cost but potentially better retention."
        )
        st.caption(f"💰 Cost: {discount_pct}% of monthly revenue per customer")
    else:
        discount_pct = 0
    
    apply_priority_support = st.checkbox(
        "Priority Support", 
        value=False,
        help="Provide priority customer support (faster response times, dedicated support)"
    )
    if apply_priority_support:
        st.caption(f"💰 Cost: ${PRIORITY_SUPPORT_COST} per customer per month")
    
    apply_feature_unlock = st.checkbox(
        "Feature Unlock", 
        value=False,
        help="Unlock premium features for free-tier customers or add-ons for paid customers"
    )
    if apply_feature_unlock:
        st.caption(f"💰 Cost: ${FEATURE_UNLOCK_COST} per customer per month")
    
    # Update Filter Summary (calculated after all filters)
    filter_summary = []
    if use_signup_filter and signup_date_range is not None and isinstance(signup_date_range, tuple) and len(signup_date_range) == 2:
        filter_summary.append(f"📅 Signup: {signup_date_range[0]} to {signup_date_range[1]}")
    if use_last_active_filter and last_active_date_range is not None and isinstance(last_active_date_range, tuple) and len(last_active_date_range) == 2:
        filter_summary.append(f"📅 Last Active: {last_active_date_range[0]} to {last_active_date_range[1]}")
    if filter_plan:
        filter_summary.append(f"📦 Plans: {', '.join(filter_plan)}")
    if filter_region:
        filter_summary.append(f"🌍 Regions: {', '.join(filter_region)}")
    
    if len(df_filtered) > 0:
        targeted_customers_temp = df_filtered[df_filtered["churn_risk_score"] >= risk_threshold].copy()
        max_target_count_temp = int(len(df_filtered) * max_target_pct / 100)
        targeted_customers_temp = targeted_customers_temp.nlargest(max_target_count_temp, "churn_risk_score")
        filter_summary.append(f"✅ Eligible: {len(df_filtered):,} customers")
        filter_summary.append(f"🎯 Risk Threshold: ≥{risk_threshold:.2%}")
        filter_summary.append(f"📊 Max Target: {max_target_pct}% ({max_target_count_temp:,} customers)")
        filter_summary.append(f"✅ Selected: {len(targeted_customers_temp):,} customers")
    else:
        filter_summary.append("⚠️ **No customers match current filters**")
    
    # Update the placeholder at top with summary
    with summary_placeholder.container():
        st.markdown("**ℹ️ Filter Summary**")
        for summary in filter_summary:
            st.caption(summary)

# Calculate final targeted customers outside expander
if len(df_filtered) > 0:
    targeted_customers = df_filtered[df_filtered["churn_risk_score"] >= risk_threshold].copy()
    max_target_count = int(len(df_filtered) * max_target_pct / 100)
    targeted_customers = targeted_customers.nlargest(max_target_count, "churn_risk_score")
else:
    targeted_customers = pd.DataFrame()
    max_target_count = 0

st.sidebar.markdown("---")
with st.sidebar.expander("📋 About Project", expanded=False):
    st.markdown("<h3 style='text-align: center;'>Churn & Retention Analytics</h3>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    **Project Highlights:**
    
    • End-to-end churn prediction system
    • SQL-based feature engineering pipeline
    • ML models (Logistic Regression + Random Forest)
    • SHAP-based explainability
    • Lifecycle-based customer segmentation
    • Interactive retention strategy simulator
    
    **Technologies:**
    • Python, SQL, DuckDB
    • scikit-learn, SHAP
    • Streamlit, Plotly, Seaborn
    • Pandas, NumPy
    
    **Key Features:**
    • 100K+ customer synthetic dataset
    • Production-ready SQL pipelines
    • Risk scoring & targeting
    • ROI-optimized retention strategies
    """)

st.sidebar.markdown("---")
with st.sidebar.expander("🔗 Connect with Me", expanded=False):
    icon_col1, icon_col2, icon_col3, icon_col4 = st.columns(4)

with icon_col1:
    st.markdown("""
    <div style="text-align: center;">
        <a href="https://github.com/" target="_blank">
            <img src="https://mitsus-.life-is-pa.in/7s66cDoBZ.png" 
                 style="width: 50px; height: 50px; cursor: pointer; transition: transform 0.2s;" 
                 onmouseover="this.style.transform='scale(1.1)'" 
                 onmouseout="this.style.transform='scale(1)'"
                 alt="Github">
        </a>
        <p style="margin-top: 5px; font-size: 0.75em;">Github</p>
    </div>
    """, unsafe_allow_html=True)

with icon_col2:
    st.markdown("""
    <div style="text-align: center;">
        <a href="https://www.kaggle.com/" target="_blank">
            <img src="https://mitsus-.life-is-pa.in/7s65GCpDu.png" 
                 style="width: 50px; height: 50px; cursor: pointer; transition: transform 0.2s;" 
                 onmouseover="this.style.transform='scale(1.1)'" 
                 onmouseout="this.style.transform='scale(1)'"
                 alt="Kaggle">
        </a>
        <p style="margin-top: 5px; font-size: 0.75em;">Kaggle</p>
    </div>
    """, unsafe_allow_html=True)

with icon_col3:
    st.markdown("""
    <div style="text-align: center;">
        <a href="https://www.linkedin.com/in/" target="_blank">
            <img src="https://mitsus-.life-is-pa.in/7s65TFl9W.png" 
                 style="width: 50px; height: 50px; cursor: pointer; transition: transform 0.2s;" 
                 onmouseover="this.style.transform='scale(1.1)'" 
                 onmouseout="this.style.transform='scale(1)'"
                 alt="LinkedIn">
        </a>
        <p style="margin-top: 5px; font-size: 0.75em;">LinkedIn</p>
    </div>
    """, unsafe_allow_html=True)

with icon_col4:
    st.markdown("""
    <div style="text-align: center;">
        <a href="mailto:your.email@example.com">
            <svg width="50" height="50" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" 
                 style="cursor: pointer; transition: transform 0.2s;" 
                 onmouseover="this.style.transform='scale(1.1)'" 
                 onmouseout="this.style.transform='scale(1)'">
                <path d="M20 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 4l-8 5-8-5V6l8 5 8-5v2z" fill="#1f77b4"/>
            </svg>
        </a>
        <p style="margin-top: 5px; font-size: 0.75em;">Email</p>
    </div>
    """, unsafe_allow_html=True)

st.subheader("Targeted Customers")
if len(targeted_customers) > 0:
    st.write(f"**{len(targeted_customers):,} customers** selected for retention intervention")
    st.caption(f"Selected from {len(df_filtered):,} eligible customers (after applying filters) out of {len(df_model):,} total customers")
else:
    st.warning("⚠️ **No customers selected.** Adjust filters or risk threshold in the sidebar to select customers for retention intervention.")
    st.caption(f"Current filters result in {len(df_filtered):,} eligible customers. Try lowering the risk threshold or adjusting other filters.")

if len(targeted_customers) > 0:
    st.markdown("---")
    st.header("📌 Key Insights for Selected Cohort")
    
    usage_decline_pct = (targeted_customers["usage_decline_flag"] == 1).mean() * 100
    payment_issue_pct = (targeted_customers["payment_issue_flag"] == 1).mean() * 100
    high_support_pct = (targeted_customers["high_support_flag"] == 1).mean() * 100
    
    negative_trend_count = (targeted_customers["usage_trend_30d"] < 0).sum()
    null_trend_count = targeted_customers["usage_trend_30d"].isna().sum()
    avg_risk_cohort = targeted_customers["churn_risk_score"].mean()
    avg_risk_overall = df_model["churn_risk_score"].mean()
    
    usage_trend_negative = (targeted_customers["usage_trend_30d"] < 0).sum()
    usage_trend_null = targeted_customers["usage_trend_30d"].isna().sum()
    usage_trend_positive = ((targeted_customers["usage_trend_30d"] >= 0) & (targeted_customers["usage_trend_30d"].notna())).sum()
    
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
    ]
    
    if negative_trend_count > 0:
        insights.append(f"**{negative_trend_count:,} customers ({negative_trend_count/len(targeted_customers)*100:.1f}%)** show negative usage trends (declining engagement)")
    if null_trend_count > 0:
        insights.append(f"**{null_trend_count:,} customers ({null_trend_count/len(targeted_customers)*100:.1f}%)** have insufficient usage history for trend calculation")
    
    if usage_decline_pct < 10 and negative_trend_count > 0:
        insights.append(f"⚠️ **Low usage decline signal** ({usage_decline_pct:.1f}%) — high-risk customers may be driven by other factors (plan tier, payment issues, support load) rather than usage decline")
    elif usage_decline_pct < 10:
        insights.append(f"ℹ️ **Usage decline not a primary driver** — High-risk customers in this cohort are identified primarily by plan tier, payment issues, or support load rather than usage trends")
    
    insights.append(f"**{payment_issue_pct:.1f}%** have payment issues, **{high_support_pct:.1f}%** have high support load (3+ tickets)")
    
    if avg_risk_cohort > avg_risk_overall * 1.5:
        insights.append(f"Cohort risk is **{((avg_risk_cohort / avg_risk_overall - 1) * 100):.0f}% higher** than overall average — high-priority intervention segment")
    
    for insight in insights:
        st.markdown(f"• {insight}")
    
    st.markdown("")
    
    col1, col2 = st.columns(2)
    
    with col1:
        driver_data = pd.DataFrame({
            "Risk Driver": ["Usage Decline", "Payment Issues", "Support Load"],
            "% Affected": [usage_decline_pct, payment_issue_pct, high_support_pct],
            "Count": [
                (targeted_customers["usage_decline_flag"] == 1).sum(),
                (targeted_customers["payment_issue_flag"] == 1).sum(),
                (targeted_customers["high_support_flag"] == 1).sum()
            ]
        })
        
        fig_drivers = px.bar(
            driver_data,
            x="Risk Driver",
            y="% Affected",
            title="Primary Churn Signals in Selected Cohort",
            labels={"% Affected": "% of Customers Affected"},
            color="% Affected",
            color_continuous_scale="Reds",
            text="Count"
        )
        fig_drivers.update_traces(
            hovertemplate="<b>%{x}</b><br>%{y:.1f}% of customers affected<br>Count: %{text}<extra></extra>",
            texttemplate="%{y:.1f}%<br>(%{text})",
            textposition="outside"
        )
        fig_drivers.update_layout(
            height=400,
            showlegend=False,
            yaxis_range=[0, max(driver_data["% Affected"]) * 1.2 if max(driver_data["% Affected"]) > 0 else 100]
        )
        st.plotly_chart(fig_drivers, width='stretch')
        
        if usage_decline_pct < 5:
            st.info("💡 **Note:** Usage decline flag is low because high-risk customers in this cohort are primarily identified by other factors (plan tier, payment issues, support load). The Random Forest model uses multiple signals, not just usage trends.")
    
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
        st.dataframe(plan_contrib_df[["Plan", "Expected Churn", "Contribution %"]], width='stretch', hide_index=True)
    
    st.markdown("---")
    st.subheader("📊 Deep Dive: Cohort Analysis")
    
    targeted_customers["monthly_revenue"] = targeted_customers["plan"].map(PLAN_REVENUE)
    
    col1, col2 = st.columns(2)
    
    with col1:
        plan_risk_avg = targeted_customers.groupby("plan")["churn_risk_score"].mean().sort_values(ascending=False)
        fig_plan_risk = px.bar(
            x=plan_risk_avg.index,
            y=plan_risk_avg.values,
            title="Average Churn Risk by Plan (Selected Cohort)",
            labels={"x": "Plan", "y": "Average Churn Risk"},
            color=plan_risk_avg.values,
            color_continuous_scale="Reds"
        )
        fig_plan_risk.update_traces(
            texttemplate="%{y:.1%}",
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Avg Risk: %{y:.2%}<extra></extra>"
        )
        fig_plan_risk.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_plan_risk, width='stretch')
    
    with col2:
        usage_trend_data = targeted_customers[targeted_customers["usage_trend_30d"].notna()].copy()
        if len(usage_trend_data) > 0:
            fig_trend = px.histogram(
                usage_trend_data,
                x="usage_trend_30d",
                nbins=30,
                title="Usage Trend Distribution (Selected Cohort)",
                labels={"usage_trend_30d": "Usage Trend (min/day)", "count": "Number of Customers"},
                color_discrete_sequence=["#ef4444"]
            )
            fig_trend.add_vline(
                x=0,
                line_dash="dash",
                line_color="blue",
                annotation_text="Decline Threshold",
                annotation_position="top"
            )
            fig_trend.update_layout(height=400)
            st.plotly_chart(fig_trend, width='stretch')
        else:
            st.info("No usage trend data available for selected cohort")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_risk_value = px.scatter(
            targeted_customers,
            x="churn_risk_score",
            y="monthly_revenue",
            color="plan",
            size="total_tickets",
            title="Risk vs Value: Churn Risk vs Monthly Revenue",
            labels={"churn_risk_score": "Churn Risk Score", "monthly_revenue": "Monthly Revenue ($)"},
            hover_data=["customer_id", "total_payment_issues"],
            category_orders={"plan": ["Free", "Basic", "Pro", "Enterprise"]}
        )
        fig_risk_value.update_layout(height=400)
        st.plotly_chart(fig_risk_value, width='stretch')
    
    with col2:
        region_risk = targeted_customers.groupby("region")["churn_risk_score"].mean().sort_values(ascending=False)
        fig_region = px.bar(
            x=region_risk.index,
            y=region_risk.values,
            title="Average Churn Risk by Region (Selected Cohort)",
            labels={"x": "Region", "y": "Average Churn Risk"},
            color=region_risk.values,
            color_continuous_scale="Oranges"
        )
        fig_region.update_traces(
            texttemplate="%{y:.1%}",
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Avg Risk: %{y:.2%}<extra></extra>"
        )
        fig_region.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_region, width='stretch')
    
    st.markdown("---")
    st.subheader("🔍 Feature Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Avg Active Days",
            f"{targeted_customers['active_days'].mean():.1f}",
            delta=f"{(targeted_customers['active_days'].mean() - df_model['active_days'].mean()):.1f} vs overall"
        )
        st.metric(
            "Avg Sessions/Day",
            f"{targeted_customers['avg_sessions'].mean():.2f}",
            delta=f"{(targeted_customers['avg_sessions'].mean() - df_model['avg_sessions'].mean()):.2f} vs overall"
        )
    
    with col2:
        st.metric(
            "Avg Usage Minutes/Day",
            f"{targeted_customers['avg_usage_minutes'].mean():.1f}",
            delta=f"{(targeted_customers['avg_usage_minutes'].mean() - df_model['avg_usage_minutes'].mean()):.1f} vs overall"
        )
        st.metric(
            "Avg Payment Issues",
            f"{targeted_customers['total_payment_issues'].mean():.1f}",
            delta=f"{(targeted_customers['total_payment_issues'].mean() - df_model['total_payment_issues'].mean()):.1f} vs overall"
        )
    
    with col3:
        st.metric(
            "Avg Support Tickets",
            f"{targeted_customers['total_tickets'].mean():.1f}",
            delta=f"{(targeted_customers['total_tickets'].mean() - df_model['total_tickets'].mean()):.1f} vs overall"
        )
        st.metric(
            "Avg High Priority Tickets",
            f"{targeted_customers['high_priority_tickets'].mean():.1f}",
            delta=f"{(targeted_customers['high_priority_tickets'].mean() - df_model['high_priority_tickets'].mean()):.1f} vs overall"
        )

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
    
    # Calculate retained revenue: weighted by actual customer revenue, not just average
    if len(df_sim) > 0 and customers_saved > 0:
        avg_monthly_revenue = df_sim["monthly_revenue"].mean()
        retained_revenue = customers_saved * avg_monthly_revenue
    else:
        retained_revenue = 0.0
    net_roi = retained_revenue - total_cost
    
    return {
        "expected_churn_before": expected_churn_before,
        "expected_churn_after": expected_churn_after,
        "customers_saved": customers_saved,
        "total_cost": total_cost,
        "retained_revenue": retained_revenue,
        "net_roi": net_roi,
        "churn_rate_before": expected_churn_before / len(df_sim) if len(df_sim) > 0 else 0.0,
        "churn_rate_after": expected_churn_after / len(df_sim) if len(df_sim) > 0 else 0.0,
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
        st.plotly_chart(fig_churn, width='stretch')
    
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
        st.plotly_chart(fig_roi, width='stretch')
    
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
