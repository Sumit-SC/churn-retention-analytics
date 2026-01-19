"""Exploratory Data Analysis for churn and retention data."""

import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.dpi"] = 300

INCLUDE_SOFT_CHURN = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "churn.duckdb"
OUTPUT_DIR = PROJECT_ROOT / "eda_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("EXPLORATORY DATA ANALYSIS - CHURN & RETENTION")
print("=" * 80)
print("\n1. LOADING DATA...")

conn = duckdb.connect(DB_PATH.as_posix())
df_full = conn.execute("SELECT * FROM analytics.churn_features").df()
conn.close()

print(f"   Shape: {df_full.shape[0]:,} rows × {df_full.shape[1]} columns")

df_full['lifecycle_stage'] = df_full['recency_days'].apply(
    lambda x: 'Active' if x <= 30 else ('At Risk (31–45d)' if x <= 45 else 'Churned')
)

df_lifecycle = df_full.copy()

if not INCLUDE_SOFT_CHURN:
    df = df_full[df_full['churn_label'].notna()].copy()
    print(f"   Filtered to labeled customers only (INCLUDE_SOFT_CHURN=False)")
else:
    df = df_full.copy()

df_labeled = df.copy()

churned = df['churn_label'] == 1
active = df['churn_label'] == 0
churn_rate = churned.sum() / (churned.sum() + active.sum()) * 100 if (churned.sum() + active.sum()) > 0 else 0
print(f"   Churn rate (labeled): {churn_rate:.2f}% ({churned.sum():,} churned / {active.sum():,} active)")

df['churn_status'] = df['churn_label'].map({1: 'Churned', 0: 'Active'})

print("\n" + "=" * 80)
print("2. LIFECYCLE DISTRIBUTION")
print("=" * 80)

lifecycle_counts = df_lifecycle['lifecycle_stage'].value_counts()
lifecycle_order = ['Active', 'At Risk (31–45d)', 'Churned']
lifecycle_counts = lifecycle_counts.reindex([s for s in lifecycle_order if s in lifecycle_counts.index])

print("\nLifecycle stage counts:")
for stage in lifecycle_order:
    if stage in lifecycle_counts.index:
        count = lifecycle_counts[stage]
        pct = (count / len(df_lifecycle)) * 100
        print(f"   {stage}: {count:,} ({pct:.1f}%)")

plt.figure(figsize=(10, 6))
sns.countplot(
    data=df_lifecycle,
    x='lifecycle_stage',
    hue='lifecycle_stage',
    order=lifecycle_order,
    palette={'Active': 'green', 'At Risk (31–45d)': 'orange', 'Churned': 'red'},
    legend=False
)
plt.title('Customer Lifecycle Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Lifecycle Stage', fontsize=12)
plt.ylabel('Number of Customers', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "lifecycle_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"\n   Saved: eda_outputs/lifecycle_distribution.png (static)")

print("\n" + "=" * 80)
print("3. LIFECYCLE BY PLAN")
print("=" * 80)

lifecycle_by_plan = df_lifecycle.groupby(['plan', 'lifecycle_stage']).size().reset_index(name='count')
lifecycle_by_plan_pivot = lifecycle_by_plan.pivot(index='plan', columns='lifecycle_stage', values='count').fillna(0)
lifecycle_by_plan_pivot = lifecycle_by_plan_pivot.reindex(columns=[s for s in lifecycle_order if s in lifecycle_by_plan_pivot.columns])

print("\nLifecycle by plan:")
print(lifecycle_by_plan_pivot)

fig = px.bar(
    lifecycle_by_plan,
    x='plan',
    y='count',
    color='lifecycle_stage',
    color_discrete_map={'Active': 'green', 'At Risk (31–45d)': 'orange', 'Churned': 'red'},
    title='Customer Lifecycle by Plan',
    labels={'count': 'Number of Customers', 'plan': 'Plan'},
    barmode='stack'
)
fig.update_layout(
    height=500,
    xaxis={'categoryorder': 'total descending'}
)
fig.write_html(str(OUTPUT_DIR / "lifecycle_by_plan.html"))
print(f"\n   Saved: eda_outputs/lifecycle_by_plan.html (interactive)")

print("\n" + "=" * 80)
print("4. RECENCY DENSITY BY LIFECYCLE")
print("=" * 80)

plt.figure(figsize=(12, 6))
df_lifecycle_recency = df_lifecycle[df_lifecycle['recency_days'].notna()].copy()
if len(df_lifecycle_recency) > 0:
    sns.kdeplot(
        data=df_lifecycle_recency,
        x='recency_days',
        hue='lifecycle_stage',
        common_norm=False,
        palette={'Active': 'green', 'At Risk (31–45d)': 'orange', 'Churned': 'red'}
    )
else:
    plt.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=plt.gca().transAxes)

plt.title('Recency Density by Lifecycle Stage', fontsize=14, fontweight='bold')
plt.xlabel('Recency (days)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "recency_kde.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"\n   Saved: eda_outputs/recency_kde.png (static)")

print("\n" + "=" * 80)
print("5. OVERVIEW STATISTICS")
print("=" * 80)

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
numeric_cols = [col for col in numeric_cols if col != 'customer_id']

print("\nNumeric columns summary:")
print(df[numeric_cols].describe().round(2))

print("\nKey medians:")
print(f"   Recency days: {df['recency_days'].median():.1f}")
print(f"   Usage trend 30d: {df['usage_trend_30d'].median():.2f}")
print(f"   Total payment issues: {df['total_payment_issues'].median():.0f}")
print(f"   Failed payments (30d): {df['failed_payments_30d'].median():.0f}")
print(f"   Total tickets: {df['total_tickets'].median():.0f}")
print(f"   Active days: {df['active_days'].median():.0f}")
print(f"   Avg sessions: {df['avg_sessions'].median():.2f}")
print(f"   Avg usage minutes: {df['avg_usage_minutes'].median():.1f}")

print("\n" + "=" * 80)
print("6. CHURN BY PLAN")
print("=" * 80)

churn_by_plan = df.groupby('plan').agg({
    'churn_label': lambda x: ((x == 1).sum() / ((x == 1).sum() + (x == 0).sum()) * 100) if ((x == 1).sum() + (x == 0).sum()) > 0 else 0,
    'customer_id': 'count'
}).rename(columns={'churn_label': 'churn_rate', 'customer_id': 'total_customers'})
churn_by_plan = churn_by_plan.sort_values('churn_rate', ascending=False)

print("\nChurn rate by plan:")
print(churn_by_plan.round(2))

fig = px.bar(
    churn_by_plan.reset_index(),
    x='plan',
    y='churn_rate',
    title='Churn Rate by Plan',
    labels={'churn_rate': 'Churn Rate (%)', 'plan': 'Plan'},
    text='churn_rate',
    color='churn_rate',
    color_continuous_scale='Reds'
)
fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig.update_layout(
    showlegend=False,
    height=500,
    xaxis={'categoryorder': 'total descending'}
)
fig.write_html(str(OUTPUT_DIR / "churn_by_plan.html"))
print(f"\n   Saved: eda_outputs/churn_by_plan.html (interactive)")

print("\n" + "=" * 80)
print("7. CHURN BY REGION")
print("=" * 80)

churn_by_region = df.groupby('region').agg({
    'churn_label': lambda x: ((x == 1).sum() / ((x == 1).sum() + (x == 0).sum()) * 100) if ((x == 1).sum() + (x == 0).sum()) > 0 else 0,
    'customer_id': 'count'
}).rename(columns={'churn_label': 'churn_rate', 'customer_id': 'total_customers'})
churn_by_region = churn_by_region.sort_values('churn_rate', ascending=False)

print("\nChurn rate by region:")
print(churn_by_region.round(2))

fig = px.bar(
    churn_by_region.reset_index(),
    x='region',
    y='churn_rate',
    title='Churn Rate by Region',
    labels={'churn_rate': 'Churn Rate (%)', 'region': 'Region'},
    text='churn_rate',
    color='churn_rate',
    color_continuous_scale='Oranges'
)
fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig.update_layout(
    showlegend=False,
    height=500,
    xaxis={'categoryorder': 'total descending'}
)
fig.write_html(str(OUTPUT_DIR / "churn_by_region.html"))
print(f"\n   Saved: eda_outputs/churn_by_region.html (interactive)")

print("\n" + "=" * 80)
print("8. USAGE TREND DISTRIBUTION")
print("=" * 80)

print(f"\nComparing usage_trend_30d for {len(df):,} customers...")
active_median = df[df['churn_label'] == 0]['usage_trend_30d'].median()
churned_median = df[df['churn_label'] == 1]['usage_trend_30d'].median()
print(f"   Active median: {active_median:.2f}" if not pd.isna(active_median) else "   Active median: N/A (no active customers)")
print(f"   Churned median: {churned_median:.2f}" if not pd.isna(churned_median) else "   Churned median: N/A")

plt.figure(figsize=(12, 6))
df_usage = df[df['usage_trend_30d'].notna()].copy()
if len(df_usage) > 0:
    has_both_statuses = len(df_usage['churn_status'].unique()) > 1
    use_kde = len(df_usage) >= 100 and has_both_statuses
    
    if has_both_statuses:
        sns.histplot(
            data=df_usage,
            x='usage_trend_30d',
            hue='churn_status',
            bins=50,
            kde=use_kde,
            alpha=0.6,
            palette={'Active': 'green', 'Churned': 'red'}
        )
    else:
        sns.histplot(
            data=df_usage,
            x='usage_trend_30d',
            bins=50,
            kde=use_kde,
            alpha=0.6,
            color='red' if df_usage['churn_status'].iloc[0] == 'Churned' else 'green'
        )
else:
    print("   Warning: No valid usage_trend_30d data to plot")
    plt.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=plt.gca().transAxes)

plt.title('Usage Trend (30d) Distribution by Churn Status', fontsize=14, fontweight='bold')
plt.xlabel('Usage Trend 30d (minutes)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
if len(df_usage) > 0 and len(df_usage['churn_status'].unique()) > 1:
    plt.legend(title='Churn Status')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "usage_trend_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"\n   Saved: eda_outputs/usage_trend_distribution.png (static)")

print("\n" + "=" * 80)
print("9. RECENCY DISTRIBUTION")
print("=" * 80)

print(f"\nRecency distribution analysis...")
active_recency = df[df['churn_label'] == 0]['recency_days'].median()
churned_recency = df[df['churn_label'] == 1]['recency_days'].median()
print(f"   Active median recency: {active_recency:.1f} days" if not pd.isna(active_recency) else "   Active median recency: N/A (no active customers)")
print(f"   Churned median recency: {churned_recency:.1f} days" if not pd.isna(churned_recency) else "   Churned median recency: N/A")

plt.figure(figsize=(10, 6))
has_both_statuses = len(df['churn_status'].unique()) > 1
if has_both_statuses:
    sns.boxplot(
        data=df,
        x='churn_status',
        y='recency_days',
        palette={'Active': 'green', 'Churned': 'red'}
    )
else:
    status = df['churn_status'].iloc[0]
    color = 'red' if status == 'Churned' else 'green'
    sns.boxplot(
        data=df,
        y='recency_days',
        color=color
    )
    plt.xlabel('Churn Status', fontsize=12)
    plt.xticks([0], [status])
plt.title('Recency Distribution by Churn Status', fontsize=14, fontweight='bold')
if has_both_statuses:
    plt.xlabel('Churn Status', fontsize=12)
plt.ylabel('Recency (days)', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "recency_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"\n   Saved: eda_outputs/recency_distribution.png (static)")

print("\n" + "=" * 80)
print("10. RETENTION FUNNEL")
print("=" * 80)

active_count = len(df_lifecycle[df_lifecycle['lifecycle_stage'] == 'Active'])
at_risk_count = len(df_lifecycle[df_lifecycle['lifecycle_stage'] == 'At Risk (31–45d)'])
churned_count = len(df_lifecycle[df_lifecycle['lifecycle_stage'] == 'Churned'])
total_customers = len(df_lifecycle)

funnel_data = pd.DataFrame({
    'Stage': ['Total Customers', 'Active (<=30 days)', 'At Risk (31-45 days)', 'Churned (>45 days)'],
    'Count': [total_customers, active_count, at_risk_count, churned_count]
})

print("\nRetention funnel:")
print(funnel_data.to_string())

fig = go.Figure(go.Funnel(
    y=funnel_data['Stage'],
    x=funnel_data['Count'],
    textposition="inside",
    textinfo="value+percent initial",
    marker={"color": ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]}
))
fig.update_layout(
    title='Customer Retention Funnel',
    height=500
)
fig.write_html(str(OUTPUT_DIR / "retention_funnel.html"))
print(f"\n   Saved: eda_outputs/retention_funnel.html (interactive)")

print("\n" + "=" * 80)
print("11. SUPPORT LOAD DISTRIBUTION")
print("=" * 80)

print(f"\nSupport ticket distribution analysis...")
active_tickets_med = df[df['churn_label'] == 0]['total_tickets'].median()
churned_tickets_med = df[df['churn_label'] == 1]['total_tickets'].median()
print(f"   Active median tickets: {active_tickets_med:.0f}" if not pd.isna(active_tickets_med) else "   Active median tickets: N/A (no active customers)")
print(f"   Churned median tickets: {churned_tickets_med:.0f}" if not pd.isna(churned_tickets_med) else "   Churned median tickets: N/A")

plt.figure(figsize=(10, 6))
has_both_statuses = len(df['churn_status'].unique()) > 1
if has_both_statuses:
    sns.boxplot(
        data=df,
        x='churn_status',
        y='total_tickets',
        palette={'Active': 'green', 'Churned': 'red'}
    )
else:
    status = df['churn_status'].iloc[0]
    color = 'red' if status == 'Churned' else 'green'
    sns.boxplot(
        data=df,
        y='total_tickets',
        color=color
    )
    plt.xticks([0], [status])
plt.title('Support Ticket Distribution by Churn Status', fontsize=14, fontweight='bold')
plt.xlabel('Churn Status', fontsize=12)
plt.ylabel('Total Tickets', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "support_load_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"\n   Saved: eda_outputs/support_load_distribution.png (static)")

print("\n" + "=" * 80)
print("12. KEY INSIGHTS")
print("=" * 80)

insights = []

highest_plan_churn = churn_by_plan.index[0]
highest_plan_rate = churn_by_plan.loc[highest_plan_churn, 'churn_rate']
lowest_plan_churn = churn_by_plan.index[-1]
lowest_plan_rate = churn_by_plan.loc[lowest_plan_churn, 'churn_rate']
insights.append(f"• Plan '{highest_plan_churn}' has the highest churn rate ({highest_plan_rate:.1f}%), while '{lowest_plan_churn}' has the lowest ({lowest_plan_rate:.1f}%)")

highest_region_churn = churn_by_region.index[0]
highest_region_rate = churn_by_region.loc[highest_region_churn, 'churn_rate']
insights.append(f"• Region '{highest_region_churn}' shows the highest churn rate ({highest_region_rate:.1f}%)")

at_risk_pct = (at_risk_count / total_customers) * 100
insights.append(f"• {at_risk_pct:.1f}% of customers are in At Risk (31–45d) stage, representing a key intervention opportunity")

at_risk_by_plan = df_lifecycle[df_lifecycle['lifecycle_stage'] == 'At Risk (31–45d)'].groupby('plan').size()
if len(at_risk_by_plan) > 0:
    highest_at_risk_plan = at_risk_by_plan.idxmax()
    highest_at_risk_count = at_risk_by_plan.max()
    plan_total = df_lifecycle[df_lifecycle['plan'] == highest_at_risk_plan].shape[0]
    highest_at_risk_pct = (highest_at_risk_count / plan_total) * 100 if plan_total > 0 else 0
    insights.append(f"• Plan '{highest_at_risk_plan}' has the highest concentration of At Risk customers ({highest_at_risk_pct:.1f}% of plan customers)")

active_trend = df[df['churn_label'] == 0]['usage_trend_30d'].median()
churned_trend = df[df['churn_label'] == 1]['usage_trend_30d'].median()
insights.append(f"• Churned customers show {churned_trend:.2f} min/day usage trend vs {active_trend:.2f} for active customers (median)")

active_tickets = df[df['churn_label'] == 0]['total_tickets'].median()
churned_tickets = df[df['churn_label'] == 1]['total_tickets'].median()
insights.append(f"• Churned customers have median {churned_tickets:.0f} support tickets vs {active_tickets:.0f} for active customers")

insights.append(f"• Overall churn rate: {churn_rate:.2f}% ({churned.sum():,} churned / {active.sum():,} active)")

high_recency = df[df['churn_label'] == 1]['recency_days'].median()
low_recency = df[df['churn_label'] == 0]['recency_days'].median()
insights.append(f"• Churned customers have median recency of {high_recency:.0f} days vs {low_recency:.0f} days for active customers")

print("\n")
for insight in insights:
    print(insight)

insights_file = OUTPUT_DIR / "key_insights.txt"
with open(insights_file, 'w') as f:
    f.write("KEY INSIGHTS - CHURN & RETENTION ANALYSIS\n")
    f.write("=" * 80 + "\n\n")
    for insight in insights:
        f.write(insight + "\n")
print(f"\n   Saved: eda_outputs/key_insights.txt")

print("\n" + "=" * 80)
print("EDA COMPLETE")
print("=" * 80)
print(f"\nAll plots saved to: {OUTPUT_DIR}")
print("\nGenerated files:")
print("  • lifecycle_distribution.png (static Seaborn)")
print("  • lifecycle_by_plan.html (interactive Plotly)")
print("  • recency_kde.png (static Seaborn)")
print("  • churn_by_plan.html (interactive Plotly)")
print("  • churn_by_region.html (interactive Plotly)")
print("  • usage_trend_distribution.png (static Seaborn)")
print("  • recency_distribution.png (static Seaborn)")
print("  • retention_funnel.html (interactive Plotly)")
print("  • support_load_distribution.png (static Seaborn)")
print("  • key_insights.txt")
print("\n" + "=" * 80)
