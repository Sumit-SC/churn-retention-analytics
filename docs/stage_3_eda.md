# Stage 3: EDA & Lifecycle Analysis

## Overview

Stage 3 performs comprehensive exploratory data analysis on cleaned churn features. The analysis introduces a lifecycle-based segmentation that makes the 31–45 day at-risk segment visible and actionable for business decisions.

---

## Objectives

- Identify churn drivers through statistical analysis
- Segment customers into lifecycle stages (Active, At Risk, Churned)
- Visualize patterns across plan tiers, regions, and behavioral features
- Generate publication-quality static plots and interactive dashboards
- Extract key insights for retention strategy

---

## Lifecycle Segmentation

The EDA introduces a **lifecycle view** that complements the hard churn label:

- **Active:** `recency_days <= 30`
- **At Risk (31–45d):** `30 < recency_days <= 45`
- **Churned:** `recency_days > 45`

This segmentation makes the at-risk segment (previously hidden as `churn_label = NULL`) visible for intervention.

---

## Visualizations

**File:** `src/eda.py`

### Static Plots (Seaborn/Matplotlib)
Saved as PNG for reports and presentations:

1. **Lifecycle Distribution** — Countplot showing customer distribution across lifecycle stages
2. **Recency Density by Lifecycle** — KDE plot showing recency distribution by stage
3. **Usage Trend Distribution** — Histogram with KDE comparing active vs churned
4. **Recency Distribution** — Boxplot by churn status
5. **Support Load Distribution** — Boxplot comparing ticket volume by churn status

### Interactive Plots (Plotly)
Saved as HTML for dashboards:

1. **Lifecycle by Plan** — Stacked bar chart showing lifecycle distribution across plan tiers
2. **Churn by Plan** — Bar chart comparing churn rates by plan
3. **Churn by Region** — Bar chart comparing churn rates by region
4. **Retention Funnel** — Funnel chart showing customer progression through lifecycle stages

---

## Key Findings

### Lifecycle Distribution
- **Active (≤30 days):** 86.2% of customers
- **At Risk (31–45d):** 6.3% of customers ← **Key intervention opportunity**
- **Churned (>45 days):** 7.4% of customers

### Churn by Plan
- **Free:** 13.2% churn rate (highest)
- **Basic:** 5.4% churn rate
- **Pro:** 2.5% churn rate
- **Enterprise:** 0.9% churn rate (lowest)

### Behavioral Patterns
- **Support Tickets:** Churned customers have median 6 tickets vs 4 for active (50% higher)
- **Recency:** Churned customers show median 63 days since last activity vs 5 days for active
- **Usage Trends:** Active customers maintain consistent usage; churned show declining trends

### At-Risk Concentration
- **Free Plan:** 8.8% of Free plan customers are At Risk (highest concentration)
- **Intervention Window:** 6,347 customers in 31–45 day window

---

## How to Run

```bash
python src/eda.py
```

**Runtime:** < 30 seconds

**Prerequisites:**
- Stage 2 completed (`analytics.churn_features` table exists)

---

## Outputs

All outputs saved to `eda_outputs/`:

**Static Plots (PNG):**
- `lifecycle_distribution.png`
- `recency_kde.png`
- `usage_trend_distribution.png`
- `recency_distribution.png`
- `support_load_distribution.png`

**Interactive Plots (HTML):**
- `lifecycle_by_plan.html`
- `churn_by_plan.html`
- `churn_by_region.html`
- `retention_funnel.html`

**Insights:**
- `key_insights.txt` — Text summary of key findings

---

## Configuration

**`INCLUDE_SOFT_CHURN`** (default: `False`):
- If `False`: Filters to only customers with non-null `churn_label` (excludes at-risk segment for modeling EDA)
- If `True`: Includes all customers (full lifecycle analysis)

---

## Next Steps

After Stage 3, proceed to **[Stage 4: ML-lite Modeling & Explainability](stage_4_modeling.md)** for predictive modeling.
