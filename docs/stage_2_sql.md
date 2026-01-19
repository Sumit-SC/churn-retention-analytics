# Stage 2: SQL Pipelines & Data Quality

## Overview

Stage 2 transforms raw Parquet data through a SQL-based pipeline using DuckDB. The pipeline performs staging, data quality checks, and feature engineering to create a modeling-ready dataset.

---

## Objectives

- Load raw Parquet files into DuckDB staging tables
- Run data quality checks (reporting, not cleaning)
- Clean data anomalies (negative usage, extreme outliers, logical inconsistencies)
- Engineer features for churn prediction
- Create `analytics.churn_features` table for modeling

---

## Pipeline Steps

**Orchestrator:** `pipeline/run_sql_pipeline.py`

### Step 1: Staging (`sql/staging.sql`)

Loads raw Parquet files into DuckDB staging schema:

- `staging.customers`
- `staging.usage_daily`
- `staging.billing`
- `staging.support`

**Uses DuckDB's `read_parquet()`** for direct file reading (no pandas loading).

### Step 2: Data Quality Checks (`sql/data_quality.sql`)

Creates `dq.*` tables reporting issues:

**Usage Daily:**
- Rows with `usage_minutes < 0`
- Rows with `usage_minutes > P99.9`
- Rows with `sessions = 0` AND `usage_minutes > 0`

**Billing:**
- Rows with `amount = 0` AND `payment_status = 'paid'`
- Duplicate `(customer_id, bill_date)` pairs

**Support:**
- Tickets where `ticket_date < signup_date`

**Note:** Data quality checks report issues but do not clean data.

### Step 3: Feature Engineering (`sql/churn_features.sql`)

#### Data Cleaning CTEs

**Usage Cleaning:**
- Replace `usage_minutes < 0` with NULL
- Cap `usage_minutes` at 99.9th percentile
- Fix `sessions = 0` AND `usage_minutes > 0` → set `sessions = 1`

**Billing Cleaning:**
- Flag zero-amount paid bills
- Remove exact duplicate billing rows

**Support Cleaning:**
- Remove tickets where `ticket_date < signup_date`

#### Feature Aggregation

**Usage Features:**
- `last_active_date`: Most recent usage date
- `active_days`: Count of distinct active days
- `avg_sessions`: Average sessions per day
- `avg_usage_minutes`: Average usage minutes per day
- `avg_usage_last_30d`: Average minutes in last 30 days
- `avg_usage_prev_30d`: Average minutes in previous 30 days (days 31–60)
- `usage_trend_30d`: Difference (last 30d – prev 30d)

**Billing Features:**
- `total_payment_issues`: Count of late or failed payments
- `failed_payments_30d`: Count of failed payments in last 30 days
- `zero_paid_anomalies`: Count of zero-amount paid bills

**Support Features:**
- `total_tickets`: Total support tickets (after cleaning)
- `high_priority_tickets`: Count of high-priority tickets

**Derived Features:**
- `recency_days`: Days since last active date (using `observation_date` CTE)
- `churn_label`: Binary indicator
  - `1` if `recency_days > 45` (churned)
  - `0` if `recency_days < 30` (active)
  - `NULL` if `30 <= recency_days <= 45` (uncertain/at-risk)

---

## Key Technical Details

### Temporal Logic Fix

The pipeline uses an `observation_date` CTE to handle historical data:

```sql
observation_date AS (
    SELECT MAX(date) AS obs_date
    FROM staging.usage_daily
)
```

All time-based calculations (30-day trends, recency, churn labeling) use `observation_date` instead of `CURRENT_DATE` to align with the historical nature of synthetic data.

### Output Table

**`analytics.churn_features`** — One row per customer with:
- Customer attributes (ID, signup_date, plan, region)
- All aggregated usage, billing, and support features
- Derived features (recency_days, churn_label)

---

## How to Run

```bash
python pipeline/run_sql_pipeline.py
```

**Runtime:** ~5–10 seconds

**Console Output:**
- Row counts for staging tables
- Row count for `analytics.churn_features`
- Export confirmation for `data/analytics/churn_features.parquet`

---

## Outputs

- **DuckDB database:** `churn.duckdb` (contains `staging`, `dq`, and `analytics` schemas)
- **Parquet export:** `data/analytics/churn_features.parquet`
- **Data quality reports:** `dq.*` tables in DuckDB

---

## Next Steps

After Stage 2, proceed to **[Stage 3: EDA & Lifecycle Analysis](stage_3_eda.md)** for exploratory analysis and visualization.
