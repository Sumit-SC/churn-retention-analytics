## SQL Pipeline Overview

This folder contains the **SQL transformation pipeline** (Stages 2–3) that transforms raw Parquet data into clean staging tables and customer-level feature tables ready for modeling.

The pipeline is orchestrated by `src/run_sql_pipeline.py`, which executes SQL files in this exact order:

1. `staging.sql` – Load raw Parquet into DuckDB staging tables
2. `data_quality.sql` – Run data quality checks and report issues
3. `churn_features.sql` – Clean data, aggregate features, and build final analytics table

---

## Quick Start

**Run the entire SQL pipeline:**

```bash
python src/run_sql_pipeline.py
```

This will:
- Connect to DuckDB database (`churn.duckdb` in project root)
- Execute all SQL files in order
- Print row counts for staging and analytics tables
- Export `analytics.churn_features` to `data/analytics/churn_features.parquet`

**Prerequisites:**
- Stage 1 must be completed (raw Parquet files in `data/raw/`)
- DuckDB installed (via `uv sync` or `pip install duckdb`)

---

## Stage 2: Staging Layer (`staging.sql`)

**Purpose:** Load raw Parquet files into DuckDB staging tables with proper types.

**What it does:**

- Drops and recreates `staging` schema (deterministic runs)
- Creates four staging tables by reading Parquet directly:
  - `staging.customers` ← `data/raw/customers.parquet`
  - `staging.usage_daily` ← `data/raw/usage_daily.parquet`
  - `staging.billing` ← `data/raw/billing.parquet`
  - `staging.support` ← `data/raw/support.parquet`
- Applies type casting (BIGINT, DATE, TIMESTAMP, VARCHAR, INTEGER)
- **No aggregation, no cleaning** – just raw-to-staging materialization

**Outputs:**

- Tables in DuckDB `staging` schema
- Row counts printed by pipeline runner

---

## Data Quality Checks (`data_quality.sql`)

**Purpose:** Run data quality checks on staging tables and report issue counts (does NOT clean data).

**What it checks:**

### Usage Daily Checks

1. **Negative `usage_minutes`** – Counts rows where `usage_minutes < 0`
2. **Extreme `usage_minutes`** – Counts rows above 99.9th percentile (potential outliers)
3. **Session mismatch** – Counts rows where `sessions = 0` AND `usage_minutes > 0` (logical inconsistency)

### Billing Checks

4. **Zero paid amounts** – Counts rows where `amount = 0` AND `payment_status = 'paid'`
5. **Duplicate bills** – Counts duplicate `(customer_id, bill_date)` pairs

### Support Checks

6. **Pre-signup tickets** – Counts tickets where `ticket_date < signup_date` (temporal inconsistency)

**Outputs:**

- Individual check tables in `dq` schema (e.g., `dq.usage_daily_negative_minutes`)
- Summary table: `dq.data_quality_summary` with all issue counts

**Note:** This stage only **reports** issues. Actual cleaning happens in `churn_features.sql`.

---

## Stage 3: Feature Engineering (`churn_features.sql`)

**Purpose:** Clean data, aggregate to customer-level features, and build final analytics table.

**Execution order:**

1. **Data cleaning CTEs** (applied before aggregation)
2. **Feature aggregation CTEs**
3. **Final customer-level table**

---

### Data Cleaning

All cleaning happens in CTEs **before** aggregation to ensure clean features.

#### Usage Cleaning (`usage_clean` CTE)

- **Negative `usage_minutes`** → Replaced with `NULL`
- **Extreme `usage_minutes`** → Capped at 99.9th percentile
- **Session mismatch** → If `sessions = 0` AND `usage_minutes > 0`, set `sessions = 1`

#### Billing Cleaning (`billing_clean` CTE)

- **Duplicate removal** → `SELECT DISTINCT` removes exact duplicate rows
- **Zero paid flag** → Adds `is_zero_paid_flag` column (1 if `amount = 0` AND `payment_status = 'paid'`, else 0)

#### Support Cleaning (`support_clean` CTE)

- **Pre-signup removal** → Filters out tickets where `ticket_date < signup_date` (joins to `staging.customers`)

---

### Feature Aggregation

After cleaning, features are aggregated per customer:

#### Usage Features (`usage_agg`, `usage_trends`)

- `last_active_date` – Most recent usage date
- `active_days` – Count of distinct active days
- `avg_sessions` – Average sessions per day
- `avg_usage_minutes` – Average usage minutes per day
- `avg_usage_last_30d` – Average minutes in last 30 days
- `avg_usage_prev_30d` – Average minutes in previous 30 days (days 31–60)
- `usage_trend_30d` – Difference (last 30d – prev 30d)

#### Billing Features (`billing_agg`)

- `total_payment_issues` – Count of `late` or `failed` payments
- `failed_payments_30d` – Count of `failed` payments in last 30 days
- `zero_paid_anomalies` – Count of zero-amount paid bills (flag sum)

#### Support Features (`support_agg`)

- `total_tickets` – Total support tickets (after cleaning)
- `high_priority_tickets` – Count of high-priority tickets

---

### Final Table: `analytics.churn_features`

**One row per customer** with:

- **Customer attributes:** `customer_id`, `signup_date`, `plan`, `region`
- **Usage features:** All aggregated usage metrics (cleaned)
- **Billing features:** Payment health signals (with flags)
- **Support features:** Ticket counts (cleaned)
- **Derived features:**
  - `recency_days` – Days since last active date
  - `churn_label` – Binary churn indicator:
    - `1` if `recency_days > 45` (churned)
    - `0` if `recency_days < 30` (active)
    - `NULL` if `30 <= recency_days <= 45` (uncertain)

**Output:**

- Table: `analytics.churn_features` in DuckDB
- Parquet export: `data/analytics/churn_features.parquet` (created by `run_sql_pipeline.py`)

---

## Pipeline Execution Details

### How `run_sql_pipeline.py` Works

1. **Connects to DuckDB** – Creates/connects to `churn.duckdb` file in project root
2. **Reads SQL files** – Loads each `.sql` file as text
3. **Splits statements** – Splits on `;` and executes each statement separately
4. **Executes in order:**
   - `staging.sql` → Creates `staging.*` tables
   - `data_quality.sql` → Creates `dq.*` tables
   - `churn_features.sql` → Creates `analytics.churn_features`
5. **Prints row counts** – Shows counts for staging tables and analytics table
6. **Exports Parquet** – Writes `analytics.churn_features` to `data/analytics/churn_features.parquet`

### Database Schema Structure

After running the pipeline, DuckDB contains:

- **`staging` schema:**
  - `customers`, `usage_daily`, `billing`, `support`
- **`dq` schema:**
  - Individual check tables + `data_quality_summary`
- **`analytics` schema:**
  - `churn_features` (final customer-level table)

### Running Individual SQL Files

You can also run SQL files directly in DuckDB CLI or a SQL client:

```bash
# Example: Run only staging
duckdb churn.duckdb < sql/staging.sql

# Or connect interactively
duckdb churn.duckdb
```

---

## Data Flow Summary

```
data/raw/*.parquet
    ↓ (staging.sql)
staging.* tables (DuckDB)
    ↓ (data_quality.sql)
dq.* tables (issue reports)
    ↓ (churn_features.sql)
analytics.churn_features (DuckDB)
    ↓ (run_sql_pipeline.py export)
data/analytics/churn_features.parquet
```

---

## Troubleshooting

**Error: "Parser Error: syntax error"**
- Ensure SQL files end with semicolons
- Check for unmatched parentheses or quotes
- Verify DuckDB version supports all functions used

**Error: "Table not found"**
- Run `staging.sql` before `churn_features.sql`
- Ensure Stage 1 (data generation) completed successfully

**Empty or missing tables:**
- Check that `data/raw/*.parquet` files exist and have data
- Verify file paths in SQL are relative to project root

**Performance issues:**
- DuckDB handles millions of rows efficiently
- If slow, check disk I/O and available memory
- Consider using DuckDB's `SET memory_limit` if needed

---

## Extending the Pipeline

**Add new quality checks:**
- Add checks to `data_quality.sql`
- Follow existing pattern: `CREATE TABLE dq.check_name AS SELECT ...`

**Add new features:**
- Add aggregation CTEs in `churn_features.sql`
- Join new features in final SELECT statement

**Add new staging tables:**
- Add `CREATE TABLE staging.new_table AS SELECT ...` in `staging.sql`
- Update `run_sql_pipeline.py` row count printing if needed
