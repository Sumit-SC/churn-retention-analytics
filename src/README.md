## Overview

This folder contains Python modules that power the churn / retention analytics demo.  
Stage 1 is a **synthetic data generator** that creates realistic SaaS product data in Parquet format. Later stages (via SQL) will transform this raw data into staging and feature tables for modeling and dashboards.

---

## `data_generator.py` – Stage 1 synthetic raw data

**Purpose**

- **Generate synthetic but realistic SaaS data** for:
  - `customers`
  - `usage_daily`
  - `billing`
  - `support`
- Save each table as a Parquet file under `data/raw/`.

**What it simulates**

- **Customers**
  - 100,000 customers with:
    - `customer_id`
    - `signup_date` over an 18‑month window
    - `plan` (`Free`, `Basic`, `Pro`, `Enterprise`)
    - `region` (`NA`, `EU`, `APAC`, `LATAM`)
- **Daily usage (`usage_daily`)**
  - Per‑customer, per‑day usage with:
    - `sessions`
    - `usage_minutes`
    - `feature_events`
  - Activity decays over time using a **retention probability curve** that depends on plan and days since signup.
  - Random “missing logs” are dropped to mimic real-world tracking noise.
- **Billing**
  - Only for non‑Free plans.
  - Bills generated at roughly 13‑day intervals.
  - Fields:
    - `bill_date`
    - `amount` (based on plan price)
    - `payment_status` (`paid`, `late`, `failed`) from a probabilistic model.
- **Support tickets**
  - Ticket counts scale with customer tenure.
  - Fields:
    - `ticket_date`
    - `category` (`billing`, `technical`, `onboarding`)
    - `priority` (`low`, `medium`, `high`)

**Outputs**

All files are written to `data/raw` (folder is created if needed):

- `customers.parquet`
- `usage_daily.parquet`
- `billing.parquet`
- `support.parquet`

**How to run Stage 1 (from project root)**

Using plain Python:

```bash
python -m src.data_generator
```

or, if you prefer running the file directly:

```bash
python src/data_generator.py
```

After this finishes, you should see the four Parquet files in `data/raw/`. The script prints row counts and total runtime to the console.

---

## `run_sql_pipeline.py` – SQL pipeline orchestrator

**Purpose**

Orchestrates **Stage 2–3 SQL transformations** by executing SQL files in order and exporting results.

**What it does**

1. **Connects to DuckDB** – Creates/connects to `churn.duckdb` database file in project root
2. **Executes SQL files in order:**
   - `sql/staging.sql` – Loads raw Parquet into DuckDB `staging.*` tables
   - `sql/data_quality.sql` – Runs data quality checks, creates `dq.*` tables
   - `sql/churn_features.sql` – Cleans data, aggregates features, creates `analytics.churn_features`
3. **Uses DuckDB's `read_parquet()`** – Reads Parquet files directly in SQL (no pandas loading)
4. **Prints row counts** – Shows counts for staging tables and analytics table
5. **Exports final table** – Writes `analytics.churn_features` to `data/analytics/churn_features.parquet`

**How to run**

From project root:

```bash
python src/run_sql_pipeline.py
```

**Prerequisites**

- Stage 1 completed (raw Parquet files in `data/raw/`)
- DuckDB installed (`pip install duckdb` or via `uv sync`)

**Outputs**

- DuckDB database: `churn.duckdb` (contains `staging`, `dq`, and `analytics` schemas)
- Parquet export: `data/analytics/churn_features.parquet`

**For detailed SQL pipeline documentation, see `sql/README.md`**

---

## `eda.py` – Exploratory Data Analysis

**Purpose**

Performs comprehensive EDA on cleaned churn features from `analytics.churn_features`. Generates visualizations and insights for reports and dashboards.

**Visualization Strategy**

- **Static plots (Seaborn/Matplotlib)**: Used for distribution analysis and explanatory visualizations that benefit from publication-quality static outputs. Saved as PNG for reports and presentations.
- **Interactive plots (Plotly)**: Used selectively for categorical comparisons (plan, region) and specialized visualizations (funnel) where interactivity adds value. Saved as HTML for dashboards.
- **No hypothesis testing**: This is exploratory analysis focused on pattern discovery, not statistical inference.

**What it does**

1. **Loads data** from `analytics.churn_features` in DuckDB
2. **Filters data** based on `INCLUDE_SOFT_CHURN` config (default: excludes NULL churn_label)
3. **Creates lifecycle view** with `lifecycle_stage` (Active, At Risk 31–45d, Churned) based on recency_days
4. **Generates visualizations**:
   - Lifecycle Distribution (static Seaborn countplot)
   - Lifecycle by Plan (interactive Plotly stacked bar chart)
   - Recency Density by Lifecycle (static Seaborn KDE plot)
   - Churn by Plan (interactive Plotly bar chart)
   - Churn by Region (interactive Plotly bar chart)
   - Usage Trend Distribution (static Seaborn histogram with KDE)
   - Recency Distribution (static Seaborn boxplot)
   - Retention Funnel (interactive Plotly funnel chart using lifecycle stages)
   - Support Load Distribution (static Seaborn boxplot)
5. **Extracts insights** including lifecycle-focused findings and saves to `key_insights.txt`

**Configuration**

- `INCLUDE_SOFT_CHURN`: If `False`, filters to only customers with non-null `churn_label` (excludes "uncertain" 30-45 day window). If `True`, includes all customers.

**How to run**

From project root:

```bash
python src/eda.py
```

**Prerequisites**

- Stage 2-3 SQL pipeline completed (`analytics.churn_features` table exists in DuckDB)
- Required Python packages: `duckdb`, `pandas`, `matplotlib`, `seaborn`, `plotly`

**Outputs**

All outputs saved to `eda_outputs/`:

- `lifecycle_distribution.png` (static Seaborn)
- `lifecycle_by_plan.html` (interactive Plotly)
- `recency_kde.png` (static Seaborn)
- `churn_by_plan.html` (interactive Plotly)
- `churn_by_region.html` (interactive Plotly)
- `usage_trend_distribution.png` (static Seaborn)
- `recency_distribution.png` (static Seaborn)
- `retention_funnel.html` (interactive Plotly)
- `support_load_distribution.png` (static Seaborn)
- `key_insights.txt` (text summary)

**Runtime**

Designed to complete in < 30 seconds for typical datasets.

