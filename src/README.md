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

## `run_sql_pipeline.py` – SQL staging and feature pipeline (overview)

**Intended role**

- Orchestrate **Stage 2+** transformations:
  - Read raw Parquet files from `data/raw/`.
  - Execute SQL in:
    - `sql/staging.sql` – build cleaned, joined **staging** tables.
    - `sql/churn_features.sql` – build customer‑level **feature tables** for churn modeling and analytics.
  - Write results to:
    - `data/staging/`
    - `data/analytics/`

**High-level flow (design)**

1. Load `customers.parquet`, `usage_daily.parquet`, `billing.parquet`, `support.parquet` from `data/raw/`.
2. Register them as tables in an in‑process SQL engine (e.g., DuckDB, SQLite, or similar).
3. Run the SQL files in `sql/` to:
   - Create normalized / denoised staging tables.
   - Aggregate to customer‑level features (usage, billing, support behavior, plan/region, etc.).
4. Persist the resulting tables as Parquet (or a database) under `data/staging` and `data/analytics`.

> Note: The exact implementation details of `run_sql_pipeline.py` may evolve, but this README documents the **intended contract** between:
> - **Stage 1** Python generators (`data/raw/`),
> - and the **SQL-based** staging / feature engineering steps (`sql/` → `data/staging`, `data/analytics`).

