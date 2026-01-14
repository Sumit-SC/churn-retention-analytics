## Data folder overview

This folder holds all data used in the churn / retention analytics project.  
Stage 1 creates **synthetic raw Parquet tables**; later stages (SQL and modeling) build on top of these.

---

## Folder structure

- `raw/`
  - **Input / base layer** created by Stage 1 (`src/data_generator.py`).
  - Contains the “system of record” synthetic tables:
    - `customers.parquet`
    - `usage_daily.parquet`
    - `billing.parquet`
    - `support.parquet`
- `staging/`
  - **Intermediate layer** created by SQL in `sql/staging.sql`.
  - Holds cleaned / conformed tables ready for feature engineering and modeling.
- `analytics/`
  - **Downstream layer** created by SQL in `sql/churn_features.sql` (and/or modeling scripts).
  - Holds denormalized, analysis‑friendly tables (e.g., customer‑level churn features).

Empty folders are tracked with `.gitkeep` so the structure is visible before data is generated.

---

## Stage 1 – synthetic raw data generation

**What Stage 1 does**

Running `src/data_generator.py` will:

- Create 100k synthetic customers with:
  - Plan, region, and signup dates over an 18‑month period.
- Simulate:
  - **Daily product usage** (`usage_daily.parquet`) with sessions, minutes, and feature events.
  - **Recurring billing** (`billing.parquet`) with bill dates, amounts, and payment status.
  - **Support tickets** (`support.parquet`) with ticket dates, categories, and priorities.
- Save all four tables as Parquet files under `data/raw/`.

**How to (re)generate raw data**

From the project root (`churn-retention-analytics`):

```bash
python -m src.data_generator
```

or:

```bash
python src/data_generator.py
```

This will (re)create the following files:

- `data/raw/customers.parquet`
- `data/raw/usage_daily.parquet`
- `data/raw/billing.parquet`
- `data/raw/support.parquet`

> Regenerating will overwrite existing Parquet files in `data/raw/`.  
> Run this before any SQL or modeling steps so downstream stages see a complete, consistent dataset.

---

## Later stages (SQL and modeling) – brief overview

- **SQL staging (`sql/staging.sql`)**
  - Reads from `data/raw/`.
  - Produces cleaned, joined staging tables in `data/staging/`.
- **SQL feature engineering (`sql/churn_features.sql`)**
  - Reads from `data/staging/`.
  - Produces customer‑level feature tables in `data/analytics/` for churn modeling and dashboarding.

These stages are typically orchestrated by a Python SQL runner (e.g., `src/run_sql_pipeline.py`) or executed manually in a SQL engine, using the Parquet files in this folder as the underlying data source.

