## Churn Retention Analytics

### Executive summary

This project is an **end‑to‑end churn and retention analytics sandbox** for a subscription/SaaS product.  
It generates synthetic but realistic customer, usage, billing, and support data; transforms it with SQL into modeling‑ready features; trains a lightweight churn model; and provides tooling to explore **drivers of churn** and **retention strategy scenarios**.

The repo is structured so you can treat it as a **portfolio‑ready case study** or a **playground** for experimenting with churn analytics.

---

## Project overview

- **Stage 1 – Synthetic data generation (Python)**
  - `src/data_generator.py` creates:
    - `customers.parquet`
    - `usage_daily.parquet`
    - `billing.parquet`
    - `support.parquet`
  - Written to `data/raw/`.  
  - Details: see `src/README.md` and `data/README.md`.

- **Stage 2 – Staging layer (SQL)**
  - `sql/staging.sql` reads raw Parquet files and builds cleaned, conformed **staging tables** in `data/staging/`.

- **Stage 3 – Feature engineering (SQL)**
  - `sql/churn_features.sql` aggregates staging tables into **customer‑level features** in `data/analytics/`.

- **Stage 4 – EDA and modeling (Python)**
  - `src/eda.py` explores churn drivers and data quality.
  - `src/train_model.py` trains a simple churn model (ML‑lite) using engineered features.

- **Stage 5 – Retention simulator / app**
  - `src/retention_simulator.py` and `app/streamlit_app.py` provide a way to simulate retention strategies and visualize impact.

For deeper details of each piece, see:

- `src/README.md` – Python modules and Stage 1 behavior  
- `data/README.md` – data layout and generation rules

---

## Business problem

You are an analytics/DS team at a subscription SaaS company.  
Churn is creeping up, and leadership wants answers to:

- **Who is at highest risk of churn?**
- **Why are they churning?**
- **Which levers (pricing, plan mix, engagement, support) are most impactful?**
- **What retention strategies are likely to move the needle?**

This project provides a **self‑contained environment** to explore these questions with synthetic data.

---

## Churn definition & assumptions

- **Customer unit:** one `customer_id` represents an account / company.
- **Churn definition (conceptual):**
  - A customer is considered churned if they **stop generating usage and billing events** beyond a certain time window (e.g., no activity in the last N days).
  - Exact SQL definition is implemented in the feature engineering SQL (to be defined in `sql/churn_features.sql`).
- **Assumptions baked into the generator:**
  - Higher‑tier plans (Pro/Enterprise) tend to have **higher retention** but may churn with different patterns than Free/Basic.
  - Regions differ slightly in plan mix and behavior.
  - Support interactions, billing failures, and usage drop‑off are all potential churn signals.

These assumptions are intentionally stylized to make the dataset useful for **storytelling and analysis**.

---

## Data design & scale

Stage 1 (`src/data_generator.py`) builds a **multi‑table event schema**:

- `customers` – one row per customer
  - Keys: `customer_id`
  - Features: `signup_date`, `plan`, `region`, etc.
- `usage_daily` – one row per customer per active day
  - Keys: `customer_id`, `date`
  - Metrics: `sessions`, `usage_minutes`, `feature_events`
- `billing` – recurring billing events for paying customers
  - Keys: `customer_id`, `bill_date`
  - Fields: `amount`, `payment_status`
- `support` – support tickets
  - Keys: `customer_id`, `ticket_date`
  - Fields: `category`, `priority`

The generator is tuned to produce **tens of thousands to millions of rows** across tables, big enough to feel realistic but still manageable on a laptop.

Data layout:

- Raw: `data/raw/` – generator outputs (see `data/README.md`)
- Staging: `data/staging/` – SQL‑cleaned tables
- Analytics: `data/analytics/` – feature tables for modeling and BI

---

## Analytics architecture

High‑level architecture:

1. **Synthetic data (Python)**
   - `src/data_generator.py` → `data/raw/*.parquet`
2. **SQL transformations**
   - `sql/staging.sql` → `data/staging/*.parquet`
   - `sql/churn_features.sql` → `data/analytics/*.parquet`
3. **Modeling & EDA (Python)**
   - `src/eda.py`, `src/train_model.py`
4. **App / simulator**
   - `app/streamlit_app.py` and `src/retention_simulator.py`

You can run each part independently, or wire them together into a small pipeline using `src/run_sql_pipeline.py` (see `src/README.md` for intent).

---

## Feature engineering (SQL)

Implemented (or planned) in:

- `sql/staging.sql` – joins, de‑duplication, handling of missingness, key normalization.
- `sql/churn_features.sql` – customer‑level features such as:
  - Lifetime / tenure
  - Rolling usage (e.g., last 7/30/90 days sessions, minutes)
  - Billing health (e.g., late/failed payments, ARPU)
  - Support interaction patterns (volume, category mix, priority)
  - Plan / pricing and region attributes

The goal is to mimic a **modern analytics stack** where feature engineering happens in SQL and modeling reads from a clean semantic layer.

---

## Exploratory Data Analysis

- `src/eda.py` can be used to:
  - Visualize churn rates by plan, region, tenure buckets.
  - Explore correlations between usage, billing behavior, and churn.
  - Validate that the generator is producing sensible patterns.

You are free to extend this with notebooks or additional scripts depending on your workflow.

---

## Churn Modeling (ML-lite)

- `src/train_model.py` is intended to:
  - Load churn feature tables from `data/analytics/`.
  - Define a target label (e.g., churn in next 30 days).
  - Train a simple model (e.g., logistic regression / tree‑based model).
  - Output performance metrics and feature importance summaries.

The aim is not SOTA modeling, but a **clear, explainable baseline** that supports analytics storytelling.

---

## Explainability & Insights

Depending on how far you want to go, the project can surface:

- **Global insights:** which features are generally most predictive of churn.
- **Segmented views:** high‑risk segments by plan, region, or tenure.
- **Scenario thinking:** what would it take to move churn from X% → Y%?

You can implement SHAP, partial dependence plots, or simple feature importance charts within `src/train_model.py` or companion notebooks.

---

## Retention Strategy Simulator

- `src/retention_simulator.py` and `app/streamlit_app.py` can be used to:
  - Play with **“what‑if” levers** (e.g., improved onboarding, better support SLAs, pricing changes).
  - Map changes in key behaviors back to expected churn impact using the trained model.

This turns the project into a **conversation tool** for product and growth teams.

---

## Key Business Recommendations

This section is meant to be filled after you run your own experiments, but typical themes might include:

- Focus retention efforts on **specific high‑risk segments** (e.g., early‑tenure Basic plan users in certain regions).
- Invest in **support quality** or **onboarding** if those prove to be strong churn drivers.
- Consider **plan design / pricing adjustments** if downgrades or payment failures cluster in particular offerings.

Use this section as a **storytelling summary** for your portfolio / case study.

---

## Tech Stack

- **Language:** Python 3.11+  
- **Environment / dependency management:** `uv` (recommended) or `pip`  
- **Data:** Parquet files on local disk  
- **SQL engine:** flexible (e.g., DuckDB, SQLite, or in‑memory via Python wrappers; adjust `run_sql_pipeline.py` accordingly)  
- **Visualization / app:** Streamlit for interactive dashboards and simulators  

Dependencies are specified in `pyproject.toml` (and locked in `uv.lock`).

---

## Setup & Installation

You can use either **uv (recommended)** or **plain pip**.

### Option 1 – Using `uv` (recommended)

1. **Install uv** (if not already installed) – see official docs: `https://github.com/astral-sh/uv`  
   On many systems:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   (On Windows, use the PowerShell installer from the uv docs.)

2. **Create and activate a virtual environment** (from project root):
   ```bash
   uv venv .venv
   # On Windows PowerShell
   .venv\Scripts\Activate.ps1
   # On macOS/Linux
   # source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   uv sync
   ```

### Option 2 – Using `pip`

1. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # On Windows PowerShell
   .venv\Scripts\Activate.ps1
   # On macOS/Linux
   # source .venv/bin/activate
   ```

2. **Install dependencies from `pyproject.toml`**:  
   If you prefer a simple `requirements.txt`, you can export from uv or add one manually. For a quick start:
   ```bash
   pip install -e .
   ```
   (Assumes your `pyproject.toml` is set up as a simple project; adjust as needed.)

---

## How to Run the Project

From the project root:

1. **Generate synthetic raw data (Stage 1)**
   ```bash
   python -m src.data_generator
   # or
   python src/data_generator.py
   ```
   - Outputs: `data/raw/customers.parquet`, `usage_daily.parquet`, `billing.parquet`, `support.parquet`
   - More: see `src/README.md` and `data/README.md`.

2. **Run SQL pipeline (Stages 2–3)**
   - Implement / configure `src/run_sql_pipeline.py` to:
     - Read `data/raw/*.parquet`
     - Execute `sql/staging.sql` and `sql/churn_features.sql`
     - Write results to `data/staging/` and `data/analytics/`
   - Then run:
     ```bash
     python -m src.run_sql_pipeline
     # or
     python src/run_sql_pipeline.py
     ```

3. **Run EDA / modeling (Stage 4)**
   - Example:
     ```bash
     python src/eda.py
     python src/train_model.py
     ```

4. **Launch Streamlit app / simulator (Stage 5)**
   - If you have a Streamlit app in `app/streamlit_app.py`:
     ```bash
     streamlit run app/streamlit_app.py
     ```

Feel free to run individual components as standalone scripts if you only need a subset of the pipeline.

---

## Screenshots & Demo

Add:

- Screenshots of key charts / dashboards
- GIF or link to a short video walkthrough of:
  - The app
  - The analytics notebook
  - Key model / EDA outputs

This section is intentionally left open for customization.

---

## Future Improvements

Ideas for extending the project:

- More realistic product/usage and pricing models
- Experiment tracking and model registry integration
- Orchestration (e.g., Airflow, Prefect, Dagster) for the full pipeline
- Cloud‑scale storage / compute options (e.g., S3, DuckDB, BigQuery)
- Additional explainability tooling (e.g., SHAP, LIME)
- More sophisticated retention strategy simulation and UI polish

