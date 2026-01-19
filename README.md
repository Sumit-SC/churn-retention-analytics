# Churn & Retention Analytics

## Executive Summary

This project is an **end-to-end churn and retention analytics sandbox** that demonstrates a complete data science workflow from synthetic data generation to actionable business insights. The system processes 100,000+ customer records through a SQL-based feature engineering pipeline and produces comprehensive lifecycle analysis with clear intervention opportunities.

### Key Findings

- **Overall churn rate: 7.58%** (7,020 churned / 85,590 active customers)
- **6.3% of customers** (6,347) are in the **At Risk (31–45d) stage**, representing a critical intervention opportunity
- **Plan 'Free'** has the highest churn rate (13.2%) and highest concentration of At Risk customers (8.8% of plan base)
- **Plan 'Enterprise'** shows the lowest churn rate (0.9%), demonstrating strong retention for premium tiers
- **Churned customers** have significantly higher support ticket volume (median 6 vs 4 for active) and longer recency (63 days vs 5 days)
- **Region 'APAC'** shows the highest churn rate (7.7%) across all regions

These insights enable targeted retention strategies focused on high-risk segments and intervention timing.

---

## Project Objectives

This project addresses critical business questions for a subscription SaaS company:

1. **Who is at highest risk of churn?** → Lifecycle segmentation identifies 6.3% of customers in the 31–45 day window
2. **Why are they churning?** → Feature analysis reveals plan tier, support load, and usage patterns as key drivers
3. **Which levers are most impactful?** → Plan design, support quality, and early engagement emerge as primary factors
4. **What retention strategies work?** → Targeted interventions for At Risk segments show highest ROI potential

The project provides a **self-contained environment** to explore these questions with realistic synthetic data and production-grade analytics workflows.

---

## Architecture Overview

```
┌─────────────────┐
│  Stage 1: Data  │  Python: Generate synthetic customer, usage, billing, support data
│   Generation    │  → data/raw/*.parquet
└────────┬────────┘
         │
┌────────▼────────┐
│  Stage 2: SQL  │  DuckDB: Staging, data quality checks, feature engineering
│  Pipeline      │  → analytics.churn_features
└────────┬────────┘
         │
┌────────▼────────┐
│  Stage 3: EDA   │  Python: Lifecycle analysis, churn drivers, visualizations
│  & Insights     │  → eda_outputs/*.html, *.png
└─────────────────┘
```

**Tech Stack:** Python 3.11+, DuckDB, Pandas, Seaborn, Plotly, Matplotlib

---

## Procedure Steps

### Stage 1: Synthetic Data Generation

Generate realistic SaaS customer data with retention probability curves.

**Run:**
```bash
python src/data_generator.py
```

**Outputs:** `data/raw/*.parquet` (customers, usage_daily, billing, support)

**Details:** See [`src/README.md`](src/README.md) and [`data/README.md`](data/README.md)

---

### Stage 2: SQL Pipeline

Transform raw data through staging, data quality checks, and feature engineering.

**Run:**
```bash
python src/run_sql_pipeline.py
```

**Pipeline Steps:**
1. **Staging** (`sql/staging.sql`) → Loads raw Parquet into DuckDB staging tables
2. **Data Quality** (`sql/data_quality.sql`) → Reports data quality issues (does not clean)
3. **Feature Engineering** (`sql/churn_features.sql`) → Cleans data, aggregates features, creates `analytics.churn_features`

**Outputs:**
- DuckDB database: `churn.duckdb`
- Parquet export: `data/analytics/churn_features.parquet`
- Data quality reports: `dq.*` tables

**Details:** See [`sql/README.md`](sql/README.md)

---

### Stage 3: Exploratory Data Analysis

Perform lifecycle-based EDA to identify churn drivers and intervention opportunities.

**Run:**
```bash
python src/eda.py
```

**Analysis Includes:**
- **Lifecycle Distribution** → Active, At Risk (31–45d), Churned segmentation
- **Lifecycle by Plan** → Plan-level risk concentration analysis
- **Recency Density** → Distribution analysis by lifecycle stage
- **Churn by Plan/Region** → Categorical churn rate comparisons
- **Usage Trends** → Usage pattern differences between active and churned
- **Support Load** → Ticket volume impact on churn
- **Retention Funnel** → Customer progression through lifecycle stages

**Outputs:** `eda_outputs/*.html`, `eda_outputs/*.png`, `eda_outputs/key_insights.txt`

**Details:** See [`src/README.md`](src/README.md#eda-py--exploratory-data-analysis)

**Interactive Exploration:** See [`notebooks/eda_exploration.ipynb`](notebooks/eda_exploration.ipynb)

---

## Key Findings & Results

### Lifecycle Distribution

- **Active (≤30 days):** 86,238 customers (86.2%)
- **At Risk (31–45d):** 6,347 customers (6.3%) ← **Key intervention opportunity**
- **Churned (>45 days):** 7,415 customers (7.4%)

### Churn by Plan

| Plan       | Churn Rate | Key Insight                          |
|------------|------------|--------------------------------------|
| Free       | 13.2%      | Highest churn, highest At Risk concentration (8.8%) |
| Basic      | 5.4%       | Moderate churn risk                  |
| Pro        | 2.5%       | Low churn, strong retention          |
| Enterprise | 0.9%       | Lowest churn, premium tier stability |

### Churn by Region

- **APAC:** 7.7% (highest)
- **EU:** 7.7%
- **NA:** 7.5%
- **LATAM:** 7.3% (lowest)

### Behavioral Patterns

- **Support Tickets:** Churned customers have median 6 tickets vs 4 for active (50% higher)
- **Recency:** Churned customers show median 63 days since last activity vs 5 days for active
- **Usage Trends:** Active customers maintain consistent usage patterns; churned show declining trends

### Intervention Opportunities

1. **At Risk Segment (6,347 customers):** Immediate intervention window (31–45 days inactive)
2. **Free Plan Focus:** 8.8% of Free plan customers are At Risk, highest concentration
3. **Support Proactivity:** Higher ticket volume correlates with churn; proactive support may reduce risk

---

## Conclusion

This project demonstrates a **production-ready analytics workflow** for churn and retention analysis:

1. **Data Generation:** Realistic synthetic data with retention probability curves
2. **SQL Pipeline:** Scalable feature engineering with data quality checks
3. **Lifecycle Analysis:** Business-focused EDA that identifies actionable intervention opportunities
4. **Insights-Driven:** Clear findings that support retention strategy decisions

**Business Impact:**
- Identified 6.3% of customer base in critical intervention window
- Quantified plan-tier impact on retention (13.2% vs 0.9% churn range)
- Revealed support load as a key churn indicator
- Enabled targeted retention strategies for high-risk segments

The system is designed to scale to millions of customers while maintaining sub-30-second EDA runtime and clear, actionable outputs.

---

## Quick Start

### Prerequisites

- Python 3.11+
- `uv` (recommended) or `pip`

### Installation

**Using `uv` (recommended):**
```bash
uv venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
uv sync
```

**Using `pip`:**
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -e .
```

### Run Full Pipeline

```bash
# Stage 1: Generate data
python src/data_generator.py

# Stage 2: SQL pipeline
python src/run_sql_pipeline.py

# Stage 3: EDA
python src/eda.py
```

**Outputs:**
- Raw data: `data/raw/*.parquet`
- Analytics table: `data/analytics/churn_features.parquet`
- Visualizations: `eda_outputs/*.html`, `eda_outputs/*.png`
- Insights: `eda_outputs/key_insights.txt`

---

## Documentation

For detailed documentation on each stage, see:

- **[`src/README.md`](src/README.md)** – Python modules, data generation, EDA, SQL pipeline orchestrator
- **[`data/README.md`](data/README.md)** – Data folder structure, raw/staging/analytics layout
- **[`sql/README.md`](sql/README.md)** – SQL pipeline details, staging, data quality, feature engineering
- **[`notebooks/eda_exploration.ipynb`](notebooks/eda_exploration.ipynb)** – Interactive EDA notebook

---

## Project Structure

```
churn-retention-analytics/
├── src/                    # Python modules
│   ├── data_generator.py  # Stage 1: Synthetic data generation
│   ├── run_sql_pipeline.py # Stage 2: SQL pipeline orchestrator
│   ├── eda.py              # Stage 3: Exploratory data analysis
│   └── README.md           # Python modules documentation
├── sql/                    # SQL transformation pipeline
│   ├── staging.sql         # Load raw data into staging
│   ├── data_quality.sql    # Data quality checks
│   ├── churn_features.sql  # Feature engineering
│   └── README.md           # SQL pipeline documentation
├── data/                   # Data storage
│   ├── raw/                # Stage 1 outputs
│   ├── staging/            # Staging tables (if exported)
│   ├── analytics/          # Final feature tables
│   └── README.md           # Data documentation
├── notebooks/              # Interactive exploration
│   └── eda_exploration.ipynb
├── eda_outputs/            # EDA visualizations and insights
├── app/                    # Streamlit application
└── README.md               # This file
```

---

## License

This project is provided as-is for portfolio and educational purposes.
