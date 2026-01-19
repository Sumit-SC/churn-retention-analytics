# Churn & Retention Analytics System

## Executive Summary

This project delivers an **end-to-end churn and retention analytics system** that transforms raw customer data into actionable retention strategies. The system processes 100,000+ customer records and millions of usage events through a production-grade SQL pipeline, identifies at-risk segments via lifecycle analysis, and enables data-driven retention decisions through ML-powered risk scoring and a business simulator. The outcome is a complete workflow from data generation to ROI-optimized retention interventions.

---

## Project Objective

Customer churn is a critical business problem across SaaS, fintech, telecom, and subscription industries. This system enables decision-makers to:

- **Identify at-risk customers** before they churn (31–45 day intervention window)
- **Prioritize retention efforts** using ML-powered risk scores
- **Simulate retention strategies** with cost vs. impact analysis
- **Optimize ROI** by targeting high-value, high-risk segments

The system demonstrates production-ready analytics workflows suitable for portfolio demonstration and technical interviews.

---

## High-Level Architecture

```
Raw Data → SQL Feature Engineering → EDA & Lifecycle Analysis → ML Models → Retention Simulator
```

> 📌 *[Insert architecture diagram here]*

**Tech Stack:** Python 3.11+, DuckDB, Pandas, Scikit-learn, Streamlit, Plotly, Seaborn

---

## Project Stages (Wiki Navigation)

For detailed technical documentation, see stage-specific guides:

- **[Stage 1 — Data Generation & Modeling](docs/stage_1_data.md)**  
  Synthetic data generation with retention probability curves

- **[Stage 2 — SQL Pipelines & Data Quality](docs/stage_2_sql.md)**  
  Staging, data quality checks, and feature engineering

- **[Stage 3 — EDA & Lifecycle Analysis](docs/stage_3_eda.md)**  
  Exploratory analysis with lifecycle segmentation

- **[Stage 4 — ML-lite Modeling & Explainability](docs/stage_4_modeling.md)**  
  Logistic Regression and Random Forest with SHAP

- **[Stage 5 — Retention Strategy Simulator](docs/stage_5_simulator.md)**  
  Streamlit app for retention strategy simulation

---

## Key Results (Summary)

- **Churn Rate:** 7.58% (7,020 churned / 85,590 active)
- **At-Risk Segment:** 6.3% of customers (6,347) in 31–45 day window
- **Model Performance:**
  - Logistic Regression: ROC AUC 0.83, Precision@Top 10% 26%
  - Random Forest: ROC AUC 0.96, Precision@Top 10% 51%
- **Business Impact:** Targeting top 10% risk customers enables 2x precision improvement

---

## Key Insights (Executive)

- **Usage decline** is the strongest churn signal (negative usage trends correlate with churn)
- **Free / lower-tier plans** show significantly higher churn risk (13.2% vs 0.9% for Enterprise)
- **Payment friction and support load** materially increase churn probability
- **Targeted retention** can yield positive ROI when focused on high-risk, high-value segments

---

## Demo & Media

### Demo Video
> 🎥 *[Insert Loom / YouTube demo link here]*

### Screenshots
> 📷 *[Insert key screenshots here: EDA visualizations, SHAP importance, Retention Simulator UI]*

---

## How to Run (Reproducibility)

### Environment Setup (uv — preferred)
```bash
uv venv
uv sync
```

### Environment Setup (pip)
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -e .
```

### Full Pipeline Execution
```bash
# Stage 1: Generate synthetic data
python pipeline/data_generator.py

# Stage 2: SQL pipeline (staging → features)
python pipeline/run_sql_pipeline.py

# Stage 3: Exploratory data analysis
python src/eda.py

# Stage 4: Train models
python pipeline/train_model.py

# Stage 5: Launch retention simulator
streamlit run app/retention_simulator.py
```

**Expected Runtime:** < 5 minutes for full pipeline

---

## Project Structure

```
churn-retention-analytics/
├── pipeline/               # Pipeline execution scripts
│   ├── data_generator.py  # Stage 1: Generate synthetic data
│   ├── run_sql_pipeline.py # Stage 2: SQL transformation pipeline
│   └── train_model.py     # Stage 4: Train ML models
├── src/                    # Python modules & analysis
│   ├── eda.py              # Stage 3: Exploratory data analysis
│   └── README.md
├── sql/                    # SQL transformation pipeline
│   ├── staging.sql
│   ├── data_quality.sql
│   ├── churn_features.sql
│   └── README.md
├── app/                    # Streamlit applications
│   └── retention_simulator.py
├── docs/                   # Stage-level documentation
│   ├── stage_1_data.md
│   ├── stage_2_sql.md
│   ├── stage_3_eda.md
│   ├── stage_4_modeling.md
│   └── stage_5_simulator.md
├── data/                   # Data storage
│   ├── raw/
│   └── analytics/
├── models/                 # ML artifacts
├── eda_outputs/            # EDA visualizations
├── notebooks/             # Interactive exploration
└── README.md              # This file
```

---

## License

This project is provided as-is for portfolio and educational purposes.
