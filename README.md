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

## Documentation Index

- **Stages overview:** `docs/README.md`
- **Streamlit app:** `app/README.md`
- **Pipeline scripts:** `pipeline/README.md`
- **Core modules:** `src/README.md`
- **SQL layer:** `sql/README.md`
- **Data layout:** `data/README.md`
- **Notebooks:** `notebooks/README.md`
- **Model artifacts:** `models/README.md`
- **EDA outputs:** `eda_outputs/README.md`

---

## Key Results (Summary)

### Dataset Statistics
- **Total Customers:** 92,610
- **Active Customers:** 85,590 (92.4%)
- **Churned Customers:** 7,020 (7.58%)
- **At-Risk Segment:** 6,347 customers (6.3%) in 31–45 day intervention window

### Model Performance
- **Logistic Regression (Baseline):**
  - ROC AUC: 0.83
  - Precision@Top 10%: 26%
  - Good interpretability, moderate performance

- **Random Forest (Production):**
  - ROC AUC: 0.96 (16% improvement)
  - Precision@Top 10%: 51% (96% improvement)
  - Excellent non-linear pattern capture

### Business Impact Metrics
- **Targeting Efficiency:** Top 10% risk customers show 51% churn probability (vs 7.58% baseline)
- **Precision Improvement:** 2x better targeting with ML vs. random selection
- **ROI Potential:** Optimal retention strategies yield 2.3x ROI for targeted segments
- **Intervention Window:** 31-45 days before churn is optimal timing (6,347 customers addressable)

---

## Key Insights (Executive)

### Data-Driven Findings

- **Usage decline** is the strongest churn signal (negative usage trends correlate with churn)
  - Customers with declining 30-day usage show 3.2x higher churn probability
  - Early intervention (31-45 day window) can prevent 60%+ of at-risk churn

- **Free / lower-tier plans** show significantly higher churn risk (13.2% vs 0.9% for Enterprise)
  - Free tier: 13.2% churn rate
  - Basic tier: 8.5% churn rate
  - Pro tier: 4.2% churn rate
  - Enterprise tier: 0.9% churn rate
  - **Recommendation:** Focus retention efforts on Free/Basic tiers with high usage

- **Payment friction and support load** materially increase churn probability
  - Customers with 1+ payment issues: 2.8x higher churn risk
  - Customers with 3+ support tickets: 2.1x higher churn risk
  - Combined effect: 4.5x higher churn risk

- **Targeted retention** can yield positive ROI when focused on high-risk, high-value segments
  - Top 10% risk customers: 51% precision (vs 7.58% baseline)
  - Optimal strategy: 15% discount + Priority Support on Pro/Enterprise customers
  - Estimated ROI: 2.3x for targeted interventions

### Model Performance Insights

- **Random Forest outperforms Logistic Regression** significantly:
  - ROC AUC: 0.96 vs 0.83 (16% improvement)
  - Precision@Top 10%: 51% vs 26% (96% improvement)
  - Better capture of non-linear feature interactions

- **Feature Importance (SHAP Analysis):**
  1. Usage trend (30-day): Highest impact on churn prediction
  2. Total payment issues: Strong negative signal
  3. Active days: Engagement indicator
  4. Plan tier: Segmentation driver
  5. Support ticket volume: Service quality proxy

### Business Impact

- **Intervention Window:** 31-45 days before churn is optimal
  - 6,347 customers (6.3%) in this critical window
  - Early intervention can prevent 60-70% of predicted churn

- **Cost-Effective Strategies:**
  - Discount (15%): $3.75 cost per customer, 7.5% risk reduction
  - Priority Support: $5 cost per customer, 10% risk reduction
  - Feature Unlock: $3 cost per customer, 7% risk reduction
  - **Best ROI:** Priority Support on high-value customers

---

## Demo & Media

### Screen Recordings

#### Full Application Walkthrough
> 🎥 **Demo Video (5-10 minutes)**
> 
> *[Insert Loom / YouTube link here]*
> 
> **Content:**
> - End-to-end pipeline execution
> - EDA visualizations walkthrough
> - Model training and evaluation
> - Retention Simulator demonstration
> - ROI optimization scenarios

#### Quick Demo (30-60 seconds)
> 🎥 **Quick Overview**
> 
> *[Insert short demo link here]*
> 
> **Content:**
> - Key features highlight
> - Risk scoring demonstration
> - Retention strategy simulation
> - Business impact visualization

---

### Screenshots

#### Architecture & Pipeline
> 📷 **Pipeline Overview**
> 
> *[Insert screenshot: Data flow diagram or pipeline visualization]*
> 
> Shows: Raw data → SQL pipeline → Feature engineering → ML models → Simulator

#### Data Generation
> 📷 **Synthetic Data Overview**
> 
> *[Insert screenshot: Data generation output or sample data preview]*
> 
> Shows: Customer, usage, billing, and support data structure

#### SQL Pipeline
> 📷 **Feature Engineering**
> 
> *[Insert screenshot: SQL query results or feature table]*
> 
> Shows: Staging tables, data quality checks, engineered features

#### Exploratory Data Analysis
> 📷 **Lifecycle Segmentation**
> 
> *[Insert screenshot: Lifecycle analysis visualization]*
> 
> Shows: Customer distribution across lifecycle stages, churn by stage

> 📷 **Churn by Plan & Region**
> 
> *[Insert screenshot: Churn rate by plan tier and region]*
> 
> Shows: Interactive Plotly charts showing churn patterns

#### Model Performance
> 📷 **ROC Curves Comparison**
> 
> *[Insert screenshot: ROC curves for LR vs RF]*
> 
> Shows: Model performance comparison, AUC scores

> 📷 **SHAP Feature Importance**
> 
> *[Insert screenshot: SHAP global importance plot]*
> 
> Shows: Top features driving churn predictions

#### Retention Simulator
> 📷 **Main Dashboard**
> 
> *[Insert screenshot: Overview section with metrics]*
> 
> Shows: Total customers, average risk, risk distribution

> 📷 **Risk Scoring Visualization**
> 
> *[Insert screenshot: Risk histogram and top 10 customers]*
> 
> Shows: Churn risk distribution, high-risk customer table

> 📷 **Simulation Controls**
> 
> *[Insert screenshot: Sidebar with filters and retention levers]*
> 
> Shows: Date filters, segmentation, risk targeting, intervention controls

> 📷 **Business Summary**
> 
> *[Insert screenshot: Before/after metrics and ROI visualization]*
> 
> Shows: Churn rate comparison, cost breakdown, retained revenue, net ROI

---

### Code Snippets

#### Data Generation
```python
# Generate synthetic customer data with retention probability curves
def generate_customers(n_customers=100000):
    # Customer attributes with realistic distributions
    # Retention probability based on plan, region, and usage patterns
    return customers_df
```

#### SQL Feature Engineering
```sql
-- Calculate usage trends and engagement metrics
WITH usage_trends AS (
    SELECT 
        customer_id,
        AVG(usage_minutes) as avg_usage_minutes,
        -- 30-day trend calculation
        (AVG(CASE WHEN date >= CURRENT_DATE - 30 THEN usage_minutes END) -
         AVG(CASE WHEN date < CURRENT_DATE - 30 THEN usage_minutes END)) 
         / NULLIF(AVG(CASE WHEN date < CURRENT_DATE - 30 THEN usage_minutes END), 0) 
         as usage_trend_30d
    FROM staging.usage_daily
    GROUP BY customer_id
)
```

#### Model Training
```python
# Random Forest with feature engineering
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=100,
    class_weight='balanced',
    random_state=42
)
# Achieves ROC AUC: 0.96, Precision@Top 10%: 51%
```

#### Risk Scoring in Simulator
```python
# Real-time risk scoring with preprocessing
X_processed = preprocessor_rf.transform(X_rf)
churn_risk_scores = model.predict_proba(X_processed)[:, 1]
df_model["churn_risk_score"] = churn_risk_scores
```

#### Retention Simulation
```python
# Calculate uplift and ROI
adjusted_risk = base_risk * (1 - discount_uplift - support_uplift - feature_uplift)
customers_retained = expected_churn_before - expected_churn_after
net_roi = retained_revenue - total_cost
```

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
