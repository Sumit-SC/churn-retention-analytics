# Stage 4: ML-lite Modeling & Explainability

## Overview

Stage 4 trains baseline and advanced churn prediction models using the cleaned feature set. The stage includes Logistic Regression (baseline) and Random Forest (champion) models, with SHAP explainability for business interpretation.

---

## Objectives

- Train explainable baseline model (Logistic Regression)
- Train non-linear model (Random Forest) to capture threshold effects
- Remove label leakage (exclude `recency_days` from features)
- Generate SHAP explanations for feature importance
- Compare model performance and business interpretability

---

## Feature Engineering

**File:** `pipeline/train_model.py`

### Base Features (Both Models)

**Numeric:**
- `active_days`
- `avg_sessions`
- `avg_usage_minutes`
- `usage_trend_30d`
- `total_payment_issues`
- `failed_payments_30d`
- `total_tickets`
- `high_priority_tickets`

**Categorical:**
- `plan`
- `region`

### Random Forest Extensions

**Binary Flags:**
- `usage_decline_flag`: 1 if `usage_trend_30d < 0`
- `high_support_flag`: 1 if `total_tickets >= 3`
- `payment_issue_flag`: 1 if `total_payment_issues >= 1`

**Note:** `recency_days` is excluded from modeling features to prevent label leakage (it's used to define `churn_label`).

---

## Data Preprocessing

### Outlier Handling
- **Winsorization** at 1st and 99th percentiles for:
  - `total_payment_issues`
  - `total_tickets`

### Missing Value Imputation
- **Median imputation** for numeric features

### Encoding & Scaling
- **One-hot encoding** for categorical features (`drop_first=True`)
- **StandardScaler** for numeric features (Logistic Regression)
- StandardScaler also applied for Random Forest (pipeline consistency)

---

## Models

### Logistic Regression (Baseline)

**Hyperparameters:**
- Solver: `liblinear`
- Class weight: `balanced`
- Max iterations: 1000

**Performance:**
- ROC AUC: **0.8336**
- Precision@Top 10%: **0.2605**

**Top Drivers:**
- **Positive:** `plan_Free`, `active_days`, `total_payment_issues`
- **Negative (Protective):** `plan_Enterprise`, `plan_Pro`, `avg_sessions`

### Random Forest (Champion)

**Hyperparameters:**
- `n_estimators`: 200
- `max_depth`: 8
- `min_samples_leaf`: 100
- `class_weight`: `balanced`
- `random_state`: 42

**Performance:**
- ROC AUC: **0.9590**
- Precision@Top 10%: **0.5063**

**Improvements:**
- ~15% higher ROC AUC vs Logistic Regression
- ~2x precision in top 10% risk segment

---

## SHAP Explainability

**TreeExplainer** used for Random Forest:

- **Global feature importance:** Mean absolute SHAP values
- **Visualization:** Bar chart of top features by SHAP importance
- **Output:** `eda_outputs/shap_global_importance.png`

**Key Insights:**
- Binary flags capture threshold effects (e.g., 3+ tickets = high risk)
- Plan tier remains strongest predictor
- RF discovers interaction effects between support load and payment issues

---

## Model Comparison

| Metric | Logistic Regression | Random Forest |
|--------|-------------------|---------------|
| ROC AUC | 0.8336 | 0.9590 |
| Precision@Top 10% | 0.2605 | 0.5063 |

**Business Impact:**
- Random Forest enables 2x precision improvement in top-risk segment
- Better risk stratification for retention campaigns

---

## How to Run

```bash
python pipeline/train_model.py
```

**Runtime:** < 60 seconds

**Prerequisites:**
- Stage 2 completed (`analytics.churn_features` table exists)

---

## Outputs

**Model Artifacts** (saved to `models/`):
- `logistic_model.joblib` — Trained Logistic Regression model
- `preprocessing.joblib` — Preprocessing pipeline (LR)
- `rf_model.joblib` — Trained Random Forest model

**Explainability:**
- `eda_outputs/shap_global_importance.png` — SHAP feature importance plot

**Console Output:**
- Model performance metrics
- Top 10 positive and negative churn drivers
- Business interpretation insights

---

## Next Steps

After Stage 4, proceed to **[Stage 5: Retention Strategy Simulator](stage_5_simulator.md)** for business application.
