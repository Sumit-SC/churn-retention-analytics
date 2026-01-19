# Stage 5: Retention Strategy Simulator

## Overview

Stage 5 delivers a Streamlit application that uses the trained Random Forest model to simulate retention strategies and estimate business impact. The simulator enables data-driven decision-making by combining ML predictions with heuristic uplift assumptions.

---

## Objectives

- Turn churn predictions into actionable business decisions
- Simulate retention levers (discounts, priority support, feature unlocks)
- Estimate costs vs. retained revenue
- Calculate ROI for retention strategies
- Provide interactive risk scoring and targeting

---

## Application Structure

**File:** `app/retention_simulator.py`

### Sections

1. **Overview** — High-level metrics (total customers, average risk, actual churn rate)
2. **Risk Scoring** — Distribution visualization and top-risk customer table
3. **Retention Strategy Simulator** — Interactive controls and targeting
4. **Business Summary** — Before/after metrics, costs, ROI

---

## Risk Scoring

### Metrics Displayed
- Total customers
- Average churn risk score
- Percentage of customers above risk thresholds:
  - Medium risk (>0.3)
  - High risk (>0.6)

### Visualizations
- **Histogram:** Churn risk score distribution
- **Table:** Top 10 highest-risk customers (sample)

---

## Retention Strategy Simulator

### Sidebar Controls

**Targeting:**
- **Risk Threshold Slider:** Default 0.6 (target customers above this score)
- **Max Customers to Target:** Default 10% of base

**Retention Levers:**
- **Discount (%):** 5–30% (checkbox + slider)
- **Priority Support:** Boolean toggle
- **Feature Unlock:** Boolean toggle

---

## Simulation Logic (Heuristic)

**Base churn probability** = `churn_risk_score` (from Random Forest model)

### Uplift Assumptions

**If Discount Applied:**
- Reduce churn risk by `(discount % × 0.5%)`
- Example: 15% discount → 7.5% risk reduction

**If Priority Support Applied:**
- Reduce churn risk by 10%

**If Feature Unlock Applied:**
- Reduce churn risk by 7%

**Minimum Cap:** Churn probability capped at 0.05 (5% minimum)

### Calculations

- **Expected churn before:** Sum of base churn probabilities
- **Expected churn after:** Sum of adjusted churn probabilities
- **Customers retained:** Difference between before and after
- **Retained revenue proxy:** Customers retained × average monthly revenue

---

## Cost & ROI Estimation

### Revenue Assumptions (Monthly per Customer)

- **Free:** $0
- **Basic:** $20
- **Pro:** $50
- **Enterprise:** $200

### Cost Calculations

- **Discount cost:** `total_revenue × discount %`
- **Priority support cost:** `$5 per customer` (fixed)
- **Feature unlock cost:** `$3 per customer` (fixed)

### ROI Metrics

- **Total retention cost:** Sum of all intervention costs
- **Retained revenue:** Customers saved × average monthly revenue
- **Net ROI:** `retained_revenue − total_cost`

---

## Visualizations

### Before vs After Churn Rate
- Bar chart comparing churn rates before and after intervention

### Cost vs Retained Revenue
- Bar chart showing total cost vs retained revenue

---

## Business Disclaimers

The application includes clear disclaimers:

- **Model predictions** are probabilities, not certainties
- **Uplift assumptions** are illustrative heuristics for scenario comparison
- **Intended use:** Relative impact assessment, not exact financial forecasting
- **Revenue estimates** based on average monthly revenue by plan tier

---

## How to Run

```bash
streamlit run app/retention_simulator.py
```

**Runtime:** < 3 seconds load time

**Prerequisites:**
- Stage 4 completed (models trained and saved)
- Stage 2 completed (`analytics.churn_features` table exists)

---

## Use Cases

- **Scenario Planning:** Compare different retention strategies
- **ROI Optimization:** Identify cost-effective intervention combinations
- **Risk Prioritization:** Focus retention efforts on highest-risk segments
- **Portfolio Demonstration:** Showcase end-to-end analytics workflow

---

## Technical Notes

- Uses Random Forest model exclusively (no model selection toggle)
- Preprocessing pipeline recreated inline to match training
- Feature extensions (binary flags) created dynamically
- Caching enabled for data loading (`@st.cache_data`)

---

## Next Steps

The simulator completes the end-to-end workflow. For production deployment, consider:
- A/B testing framework for uplift validation
- Integration with CRM systems for automated targeting
- Real-time risk scoring API
- Advanced cost modeling with customer lifetime value (CLV)
