# Streamlit Application: Retention Strategy Simulator

## Overview

The Retention Strategy Simulator is an interactive Streamlit web application that transforms ML-powered churn predictions into actionable business decisions. It enables data-driven retention strategy planning through real-time risk scoring, interactive targeting controls, and ROI-optimized intervention simulation.

---

## Features

### 🎯 Interactive Risk Scoring
- Real-time churn risk distribution visualization
- Top 10 highest-risk customers identification
- Dynamic risk threshold targeting
- Customer segmentation by plan, region, and activity dates

### 💰 Retention Strategy Simulation
- **Discount Intervention:** 5-30% discount slider with cost calculation
- **Priority Support:** Toggle with fixed cost per customer
- **Feature Unlock:** Toggle with fixed cost per customer
- Multi-lever combination support

### 📊 Business Impact Analysis
- Before/after churn rate comparison
- Cost vs. retained revenue visualization
- Net ROI calculation
- Customer retention metrics

### 🎨 Professional UI/UX
- Clean, modern interface with custom styling
- Responsive sidebar with scrollable controls
- Interactive Plotly visualizations
- Real-time filter summary updates

---

## Quick Start

### Prerequisites
- Python 3.11+
- Completed data pipeline (Stage 1-4)
- Trained Random Forest model (`models/rf_model.joblib`)
- DuckDB database (`churn.duckdb`)

### Installation

```bash
# Install dependencies (if not already done)
uv sync
# or
pip install -e .
```

### Running the Application

```bash
streamlit run app/retention_simulator.py
```

The application will automatically open in your default browser at `http://localhost:8501`

---

## Application Structure

### Main Sections

1. **Overview Dashboard**
   - Total customers count
   - Average churn risk score
   - Risk distribution metrics (Medium/High risk percentages)

2. **Risk Scoring**
   - Interactive histogram of churn risk distribution
   - Top 10 highest-risk customers table
   - Mean risk and threshold indicators

3. **Retention Strategy Simulator**
   - Sidebar controls for filtering and targeting
   - Real-time simulation results
   - Before/after metrics comparison

4. **Business Summary**
   - Intervention costs breakdown
   - Retained revenue estimation
   - Net ROI calculation
   - Cost vs. retained revenue visualization

### Sidebar Controls

#### Date Filters
- **Signup Date Range:** Filter customers by signup period
- **Last Active Date Range:** Filter by recent activity

#### Customer Segmentation
- **Plan Filter:** Multi-select by subscription tier
- **Region Filter:** Multi-select by geographic region

#### Risk Targeting
- **Risk Threshold Slider:** Dynamic range based on filtered data
- **Max Customers to Target:** Percentage-based targeting limit

#### Retention Levers
- **Discount:** 5-30% slider with cost calculation
- **Priority Support:** Toggle ($5 per customer)
- **Feature Unlock:** Toggle ($3 per customer)

---

## Technical Details

### Model Integration
- Uses Random Forest classifier (trained in Stage 4)
- Real-time preprocessing pipeline matching training
- Feature engineering (binary flags) computed on-the-fly
- Cached data loading for performance (`@st.cache_data`)

### Data Flow
```
DuckDB Database → Feature Engineering → Model Prediction → Risk Scoring → Simulation
```

### Performance
- **Load Time:** < 3 seconds
- **Filter Response:** Real-time (< 1 second)
- **Visualization Rendering:** Instant with Plotly

---

## Error Handling

The application includes comprehensive error handling:

- **Missing Database:** Clear error message with solution steps
- **Missing Model:** File path and resolution guidance
- **Database Connection Errors:** Detailed exception messages
- **Empty Filter Results:** User-friendly warnings and suggestions

---

## Customization

### Styling
Custom CSS is embedded in the application for:
- Hiding Streamlit default footer and deploy button
- Custom footer with copyright information
- Improved content padding and spacing
- Scrollable expander content

### Configuration
Key constants can be modified in the code:
- `PLAN_REVENUE`: Monthly revenue by plan tier
- `PRIORITY_SUPPORT_COST`: Fixed cost per customer
- `FEATURE_UNLOCK_COST`: Fixed cost per customer
- Uplift assumptions (discount, support, feature unlock percentages)

---

## Use Cases

### Scenario Planning
Compare different retention strategies side-by-side:
- Test discount percentages (5%, 10%, 15%, 20%, 30%)
- Combine multiple levers (discount + priority support)
- Evaluate ROI across different customer segments

### ROI Optimization
Identify cost-effective intervention combinations:
- Target high-risk, high-value customers
- Balance intervention costs with retention impact
- Optimize for maximum net ROI

### Risk Prioritization
Focus retention efforts efficiently:
- Identify top 10% highest-risk customers
- Filter by plan tier or region
- Adjust targeting based on business constraints

---

## Screenshots & Media

### Application Screenshots

#### Main Dashboard
> 📷 *[Insert screenshot: Overview dashboard with metrics and risk distribution]*

#### Risk Scoring Section
> 📷 *[Insert screenshot: Risk histogram and top 10 high-risk customers table]*

#### Retention Simulator
> 📷 *[Insert screenshot: Sidebar controls and simulation results]*

#### Business Summary
> 📷 *[Insert screenshot: Before/after metrics, cost breakdown, and ROI visualization]*

### Screen Recording

#### Full Application Walkthrough
> 🎥 *[Insert screen recording: Complete demo showing filtering, targeting, and simulation]*

#### Quick Demo (30 seconds)
> 🎥 *[Insert screen recording: Quick overview of key features]*

---

## Code Snippets

### Key Function: Data Loading
```python
@st.cache_data
def load_data_and_model():
    """Load data and trained model."""
    conn = duckdb.connect(str(DB_PATH))
    df = conn.execute("SELECT * FROM analytics.churn_features").df()
    # ... feature engineering ...
    model = joblib.load(MODELS_DIR / "rf_model.joblib")
    return df_model, model, preprocessor_rf
```

### Key Function: Risk Scoring
```python
X_processed = preprocessor_rf.transform(X_rf)
churn_risk_scores = model.predict_proba(X_processed)[:, 1]
df_model["churn_risk_score"] = churn_risk_scores
```

### Key Function: Simulation Logic
```python
def simulate_retention(df, risk_threshold, max_target_pct, 
                      apply_discount, discount_pct,
                      apply_priority_support, apply_feature_unlock):
    # Filter and target customers
    # Calculate uplift and adjusted churn probabilities
    # Compute costs and retained revenue
    return results_dict
```

---

## Troubleshooting

### Common Issues

**Issue:** "Database not found" error
- **Solution:** Ensure `churn.duckdb` exists in project root. Run `pipeline/run_sql_pipeline.py` first.

**Issue:** "Model file not found" error
- **Solution:** Ensure `models/rf_model.joblib` exists. Run `pipeline/train_model.py` first.

**Issue:** No customers match filters
- **Solution:** Adjust date ranges, plan/region filters, or lower the risk threshold.

**Issue:** Slider range errors
- **Solution:** The application automatically adjusts slider ranges based on filtered data. Ensure filters return at least some customers.

---

## Deployment

### Local Development
```bash
streamlit run app/retention_simulator.py
```

### Streamlit Cloud
1. Push code to GitHub repository
2. Connect repository to Streamlit Cloud
3. Set Python version to 3.11+
4. Deploy with default settings

### Docker (Optional)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e .
EXPOSE 8501
CMD ["streamlit", "run", "app/retention_simulator.py", "--server.port=8501"]
```

---

## Future Enhancements

- [ ] A/B testing framework integration
- [ ] Real-time API for risk scoring
- [ ] Advanced cost modeling with CLV
- [ ] Export simulation results to CSV/PDF
- [ ] Multi-user session support
- [ ] Historical simulation tracking

---

## License

This application is part of the Churn & Retention Analytics project. See main README for license information.
