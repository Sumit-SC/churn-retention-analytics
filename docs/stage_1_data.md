# Stage 1: Data Generation & Modeling

## Overview

Stage 1 generates realistic synthetic SaaS customer data with retention probability curves. The data generator creates four core tables: customers, daily usage, billing, and support tickets, simulating 18 months of customer behavior.

---

## Objectives

- Generate 100,000+ customers with realistic attributes
- Simulate usage patterns with retention decay over time
- Create billing events for paid plans with payment status modeling
- Generate support tickets correlated with customer tenure
- Output Parquet files for downstream SQL processing

---

## Implementation

**File:** `src/data_generator.py`

### Data Generation Logic

#### Customers Table
- **100,000 customers** with unique IDs
- **Signup dates** distributed over 18-month window
- **Plan distribution:** Free (40%), Basic (35%), Pro (20%), Enterprise (5%)
- **Region distribution:** NA (35%), EU (30%), APAC (25%), LATAM (10%)

#### Usage Daily Table
- **Per-customer, per-day usage** with retention probability decay
- **Retention curves** by plan tier:
  - Free: 22% base retention
  - Basic: 35% base retention
  - Pro: 50% base retention
  - Enterprise: 70% base retention
- **Decay factor:** Exponential decay based on days since signup
- **Usage metrics:**
  - `sessions`: Poisson-distributed by plan
  - `usage_minutes`: Normal distribution (capped at 3σ for realism)
  - `feature_events`: Correlated with sessions
- **Missing logs:** 12–18% random drop to simulate tracking gaps

#### Billing Table
- **Only for paid plans** (Basic, Pro, Enterprise)
- **Billing frequency:** ~13-day intervals
- **Payment status:** Probabilistic (75% paid, 15% late, 10% failed)
- **Amounts:** Based on plan pricing

#### Support Table
- **Ticket generation:** Poisson-distributed, scaled by customer tenure
- **Categories:** Billing (25%), Technical (60%), Onboarding (15%)
- **Priority:** Low (55%), Medium (40%), High (5%)

---

## Outputs

All files saved to `data/raw/`:

- `customers.parquet` (~100K rows)
- `usage_daily.parquet` (~4M rows)
- `billing.parquet` (~1.2M rows)
- `support.parquet` (~380K rows)

---

## How to Run

```bash
python src/data_generator.py
```

**Runtime:** ~10–15 seconds

**Console Output:**
- Row counts for each table
- Total runtime

---

## Data Quality Considerations

- **Outlier handling:** Usage minutes capped at 3 standard deviations
- **Missing data:** Simulated via random log drops
- **Temporal consistency:** All dates within 18-month window
- **Referential integrity:** All foreign keys valid

---

## Next Steps

After Stage 1, proceed to **[Stage 2: SQL Pipelines & Data Quality](stage_2_sql.md)** for data transformation and feature engineering.
