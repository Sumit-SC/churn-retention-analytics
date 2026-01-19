"""
Data generator module for creating synthetic churn and retention data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time

start_time = time.time()

# Set random seeds for reproducibility
np.random.seed(42)

# Get project root (parent of pipeline folder)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Date ranges
END_DATE = datetime(2024, 12, 31)
START_DATE = END_DATE - timedelta(days=18 * 30)

# Plan configurations
PLAN_DISTRIBUTION = {"Free": 0.40, "Basic": 0.35, "Pro": 0.20, "Enterprise": 0.05}
PLAN_PRICES = {"Free": 0, "Basic": 29, "Pro": 99, "Enterprise": 299}

# Generate customers table
print("Generating customers...")
n_customers = 100000
customer_ids = np.arange(1, n_customers + 1)

signup_dates = pd.to_datetime([
    START_DATE + timedelta(days=int(np.random.uniform(0, 18 * 30)))
    for _ in range(n_customers)
])

plans = np.random.choice(
    list(PLAN_DISTRIBUTION.keys()),
    size=n_customers,
    p=list(PLAN_DISTRIBUTION.values())
)

regions = np.random.choice(
    ["NA", "EU", "APAC", "LATAM"],
    size=n_customers,
    p=[0.35, 0.30, 0.25, 0.10]
)

customers = pd.DataFrame({
    "customer_id": customer_ids,
    "signup_date": signup_dates,
    "plan": plans,
    "region": regions
})

# Generate usage_daily table (vectorized, chunked by 30-day windows)
print("Generating usage_daily...")
usage_chunks = []

# Plan-based usage parameters
plan_sessions = {"Free": 2, "Basic": 8, "Pro": 25, "Enterprise": 60}
plan_minutes = {"Free": 15, "Basic": 60, "Pro": 180, "Enterprise": 480}

# Generate date ranges for all customers
all_dates = pd.date_range(START_DATE, END_DATE, freq="D")
n_days = len(all_dates)

# Process in 30-day chunks for memory efficiency
for chunk_start in range(0, n_days, 30):
    chunk_end = min(chunk_start + 30, n_days)
    chunk_dates = all_dates[chunk_start:chunk_end]
    
    # Create cartesian product of customers and dates for this chunk
    customer_chunk = customers[["customer_id", "plan", "signup_date"]].copy()
    date_chunk = pd.DataFrame({"date": chunk_dates})
    
    # Cross join customers with dates
    customer_chunk["key"] = 1
    date_chunk["key"] = 1
    usage_chunk = customer_chunk.merge(date_chunk, on="key").drop("key", axis=1)
    
    # Filter: only dates on or after signup
    usage_chunk = usage_chunk[usage_chunk["date"] >= usage_chunk["signup_date"]]
    
    # Calculate days since signup for retention modeling
    days_since_signup = (usage_chunk["date"] - usage_chunk["signup_date"]).dt.days
    
    # Customer retention probability decreases over time (vectorized)
    # Base retention by plan (increased to reach ~4.1M rows)
    plan_retention = usage_chunk["plan"].map({"Free": 0.22, "Basic": 0.35, "Pro": 0.50, "Enterprise": 0.70})
    # Decay factor based on days since signup (slower decay)
    decay_factor = np.exp(-days_since_signup.values / 220.0)
    retention_prob = plan_retention.values * decay_factor
    
    # Filter by retention probability
    usage_chunk = usage_chunk[np.random.random(len(usage_chunk)) < retention_prob]
    
    # Randomly drop 12-18% of remaining rows (missing logs)
    drop_prob = np.random.uniform(0.12, 0.18, len(usage_chunk))
    usage_chunk = usage_chunk[np.random.random(len(usage_chunk)) > drop_prob]
    
    # Vectorized usage generation based on plan
    plan_sessions_vec = usage_chunk["plan"].map(plan_sessions)
    plan_minutes_vec = usage_chunk["plan"].map(plan_minutes)
    
    # Generate sessions (Poisson)
    usage_chunk["sessions"] = np.random.poisson(plan_sessions_vec.values)
    
    # Generate usage_minutes (truncated normal to prevent extreme outliers)
    usage_mean = plan_minutes_vec.values
    usage_std = usage_mean * 0.3
    usage_minutes_raw = np.random.normal(usage_mean, usage_std)
    # Cap at 3 standard deviations above mean for realistic bounds
    usage_max = usage_mean + (usage_std * 3)
    usage_chunk["usage_minutes"] = np.clip(
        usage_minutes_raw,
        a_min=0,
        a_max=usage_max
    ).astype(int)
    
    # Generate feature_events (correlated with sessions)
    usage_chunk["feature_events"] = np.random.poisson(usage_chunk["sessions"].values * 1.5)
    
    # Drop helper columns
    usage_chunk = usage_chunk[["customer_id", "date", "sessions", "usage_minutes", "feature_events"]]
    
    usage_chunks.append(usage_chunk)

# Combine all chunks
usage_daily = pd.concat(usage_chunks, ignore_index=True)
usage_daily = usage_daily.sort_values(["customer_id", "date"]).reset_index(drop=True)

# Generate billing table (fully vectorized)
print("Generating billing...")
paid_customers = customers[customers["plan"] != "Free"].copy()

if len(paid_customers) > 0:
    # Calculate months active for each customer
    months_active = ((END_DATE - paid_customers["signup_date"]).dt.days / 30).clip(upper=18).astype(int)
    
    # Generate bill counts (ensure high bill counts to reach ~1.23M target)
    # Use tenure-based calculation with high minimum and scaling
    bill_counts = np.maximum(months_active.values + 15, np.minimum(months_active.values * 4.0, 42)).astype(int)
    
    # Create expanded dataframe
    billing_expanded = paid_customers.loc[paid_customers.index.repeat(bill_counts)].reset_index(drop=True)
    billing_expanded["bill_sequence"] = billing_expanded.groupby("customer_id").cumcount()
    
    # Calculate bill dates (more frequent billing: every ~13 days to fit more bills)
    days_offset = 13 + billing_expanded["bill_sequence"] * 13 + np.random.randint(-2, 3, len(billing_expanded))
    billing_expanded["bill_date"] = billing_expanded["signup_date"] + pd.to_timedelta(days_offset, unit="D")
    
    # Filter bills after end date
    billing_expanded = billing_expanded[billing_expanded["bill_date"] <= END_DATE]
    
    # Generate amounts and payment status
    billing_expanded["amount"] = billing_expanded["plan"].map(PLAN_PRICES)
    status_probs = np.random.random(len(billing_expanded))
    billing_expanded["payment_status"] = np.where(
        status_probs < 0.75, "paid",
        np.where(status_probs < 0.90, "late", "failed")
    )
    
    billing = billing_expanded[["customer_id", "bill_date", "amount", "payment_status"]].copy()
    billing["bill_date"] = pd.to_datetime(billing["bill_date"])
else:
    billing = pd.DataFrame(columns=["customer_id", "bill_date", "amount", "payment_status"])

# Generate support table (fully vectorized)
print("Generating support...")
# Ticket generation probabilities per customer (scaled by tenure)
customer_tenure_months = ((END_DATE - customers["signup_date"]).dt.days / 30).clip(lower=1).clip(upper=18)
ticket_lambda = 0.42 * customer_tenure_months.values
tickets_per_customer = np.random.poisson(ticket_lambda)
tickets_per_customer = np.clip(tickets_per_customer, a_min=0, a_max=None).astype(int)

# Create expanded dataframe
support_expanded = customers.loc[customers.index.repeat(tickets_per_customer)].reset_index(drop=True)

if len(support_expanded) > 0:
    # Calculate days active for each ticket
    days_active = (END_DATE - support_expanded["signup_date"]).dt.days
    days_active = days_active.clip(lower=1)
    
    # Generate ticket day offsets (vectorized)
    ticket_day_offsets = np.random.randint(0, days_active.values.max() + 1, size=len(support_expanded))
    ticket_day_offsets = np.minimum(ticket_day_offsets, days_active.values - 1)
    support_expanded["ticket_date"] = support_expanded["signup_date"] + pd.to_timedelta(ticket_day_offsets, unit="D")
    
    # Filter tickets after end date
    support_expanded = support_expanded[support_expanded["ticket_date"] <= END_DATE]
    
    # Generate categories
    category_probs = np.random.random(len(support_expanded))
    support_expanded["category"] = np.where(
        category_probs < 0.25, "billing",
        np.where(category_probs < 0.85, "technical", "onboarding")
    )
    
    # Generate priorities (high is rarer)
    priority_probs = np.random.random(len(support_expanded))
    support_expanded["priority"] = np.where(
        priority_probs < 0.55, "low",
        np.where(priority_probs < 0.95, "medium", "high")
    )
    
    support = support_expanded[["customer_id", "ticket_date", "category", "priority"]].copy()
    support["ticket_date"] = pd.to_datetime(support["ticket_date"])
else:
    support = pd.DataFrame(columns=["customer_id", "ticket_date", "category", "priority"])

# Save all tables as Parquet
print("Saving Parquet files...")
customers.to_parquet(OUTPUT_DIR / "customers.parquet", index=False)
usage_daily.to_parquet(OUTPUT_DIR / "usage_daily.parquet", index=False)
billing.to_parquet(OUTPUT_DIR / "billing.parquet", index=False)
support.to_parquet(OUTPUT_DIR / "support.parquet", index=False)

# Print row counts and runtime
elapsed_time = time.time() - start_time
print(f"\nRow counts:")
print(f"customers: {len(customers):,}")
print(f"usage_daily: {len(usage_daily):,}")
print(f"billing: {len(billing):,}")
print(f"support: {len(support):,}")
print(f"\nRuntime: {elapsed_time:.2f} seconds")
