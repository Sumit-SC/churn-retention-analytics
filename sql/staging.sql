-- ============================================================
-- STAGE 2: STAGING LAYER (DuckDB)
-- ------------------------------------------------------------
-- Purpose:
-- - Recreate the `staging` schema
-- - Materialize raw Parquet sources into DuckDB staging tables
--
-- Non-goals (by design):
-- - No aggregation
-- - No churn definition / churn computation
-- - No feature engineering
-- ============================================================

-- Recreate schema to keep runs deterministic
DROP SCHEMA IF EXISTS staging CASCADE;
CREATE SCHEMA staging;

-- ------------------------------------------------------------
-- staging.customers
-- Source: data/raw/customers.parquet
-- ------------------------------------------------------------
CREATE TABLE staging.customers AS
SELECT
  CAST(customer_id AS BIGINT)    AS customer_id,
  CAST(signup_date AS TIMESTAMP) AS signup_date,
  CAST(plan AS VARCHAR)          AS plan,
  CAST(region AS VARCHAR)        AS region
FROM read_parquet('data/raw/customers.parquet');

-- ------------------------------------------------------------
-- staging.usage_daily
-- Source: data/raw/usage_daily.parquet
-- ------------------------------------------------------------
CREATE TABLE staging.usage_daily AS
SELECT
  CAST(customer_id AS BIGINT)        AS customer_id,
  CAST(date AS DATE)                 AS date,
  CAST(sessions AS INTEGER)          AS sessions,
  CAST(usage_minutes AS INTEGER)     AS usage_minutes,
  CAST(feature_events AS INTEGER)    AS feature_events
FROM read_parquet('data/raw/usage_daily.parquet');

-- ------------------------------------------------------------
-- staging.billing
-- Source: data/raw/billing.parquet
-- ------------------------------------------------------------
CREATE TABLE staging.billing AS
SELECT
  CAST(customer_id AS BIGINT)     AS customer_id,
  CAST(bill_date AS DATE)         AS bill_date,
  CAST(amount AS INTEGER)         AS amount,
  CAST(payment_status AS VARCHAR) AS payment_status
FROM read_parquet('data/raw/billing.parquet');

-- ------------------------------------------------------------
-- staging.support
-- Source: data/raw/support.parquet
-- ------------------------------------------------------------
CREATE TABLE staging.support AS
SELECT
  CAST(customer_id AS BIGINT)  AS customer_id,
  CAST(ticket_date AS DATE)    AS ticket_date,
  CAST(category AS VARCHAR)    AS category,
  CAST(priority AS VARCHAR)    AS priority
FROM read_parquet('data/raw/support.parquet');

