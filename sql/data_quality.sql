-- ============================================================
-- DATA QUALITY CHECKS (DuckDB)
-- ------------------------------------------------------------
-- Purpose:
-- - Create `dq` schema for data quality results
-- - Run checks on staging tables and output issue counts
-- - Do NOT clean data, only report issues
-- ============================================================

-- Create schema if not exists
CREATE SCHEMA IF NOT EXISTS dq;

-- ============================================================
-- USAGE_DAILY QUALITY CHECKS
-- ============================================================

-- Check 1: Negative usage_minutes (should never happen)
CREATE OR REPLACE TABLE dq.usage_daily_negative_minutes AS
SELECT
    COUNT(*) AS issue_count,
    'usage_minutes < 0' AS check_name,
    'usage_daily' AS table_name
FROM staging.usage_daily
WHERE usage_minutes < 0;

-- Check 2: Usage minutes above 99.9th percentile (potential outliers)
CREATE OR REPLACE TABLE dq.usage_daily_extreme_minutes AS
WITH percentile_calc AS (
    SELECT
        PERCENTILE_CONT(0.999) WITHIN GROUP (ORDER BY usage_minutes) AS p999_minutes
    FROM staging.usage_daily
)
SELECT
    COUNT(*) AS issue_count,
    'usage_minutes > P99.9' AS check_name,
    'usage_daily' AS table_name
FROM staging.usage_daily, percentile_calc
WHERE usage_minutes > p999_minutes;

-- Check 3: Sessions = 0 but usage_minutes > 0 (logical inconsistency)
CREATE OR REPLACE TABLE dq.usage_daily_session_mismatch AS
SELECT
    COUNT(*) AS issue_count,
    'sessions = 0 AND usage_minutes > 0' AS check_name,
    'usage_daily' AS table_name
FROM staging.usage_daily
WHERE sessions = 0 AND usage_minutes > 0;

-- ============================================================
-- BILLING QUALITY CHECKS
-- ============================================================

-- Check 4: Paid bills with zero amount (should be free plan or error)
CREATE OR REPLACE TABLE dq.billing_zero_paid AS
SELECT
    COUNT(*) AS issue_count,
    'amount = 0 AND payment_status = paid' AS check_name,
    'billing' AS table_name
FROM staging.billing
WHERE amount = 0 AND payment_status = 'paid';

-- Check 5: Duplicate (customer_id, bill_date) pairs
CREATE OR REPLACE TABLE dq.billing_duplicates AS
WITH duplicate_counts AS (
    SELECT
        customer_id,
        bill_date,
        COUNT(*) AS duplicate_count
    FROM staging.billing
    GROUP BY customer_id, bill_date
    HAVING COUNT(*) > 1
)
SELECT
    SUM(duplicate_count - 1) AS issue_count,
    'duplicate (customer_id, bill_date)' AS check_name,
    'billing' AS table_name
FROM duplicate_counts;

-- ============================================================
-- SUPPORT QUALITY CHECKS
-- ============================================================

-- Check 6: Tickets created before customer signup (temporal inconsistency)
CREATE OR REPLACE TABLE dq.support_pre_signup_tickets AS
SELECT
    COUNT(*) AS issue_count,
    'ticket_date < signup_date' AS check_name,
    'support' AS table_name
FROM staging.support AS s
INNER JOIN staging.customers AS c
    ON s.customer_id = c.customer_id
WHERE s.ticket_date < c.signup_date;

-- ============================================================
-- SUMMARY TABLE: All data quality issues
-- ============================================================

CREATE OR REPLACE TABLE dq.data_quality_summary AS
SELECT * FROM dq.usage_daily_negative_minutes
UNION ALL
SELECT * FROM dq.usage_daily_extreme_minutes
UNION ALL
SELECT * FROM dq.usage_daily_session_mismatch
UNION ALL
SELECT * FROM dq.billing_zero_paid
UNION ALL
SELECT * FROM dq.billing_duplicates
UNION ALL
SELECT * FROM dq.support_pre_signup_tickets
ORDER BY table_name, check_name;
