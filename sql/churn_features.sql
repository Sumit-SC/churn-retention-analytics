DROP SCHEMA IF EXISTS analytics CASCADE;
CREATE SCHEMA analytics;

CREATE TABLE analytics.churn_features AS
WITH
usage_percentiles AS (
    SELECT
        PERCENTILE_CONT(0.999) WITHIN GROUP (ORDER BY usage_minutes) AS p999_minutes
    FROM staging.usage_daily
),
usage_clean AS (
    SELECT
        u.customer_id,
        u.date,
        CASE
            WHEN u.usage_minutes < 0 THEN NULL
            WHEN u.usage_minutes > up.p999_minutes THEN CAST(up.p999_minutes AS INTEGER)
            ELSE u.usage_minutes
        END AS usage_minutes_clean,
        CASE
            WHEN u.sessions = 0 AND u.usage_minutes > 0 THEN 1
            ELSE u.sessions
        END AS sessions_clean,
        u.feature_events
    FROM staging.usage_daily AS u
    CROSS JOIN usage_percentiles AS up
),
usage_agg AS (
    SELECT
        customer_id,
        MAX(date)                        AS last_active_date,
        COUNT(DISTINCT date)             AS active_days,
        AVG(sessions_clean)              AS avg_sessions,
        AVG(usage_minutes_clean)         AS avg_usage_minutes
    FROM usage_clean
    GROUP BY customer_id
),
observation_date AS (
    SELECT MAX(date) AS obs_date
    FROM staging.usage_daily
),
max_date AS (
    SELECT MAX(date) AS max_usage_date
    FROM usage_clean
),
usage_trends AS (
    SELECT
        uc.customer_id,
        AVG(uc.usage_minutes_clean) FILTER (
            WHERE uc.date >= md.max_usage_date - INTERVAL '30 days'
        ) AS avg_usage_last_30d,
        AVG(uc.usage_minutes_clean) FILTER (
            WHERE uc.date >= md.max_usage_date - INTERVAL '60 days'
              AND uc.date <  md.max_usage_date - INTERVAL '30 days'
        ) AS avg_usage_prev_30d
    FROM usage_clean AS uc
    CROSS JOIN max_date AS md
    GROUP BY uc.customer_id
),
billing_clean AS (
    SELECT DISTINCT
        b.customer_id,
        b.bill_date,
        b.amount,
        b.payment_status,
        CASE
            WHEN b.amount = 0 AND b.payment_status = 'paid' THEN 1
            ELSE 0
        END AS is_zero_paid_flag
    FROM staging.billing AS b
),

billing_agg AS (
    SELECT
        bc.customer_id,
        SUM(
            CASE
                WHEN bc.payment_status IN ('late', 'failed') THEN 1
                ELSE 0
            END
        ) AS total_payment_issues,
        SUM(
            CASE
                WHEN bc.payment_status = 'failed'
                     AND bc.bill_date >= md.max_usage_date - INTERVAL '30 days'
                THEN 1
                ELSE 0
            END
        ) AS failed_payments_30d,
        SUM(bc.is_zero_paid_flag) AS zero_paid_anomalies
    FROM billing_clean AS bc
    CROSS JOIN max_date AS md
    GROUP BY bc.customer_id
),
support_clean AS (
    SELECT
        s.customer_id,
        s.ticket_date,
        s.category,
        s.priority
    FROM staging.support AS s
    INNER JOIN staging.customers AS c
        ON s.customer_id = c.customer_id
    WHERE s.ticket_date >= c.signup_date
),

support_agg AS (
    SELECT
        customer_id,
        COUNT(*) AS total_tickets,
        SUM(
            CASE
                WHEN priority = 'high' THEN 1
                ELSE 0
            END
        ) AS high_priority_tickets
    FROM support_clean
    GROUP BY customer_id
)
SELECT
    c.customer_id,
    c.signup_date,
    c.plan,
    c.region,
    ua.last_active_date,
    ua.active_days,
    ua.avg_sessions,
    ua.avg_usage_minutes,

    ut.avg_usage_last_30d,
    ut.avg_usage_prev_30d,
    (ut.avg_usage_last_30d - ut.avg_usage_prev_30d) AS usage_trend_30d,
    ba.total_payment_issues,
    ba.failed_payments_30d,
    ba.zero_paid_anomalies,
    sa.total_tickets,
    sa.high_priority_tickets,
    DATE_DIFF('day', ua.last_active_date, od.obs_date) AS recency_days,
    CASE
        WHEN DATE_DIFF('day', ua.last_active_date, od.obs_date) > 45 THEN 1
        WHEN DATE_DIFF('day', ua.last_active_date, od.obs_date) < 30 THEN 0
        ELSE NULL
    END AS churn_label
FROM staging.customers AS c
CROSS JOIN observation_date AS od
LEFT JOIN usage_agg    AS ua ON c.customer_id = ua.customer_id
LEFT JOIN usage_trends AS ut ON c.customer_id = ut.customer_id
LEFT JOIN billing_agg  AS ba ON c.customer_id = ba.customer_id
LEFT JOIN support_agg  AS sa ON c.customer_id = sa.customer_id;

