import duckdb
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SQL_DIR = PROJECT_ROOT / "sql"
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = PROJECT_ROOT / "churn.duckdb"


def read_sql_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def execute_sql_scripts(conn: duckdb.DuckDBPyConnection) -> None:
    for sql_path in (SQL_DIR / "staging.sql", SQL_DIR / "churn_features.sql"):
        sql_text = read_sql_file(sql_path)
        for statement in sql_text.split(";"):
            stmt = statement.strip()
            if not stmt:
                continue
            conn.execute(stmt)


def print_staging_row_counts(conn: duckdb.DuckDBPyConnection) -> None:
    staging_tables = conn.execute(
        """
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_schema = 'staging'
        ORDER BY table_name
        """
    ).fetchall()

    print("Staging table row counts:")
    if not staging_tables:
        print("- (no tables found in schema 'staging')")
        return

    for schema, table in staging_tables:
        count = conn.execute(
            f'SELECT COUNT(*) FROM "{schema}"."{table}"'
        ).fetchone()[0]
        print(f"- {schema}.{table}: {count:,}")


def print_analytics_churn_features_count(conn: duckdb.DuckDBPyConnection) -> None:
    count = conn.execute(
        "SELECT COUNT(*) FROM analytics.churn_features"
    ).fetchone()[0]
    print("\nAnalytics table row counts:")
    print(f"- analytics.churn_features: {count:,}")


def export_churn_features(conn: duckdb.DuckDBPyConnection) -> None:
    analytics_dir = DATA_DIR / "analytics"
    analytics_dir.mkdir(parents=True, exist_ok=True)

    output_path = analytics_dir / "churn_features.parquet"
    conn.execute(
        f"""
        COPY (
            SELECT *
            FROM analytics.churn_features
        )
        TO '{output_path.as_posix()}'
        (FORMAT PARQUET)
        """
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(DB_PATH.as_posix()) as conn:
        execute_sql_scripts(conn)
        print_staging_row_counts(conn)
        print_analytics_churn_features_count(conn)
        export_churn_features(conn)


if __name__ == "__main__":
    main()

"""
SQL pipeline runner for executing staging and feature engineering queries.
"""
