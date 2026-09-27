"""Feature computation. One template, two as_of values."""
from pathlib import Path

import duckdb

DB = Path("data/processed/lab.duckdb")

TODAY = "TIMESTAMP '2011-12-10'"

FEATURE_SQL = """
SELECT e.customer_id, e.cutoff, e.label,
       SUM(o.amount)                                AS lifetime_spend,
       SUM(o.n_invoices)                            AS lifetime_orders,
       date_diff('day', MAX(o.invoice_ts), e.as_of) AS recency_days,
       date_diff('day', MIN(o.invoice_ts), e.as_of) AS tenure_days,
       COALESCE(SUM(o.amount)     FILTER (WHERE o.invoice_ts >= e.as_of - INTERVAL 90 DAY), 0) AS spend_90d,
       COALESCE(SUM(o.n_invoices) FILTER (WHERE o.invoice_ts >= e.as_of - INTERVAL 90 DAY), 0) AS orders_90d
FROM (SELECT *, {as_of} AS as_of FROM labeled) e
JOIN orders o
  ON o.customer_id = e.customer_id
 AND o.invoice_ts  < e.as_of
GROUP BY e.customer_id, e.cutoff, e.label, e.as_of
"""

SNAPSHOTS = """
CREATE OR REPLACE TABLE snapshots AS
SELECT customer_id,
       invoice_ts AS feature_ts,
       SUM(amount)     OVER w AS lifetime_spend,
       SUM(n_invoices) OVER w AS lifetime_orders,
       MIN(invoice_ts) OVER w AS first_ts
FROM orders
WINDOW w AS (PARTITION BY customer_id ORDER BY invoice_ts
             ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW);
"""

ASOF_SQL = """
SELECT e.customer_id, e.cutoff, e.label,
       s.lifetime_spend,
       s.lifetime_orders,
       date_diff('day', s.feature_ts, e.cutoff) AS recency_days,
       date_diff('day', s.first_ts,   e.cutoff) AS tenure_days
FROM labeled e
ASOF LEFT JOIN snapshots s
  ON e.customer_id = s.customer_id
 AND e.cutoff > s.feature_ts
"""

ASOF_FEATS = ["lifetime_spend", "lifetime_orders", "recency_days", "tenure_days"]

FEATS = ["lifetime_spend", "lifetime_orders", "recency_days",
         "tenure_days", "spend_90d", "orders_90d"]


def build(con, as_of: str):
    return con.execute(FEATURE_SQL.format(as_of=as_of)).df()


def main() -> None:
    con = duckdb.connect(str(DB))
    pit = build(con, "cutoff")
    naive = build(con, TODAY)
    con.execute(SNAPSHOTS)
    asof_df = con.execute(ASOF_SQL).df()
    print(f"\n--- asof ---")
    print(f"rows {len(asof_df):,}  nulls {asof_df.lifetime_spend.isna().sum():,}")

    m = pit.merge(asof_df, on=["customer_id", "cutoff"], suffixes=("", "_asof"))
    print(f"merged {len(m):,} (expect {len(pit):,})")
    for c in ASOF_FEATS:
        diff = (m[c] - m[f"{c}_asof"]).abs()
        print(f"  {c:18s} max_abs_diff {diff.max():.6f}  mismatches {(diff > 1e-6).sum():,}")

    con.register("asof_df", asof_df)
    con.execute("CREATE OR REPLACE TABLE asof_features AS SELECT * FROM asof_df")

    for name, df in [("pit", pit), ("naive", naive)]:
        print(f"\n--- {name} ---")
        print(f"rows {len(df):,}  positives {df.label.sum():,}  "
              f"base rate {df.label.mean():.4f}")
        print(df[FEATS].describe().T.to_string())

    con.execute("CREATE OR REPLACE TABLE pit_features AS SELECT * FROM pit")
    con.execute("CREATE OR REPLACE TABLE naive_features AS SELECT * FROM naive")
    con.close()


if __name__ == "__main__":
    main()