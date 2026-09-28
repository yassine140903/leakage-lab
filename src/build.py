"""orders, cutoffs, entities, labels. Run after ingest."""
from pathlib import Path

import duckdb

DB = Path("data/processed/lab.duckdb")
PARQUET = "data/processed/transactions.parquet"

ORDERS = f"""
CREATE OR REPLACE TABLE orders AS
SELECT CAST(customer_id AS BIGINT) AS customer_id,
       invoice_ts,
       COUNT(DISTINCT invoice)     AS n_invoices,
       SUM(quantity * price)       AS amount
FROM read_parquet('{PARQUET}')
WHERE customer_id IS NOT NULL
  AND NOT starts_with(invoice, 'C')
  AND quantity > 0
  AND price > 0
GROUP BY 1, 2;
"""

CUTOFFS = """
CREATE OR REPLACE TABLE cutoffs AS
SELECT generate_series AS cutoff
FROM generate_series(TIMESTAMP '2010-03-01',
                     TIMESTAMP '2011-11-01',
                     INTERVAL 1 MONTH);
"""
ENTITIES = """
CREATE OR REPLACE TABLE entities AS
SELECT DISTINCT o.customer_id, c.cutoff
FROM cutoffs c
JOIN orders o ON o.invoice_ts < c.cutoff
             AND o.invoice_ts >= c.cutoff - INTERVAL 365 DAY;
"""

LABELED = """
CREATE OR REPLACE TABLE labeled AS
SELECT e.customer_id, e.cutoff,
       CAST(EXISTS (
         SELECT 1 FROM orders o
         WHERE o.customer_id = e.customer_id
           AND o.invoice_ts >= e.cutoff
           AND o.invoice_ts <  e.cutoff + INTERVAL 30 DAY
       ) AS INTEGER) AS label
FROM entities e;
"""


def main() -> None:
    con = duckdb.connect(str(DB))
    for name, sql in [("orders", ORDERS), ("cutoffs", CUTOFFS),
                      ("entities", ENTITIES), ("labeled", LABELED)]:
        con.execute(sql)
        n = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        print(f"{name:10s} {n:>10,}")

    print("\n--- sanity ---")
    print(con.execute("""
        SELECT MIN(invoice_ts) AS first_order,
               MAX(invoice_ts) AS last_order,
               COUNT(DISTINCT customer_id) AS customers,
               SUM(amount) AS total_amount
        FROM orders
    """).df().to_string(index=False))

    print("\nrows per cutoff / base rate:")
    print(con.execute("""
        SELECT cutoff, COUNT(*) AS n,
               SUM(label) AS positives,
               ROUND(AVG(label), 4) AS base_rate
        FROM labeled GROUP BY cutoff ORDER BY cutoff
    """).df().to_string(index=False))

    overall = con.execute("SELECT AVG(label) FROM labeled").fetchone()[0]
    print(f"\noverall base rate: {overall:.4f}")
    con.close()


if __name__ == "__main__":
    main()