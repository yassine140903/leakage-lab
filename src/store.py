"""Feature store API: offline (SQL) and online (pandas) paths."""
from pathlib import Path

import duckdb
import pandas as pd

from src.features import FEATURE_SQL, FEATS

DB = Path("data/processed/lab.duckdb")


def get_training_features(entity_df: pd.DataFrame,
                          con: duckdb.DuckDBPyConnection | None = None
                          ) -> pd.DataFrame:
    """Offline path. entity_df: customer_id, cutoff (label optional)."""
    own = con is None
    con = con or duckdb.connect(str(DB), read_only=True)
    try:
        if "label" not in entity_df.columns:
            entity_df = entity_df.assign(label=0)
        con.register("entity_df", entity_df)
        sql = FEATURE_SQL.replace("FROM labeled", "FROM entity_df")
        return con.execute(sql.format(as_of="cutoff")).df()
    finally:
        if own:
            con.close()


def get_online_features(orders: pd.DataFrame,
                        customer_id: int,
                        as_of: pd.Timestamp) -> dict:
    """Online path. Deliberately pandas, no SQL — independent implementation."""
    as_of = pd.Timestamp(as_of)
    h = orders[(orders.customer_id == customer_id)
               & (orders.invoice_ts < as_of)]
    if h.empty:
        return {f: None for f in FEATS}

    w = h[h.invoice_ts >= as_of - pd.Timedelta(days=90)]
    return {
        "lifetime_spend":  h.amount.sum(),
        "lifetime_orders": h.n_invoices.sum(),
        "recency_days":    (as_of.normalize() - h.invoice_ts.max().normalize()).days,
        "tenure_days":     (as_of.normalize() - h.invoice_ts.min().normalize()).days,
        "spend_90d":       w.amount.sum(),
        "orders_90d":      w.n_invoices.sum(),
    }


def load_orders(con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    own = con is None
    con = con or duckdb.connect(str(DB), read_only=True)
    try:
        return con.execute("SELECT * FROM orders").df()
    finally:
        if own:
            con.close()