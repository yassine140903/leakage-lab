"""xlsx -> parquet. Run once."""
import argparse
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/online_retail_II.xlsx")
OUT = Path("data/processed/transactions.parquet")

RENAME = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_ts",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}


def read_sheets() -> pd.DataFrame:
    try:
        sheets = pd.read_excel(RAW, sheet_name=None, engine="calamine")
    except ImportError:
        print("calamine unavailable, falling back to openpyxl (slow)")
        sheets = pd.read_excel(RAW, sheet_name=None, engine="openpyxl")

    frames = []
    for name, sdf in sheets.items():
        sdf = sdf.rename(columns=RENAME)
        sdf["_sheet"] = name
        frames.append(sdf)
        print(f"  sheet {name!r}: {len(sdf):,} rows, "
              f"{sdf.invoice_ts.min()} -> {sdf.invoice_ts.max()}")
    return pd.concat(frames, ignore_index=True)


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c != "_sheet"]

    spans = df.groupby("_sheet").invoice_ts.agg(["min", "max"])
    lo, hi = spans["min"].max(), spans["max"].min()
    print(f"  sheet overlap window: {lo} -> {hi}")

    dupes = df[df.duplicated(subset=cols, keep="first")]
    inside = dupes.invoice_ts.between(lo, hi).sum()
    print(f"  exact duplicate rows dropped: {len(dupes):,} "
          f"({inside:,} inside overlap, {len(dupes) - inside:,} outside)")

    return df.drop_duplicates(subset=cols, keep="first").drop(columns="_sheet")


def main(force: bool = False) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists() and not force:
        print(f"{OUT} exists, skipping. use --force to rebuild.")
        return

    print("reading sheets...")
    df = read_sheets()
    print(f"  concatenated: {len(df):,} rows")

    df = dedupe(df)

    # mixed int/str in these columns breaks the parquet write
    for c in ["invoice", "stock_code", "description"]:
        df[c] = df[c].astype(str)

    df.to_parquet(OUT, index=False)

    cancels = df.invoice.str.startswith("C")
    keep = (df.quantity > 0) & (df.price > 0) & df.customer_id.notna()
    print("\n--- ingest summary ---")
    print(f"rows                {len(df):,}")
    print(f"date range          {df.invoice_ts.min()} -> {df.invoice_ts.max()}")
    print(f"null customer_id    {df.customer_id.isna().sum():,} "
          f"({df.customer_id.isna().mean():.1%})")
    print(f"cancellation rows   {cancels.sum():,} ({cancels.mean():.1%})")
    print(f"quantity <= 0       {(df.quantity <= 0).sum():,}")
    print(f"price <= 0          {(df.price <= 0).sum():,}")
    print(f"survivors           {keep.sum():,}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    main(**vars(p.parse_args()))