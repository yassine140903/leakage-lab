"""Experiments A-E. Model fixed, data varies."""
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.features import FEATS

DB = Path("data/processed/lab.duckdb")
TRAIN_END = pd.Timestamp("2011-06-01")
TEST_START = pd.Timestamp("2011-07-01")


def model():
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                              random_state=0, verbose=-1)


def score(y, p):
    return roc_auc_score(y, p), average_precision_score(y, p)


def temporal_split(df):
    return df[df.cutoff <= TRAIN_END], df[df.cutoff >= TEST_START]


def fit_predict(tr, te_features, te_labels):
    m = model()
    m.fit(tr[FEATS], tr.label)
    p = m.predict_proba(te_features[FEATS])[:, 1]
    return m, score(te_labels, p)


def main() -> None:
    con = duckdb.connect(str(DB), read_only=True)
    pit = con.execute("SELECT * FROM pit_features").df()
    naive = con.execute("SELECT * FROM naive_features").df()
    con.close()

    key = ["customer_id", "cutoff"]
    pit = pit.sort_values(key).reset_index(drop=True)
    naive = naive.sort_values(key).reset_index(drop=True)
    assert (pit[key].values == naive[key].values).all(), "row alignment broken"

    rows, importances = [], {}

    # --- random split (shared indices so A and A' see the same test rows)
    idx_tr, idx_te = train_test_split(np.arange(len(pit)), test_size=0.3,
                                      random_state=0)

    m, s = fit_predict(naive.iloc[idx_tr], naive.iloc[idx_te],
                       naive.iloc[idx_te].label)
    rows.append(("A", "naive", "random", "naive", *s,
                 naive.iloc[idx_te].label.mean()))
    importances["A"] = m.feature_importances_

    # A': same model as A, scored on PIT test rows. Isolates the feature
    # change with the test set held fixed.
    m_a = model()
    m_a.fit(naive.iloc[idx_tr][FEATS], naive.iloc[idx_tr].label)
    p = m_a.predict_proba(pit.iloc[idx_te][FEATS])[:, 1]
    rows.append(("A'", "naive", "random", "PIT", *score(pit.iloc[idx_te].label, p),
                 pit.iloc[idx_te].label.mean()))

    m, s = fit_predict(pit.iloc[idx_tr], pit.iloc[idx_te], pit.iloc[idx_te].label)
    rows.append(("B", "PIT", "random", "PIT", *s, pit.iloc[idx_te].label.mean()))

    # --- temporal split
    ptr, pte = temporal_split(pit)
    ntr, nte = temporal_split(naive)

    m, s = fit_predict(ptr, pte, pte.label)
    rows.append(("C", "PIT", "temporal", "PIT", *s, pte.label.mean()))
    importances["C"] = m.feature_importances_

    # D: train naive, serve PIT
    m_d = model()
    m_d.fit(ntr[FEATS], ntr.label)
    p = m_d.predict_proba(pte[FEATS])[:, 1]
    rows.append(("D", "naive", "temporal", "PIT", *score(pte.label, p),
                 pte.label.mean()))

    # D': naive model on naive test rows, for reference
    p = m_d.predict_proba(nte[FEATS])[:, 1]
    rows.append(("D'", "naive", "temporal", "naive", *score(nte.label, p),
                 nte.label.mean()))

    # E: no model
    rows.append(("E", "recency only", "temporal", "PIT",
                 *score(pte.label, -pte.recency_days), pte.label.mean()))

    res = pd.DataFrame(rows, columns=["id", "features", "split", "eval_on",
                                      "roc_auc", "pr_auc", "base_rate"])
    res["pr_lift"] = res.pr_auc / res.base_rate
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n--- feature importances (gain-split counts) ---")
    imp = pd.DataFrame(importances, index=FEATS)
    print(imp.to_string())

    res.to_csv("results.csv", index=False)


if __name__ == "__main__":
    main()