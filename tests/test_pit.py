import numpy as np
import pandas as pd

from src.features import FEATS
from src.store import get_online_features

ASOF_FEATS = ["lifetime_spend", "lifetime_orders", "recency_days", "tenure_days"]


def test_no_future_data(pit_df):
    """Recency measured back from the cutoff must be strictly positive."""
    assert (pit_df.recency_days > 0).all()
    assert (pit_df.tenure_days > 0).all()


def test_tenure_at_least_recency(pit_df):
    """First purchase is never after the last one."""
    assert (pit_df.tenure_days >= pit_df.recency_days).all()


def test_windowed_features_bounded(pit_df):
    """90d spend can't exceed lifetime spend."""
    assert (pit_df.spend_90d <= pit_df.lifetime_spend + 1e-6).all()
    assert (pit_df.orders_90d <= pit_df.lifetime_orders).all()


def test_stale_window_is_zero(pit_df):
    """If the last purchase predates the window, 90d features must be 0."""
    stale = pit_df[pit_df.recency_days > 90]
    assert len(stale) > 0
    assert (stale.spend_90d == 0).all()
    assert (stale.orders_90d == 0).all()


def test_two_implementations_agree(pit_df, asof_df):
    """Group-by/filter vs window/ASOF join."""
    m = pit_df.merge(asof_df, on=["customer_id", "cutoff"], suffixes=("", "_asof"))
    assert len(m) == len(pit_df)
    for c in ASOF_FEATS:
        assert np.allclose(m[c], m[f"{c}_asof"]), c


def test_online_offline_parity(pit_df, orders_df):
    """SQL offline path vs pandas online path."""
    sample = pit_df.sample(200, random_state=0)
    for row in sample.itertuples():
        online = get_online_features(orders_df, row.customer_id, row.cutoff)
        for c in FEATS:
            assert np.isclose(online[c], getattr(row, c)), \
                (row.customer_id, row.cutoff, c, online[c], getattr(row, c))

def test_naive_is_actually_leaky(naive_df):
    """Naive features are constant per customer across cutoffs."""
    g = naive_df.groupby("customer_id")[FEATS]
    spread = (g.max() - g.min()).abs()
    assert (spread < 1e-6).all().all(), spread.max()


def test_naive_differs_from_pit(pit_df, naive_df):
    """The two pipelines must actually differ, or there's no experiment."""
    m = pit_df.merge(naive_df, on=["customer_id", "cutoff"], suffixes=("", "_naive"))
    assert (m.recency_days != m.recency_days_naive).mean() > 0.5


def test_labels_identical(pit_df, naive_df):
    """Same labels both sides — the invariant the whole experiment rests on."""
    m = pit_df.merge(naive_df, on=["customer_id", "cutoff"], suffixes=("", "_naive"))
    assert len(m) == len(pit_df)
    assert (m.label == m.label_naive).all()