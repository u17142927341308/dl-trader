"""Shared pytest fixtures: deterministic synthetic OHLCV for indicator tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    """A reproducible 500-bar daily OHLCV frame (random walk around gold-ish prices)."""
    rng = np.random.default_rng(42)
    n = 500
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    steps = rng.normal(0.0, 8.0, size=n).cumsum()
    close = 1800.0 + steps
    open_ = close + rng.normal(0.0, 2.0, size=n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0.0, 3.0, size=n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.0, 3.0, size=n))
    volume = rng.integers(1000, 5000, size=n).astype(float)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    df.index.name = "timestamp"
    return df
