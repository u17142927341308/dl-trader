"""Tests for the data layer: normalisation, caching, integrity, fake adapter."""

from __future__ import annotations

import pandas as pd
import pytest

from gold_bot.data.adapter import DataAdapter, DataError, normalize_ohlcv
from gold_bot.data.cache import ParquetCache


def test_normalize_uppercase_columns_and_tz(ohlcv: pd.DataFrame) -> None:
    raw = ohlcv.rename(columns=str.title)  # Open, High, ...
    raw.index = raw.index.tz_convert(None)  # strip tz
    out = normalize_ohlcv(raw)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert str(out.index.tz) == "UTC"
    assert out.index.is_monotonic_increasing


def test_normalize_drops_duplicates_and_nans() -> None:
    idx = pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"], utc=True)
    df = pd.DataFrame(
        {
            "open": [1.0, 1.5, 2.0],
            "high": [1.0, 1.5, 2.0],
            "low": [1.0, 1.5, 2.0],
            "close": [1.0, 1.5, float("nan")],
            "volume": [1, 1, 1],
        },
        index=idx,
    )
    out = normalize_ohlcv(df)
    assert len(out) == 1  # dup collapsed (keep last 1.5), nan-close row dropped
    assert out["close"].iloc[0] == pytest.approx(1.5)


def test_normalize_missing_columns_raises() -> None:
    df = pd.DataFrame({"open": [1.0], "close": [1.0]})
    with pytest.raises(DataError):
        normalize_ohlcv(df)


def test_cache_roundtrip(tmp_path, ohlcv: pd.DataFrame) -> None:
    cache = ParquetCache(tmp_path)
    assert cache.load("GC=F", "1d") is None  # miss
    cache.save("GC=F", "1d", ohlcv)
    loaded = cache.load("GC=F", "1d")
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, normalize_ohlcv(ohlcv))


def test_cache_integrity_miss_on_tamper(tmp_path, ohlcv: pd.DataFrame) -> None:
    cache = ParquetCache(tmp_path)
    cache.save("GC=F", "1d", ohlcv)
    # Corrupt the parquet so re-hash won't match the meta sidecar.
    data_path, _ = cache._paths("GC=F", "1d")
    tampered = normalize_ohlcv(ohlcv).copy()
    tampered.iloc[0, tampered.columns.get_loc("close")] += 1.0
    tampered.to_parquet(data_path)
    assert cache.load("GC=F", "1d") is None  # integrity check catches it


class _FakeAdapter(DataAdapter):
    """Offline adapter so tests never hit the network."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def fetch(self, symbol: str, timeframe: str, start: str | None = None) -> pd.DataFrame:
        return normalize_ohlcv(self._df)


def test_fake_adapter_conforms(ohlcv: pd.DataFrame) -> None:
    adapter = _FakeAdapter(ohlcv)
    out = adapter.fetch("GC=F", "1d")
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert str(out.index.tz) == "UTC"
