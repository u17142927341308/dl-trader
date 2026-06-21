"""Offline tests for the Alpha Vantage adapter parser (no network)."""

from __future__ import annotations

import pytest

from gold_bot.data.adapter import DataError
from gold_bot.data.alphavantage_adapter import AlphaVantageAdapter

_PAYLOAD = {
    "Meta Data": {"2. Symbol": "GLD"},
    "Time Series (Daily)": {
        "2024-01-03": {"1. open": "190.0", "2. high": "191.0", "3. low": "189.0", "4. close": "190.5", "5. volume": "1000"},
        "2024-01-02": {"1. open": "188.0", "2. high": "189.5", "3. low": "187.5", "4. close": "189.0", "5. volume": "1200"},
    },
}


def test_parse_scales_price_and_sorts() -> None:
    df = AlphaVantageAdapter._parse(_PAYLOAD, scale=10.0)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    # GLD close 190.5 * 10 = 1905 -> ~gold level; volume unscaled.
    assert df["close"].iloc[-1] == pytest.approx(1905.0)
    assert df["volume"].iloc[-1] == pytest.approx(1000.0)


def test_parse_surfaces_rate_limit_note() -> None:
    with pytest.raises(DataError):
        AlphaVantageAdapter._parse({"Note": "rate limited"}, scale=10.0)


def test_parse_rejects_unexpected_payload() -> None:
    with pytest.raises(DataError):
        AlphaVantageAdapter._parse({"Meta Data": {}}, scale=10.0)


def test_requires_api_key() -> None:
    with pytest.raises(DataError):
        AlphaVantageAdapter(api_key="")
