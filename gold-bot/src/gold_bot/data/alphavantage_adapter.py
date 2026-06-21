"""Alpha Vantage data adapter (gold proxy via the GLD ETF).

Alpha Vantage does not serve COMEX gold *futures* (GC), so we use the **GLD**
ETF as a close daily proxy and scale it to roughly the gold spot/futures price
level (GLD trades near 1/10 of spot, so the default ``scale=10`` puts the series
on a ~$2000 footing that matches the GC/MGC contract economics in settings).

This is a research proxy, not exchange settlement data: the scale factor drifts
slowly with the ETF's expense ratio. Swap in a true futures source behind the
same :class:`DataAdapter` interface when you have one.

The API key is read from settings (env ``GOLDBOT_ALPHAVANTAGE_API_KEY``) and is
NEVER written to disk or committed.
"""

from __future__ import annotations

import json
import time
import urllib.request

import pandas as pd

from .adapter import DataAdapter, DataError, normalize_ohlcv
from .cache import ParquetCache

_BASE = "https://www.alphavantage.co/query"
_COLMAP = {
    "1. open": "open",
    "2. high": "high",
    "3. low": "low",
    "4. close": "close",
    "5. volume": "volume",
}
# Intraday timeframes we map to Alpha Vantage interval strings.
_INTRADAY = {"1min", "5min", "15min", "30min", "60min"}


class AlphaVantageAdapter(DataAdapter):
    def __init__(
        self,
        api_key: str,
        cache: ParquetCache | None = None,
        scale: float = 10.0,
        max_retries: int = 3,
        intraday_months: int = 24,
    ) -> None:
        if not api_key:
            raise DataError("AlphaVantageAdapter requires an API key")
        self.api_key = api_key
        self.cache = cache or ParquetCache()
        self.scale = scale
        self.max_retries = max_retries
        self.intraday_months = intraday_months

    @staticmethod
    def _parse(payload: dict, scale: float, eastern: bool = False) -> pd.DataFrame:
        """Turn an Alpha Vantage time-series payload into scaled OHLCV.

        ``eastern=True`` (intraday) treats the naive timestamps as US/Eastern and
        converts to UTC; daily bars are taken as calendar dates (UTC).
        """
        for flag in ("Note", "Information", "Error Message"):
            if flag in payload:
                raise DataError(f"Alpha Vantage: {payload[flag]}")
        series_key = next((k for k in payload if k.startswith("Time Series")), None)
        if not series_key:
            raise DataError(f"Alpha Vantage: unexpected payload keys {list(payload)}")
        series = payload[series_key]
        df = pd.DataFrame.from_dict(series, orient="index").rename(columns=_COLMAP)
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        for col in ("open", "high", "low", "close"):
            df[col] *= scale  # scale price to ~gold level; leave volume as-is
        if eastern:
            df.index = (
                pd.to_datetime(df.index).tz_localize("America/New_York", nonexistent="shift_forward", ambiguous="NaT")
            )
            df = df[df.index.notna()]
            df.index = df.index.tz_convert("UTC")
        return normalize_ohlcv(df)

    def _cache_key(self, symbol: str, timeframe: str) -> str:
        return f"{symbol}_{timeframe}x{self.scale:g}"

    def fetch(self, symbol: str, timeframe: str, start: str | None = None) -> pd.DataFrame:
        cached = self.cache.load(self._cache_key(symbol, timeframe), timeframe)
        if cached is not None:
            df = cached
        elif timeframe == "1d":
            df = self._download_daily(symbol)
            self.cache.save(self._cache_key(symbol, timeframe), timeframe, df)
        elif timeframe in _INTRADAY:
            df = self._download_intraday(symbol, timeframe)
            self.cache.save(self._cache_key(symbol, timeframe), timeframe, df)
        else:
            raise DataError(f"AlphaVantageAdapter: unsupported timeframe {timeframe!r}")

        if start:
            df = df[df.index >= pd.Timestamp(start, tz="UTC")]
        return df

    def _get(self, url: str) -> dict:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
        raise DataError(f"Alpha Vantage request failed: {last_err}")

    def _download_daily(self, symbol: str) -> pd.DataFrame:
        url = (
            f"{_BASE}?function=TIME_SERIES_DAILY&symbol={symbol}"
            f"&outputsize=full&apikey={self.api_key}"
        )
        return self._parse(self._get(url), self.scale, eastern=False)

    def _download_intraday(self, symbol: str, interval: str) -> pd.DataFrame:
        """Pull recent months of intraday bars and stitch them together.

        Alpha Vantage serves extended intraday history one calendar month at a
        time via the ``month=YYYY-MM`` parameter; we loop back ``intraday_months``
        months and concatenate.
        """
        frames: list[pd.DataFrame] = []
        month = pd.Timestamp.utcnow().normalize().replace(day=1)
        for _ in range(max(1, self.intraday_months)):
            tag = f"{month.year:04d}-{month.month:02d}"
            url = (
                f"{_BASE}?function=TIME_SERIES_INTRADAY&symbol={symbol}"
                f"&interval={interval}&month={tag}&outputsize=full"
                f"&extended_hours=false&apikey={self.api_key}"
            )
            try:
                frames.append(self._parse(self._get(url), self.scale, eastern=True))
            except DataError:
                pass  # a month with no data / a transient note — skip it
            month = (month - pd.Timedelta(days=1)).replace(day=1)
        if not frames:
            raise DataError(f"Alpha Vantage: no intraday data for {symbol} {interval}")
        combined = pd.concat(frames)
        return normalize_ohlcv(combined)

