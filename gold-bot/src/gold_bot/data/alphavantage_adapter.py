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


class AlphaVantageAdapter(DataAdapter):
    def __init__(
        self,
        api_key: str,
        cache: ParquetCache | None = None,
        scale: float = 10.0,
        max_retries: int = 3,
    ) -> None:
        if not api_key:
            raise DataError("AlphaVantageAdapter requires an API key")
        self.api_key = api_key
        self.cache = cache or ParquetCache()
        self.scale = scale
        self.max_retries = max_retries

    @staticmethod
    def _parse(payload: dict, scale: float) -> pd.DataFrame:
        """Turn an Alpha Vantage TIME_SERIES_DAILY payload into scaled OHLCV."""
        # Surface rate-limit / error messages instead of silently failing.
        for flag in ("Note", "Information", "Error Message"):
            if flag in payload:
                raise DataError(f"Alpha Vantage: {payload[flag]}")
        series = payload.get("Time Series (Daily)")
        if not series:
            raise DataError(f"Alpha Vantage: unexpected payload keys {list(payload)}")
        df = pd.DataFrame.from_dict(series, orient="index").rename(columns=_COLMAP)
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        for col in ("open", "high", "low", "close"):
            df[col] *= scale  # scale price to ~gold level; leave volume as-is
        return normalize_ohlcv(df)

    def _cache_key(self, symbol: str) -> str:
        return f"{symbol}x{self.scale:g}"

    def fetch(self, symbol: str, timeframe: str, start: str | None = None) -> pd.DataFrame:
        if timeframe != "1d":
            raise DataError("AlphaVantageAdapter currently supports only '1d'")

        cached = self.cache.load(self._cache_key(symbol), timeframe)
        if cached is not None:
            df = cached
        else:
            df = self._download(symbol)
            self.cache.save(self._cache_key(symbol), timeframe, df)

        if start:
            df = df[df.index >= pd.Timestamp(start, tz="UTC")]
        return df

    def _download(self, symbol: str) -> pd.DataFrame:
        url = (
            f"{_BASE}?function=TIME_SERIES_DAILY&symbol={symbol}"
            f"&outputsize=full&apikey={self.api_key}"
        )
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                return self._parse(payload, self.scale)
            except DataError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
        raise DataError(f"Alpha Vantage download failed for {symbol}: {last_err}")
