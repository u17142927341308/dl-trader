"""Parquet caching with integrity checks.

GitHub Actions has limited free minutes and yfinance is rate-limited, so we
cache normalised OHLCV frames to parquet between runs. The cache is purely a
performance optimisation: deleting ``.cache/`` must never change results, only
make the next run slower.

Integrity:
    * a sidecar ``.meta.json`` stores row count + first/last timestamp + a
      content hash; on load we re-validate, and a corrupt/mismatched cache is
      treated as a miss rather than silently trusted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .adapter import OHLCV_COLUMNS, DataError, normalize_ohlcv


def _content_hash(df: pd.DataFrame) -> str:
    """Stable hash of the frame's values + index for integrity checks."""
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df.index).values.tobytes())
    for col in OHLCV_COLUMNS:
        h.update(pd.util.hash_pandas_object(df[col]).values.tobytes())
    return h.hexdigest()


def _key(symbol: str, timeframe: str) -> str:
    safe = symbol.replace("=", "").replace("/", "_")
    return f"{safe}__{timeframe}"


class ParquetCache:
    """Read/write normalised OHLCV frames to a parquet cache directory."""

    def __init__(self, cache_dir: str | Path = ".cache") -> None:
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _paths(self, symbol: str, timeframe: str) -> tuple[Path, Path]:
        key = _key(symbol, timeframe)
        return self.dir / f"{key}.parquet", self.dir / f"{key}.meta.json"

    def load(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        """Return the cached frame, or None on miss / failed integrity check."""
        data_path, meta_path = self._paths(symbol, timeframe)
        if not data_path.exists() or not meta_path.exists():
            return None
        try:
            df = pd.read_parquet(data_path)
            df = normalize_ohlcv(df)
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError, DataError, json.JSONDecodeError):
            return None  # treat corruption as a miss

        if meta.get("rows") != len(df) or meta.get("hash") != _content_hash(df):
            return None  # tampered / mismatched -> miss
        return df

    def save(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        """Persist a normalised frame plus its integrity sidecar."""
        df = normalize_ohlcv(df)
        data_path, meta_path = self._paths(symbol, timeframe)
        df.to_parquet(data_path)
        meta = {
            "symbol": symbol,
            "timeframe": timeframe,
            "rows": len(df),
            "first": df.index[0].isoformat(),
            "last": df.index[-1].isoformat(),
            "hash": _content_hash(df),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
