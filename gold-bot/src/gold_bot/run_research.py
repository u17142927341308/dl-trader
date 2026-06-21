"""Research entry point: data -> search -> gate -> holdout -> export artifacts.

Run manually or from CI (Phase 9):

    python -m gold_bot.run_research --out docs/data
    python -m gold_bot.run_research --source synthetic --out docs/data

``--source synthetic`` produces a deterministic offline dataset so the pipeline
(and the dashboard) can be exercised without network access. Synthetic runs are
flagged in ``status.json`` so nobody mistakes them for real results.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from config.settings import get_settings

from .data.yfinance_adapter import YFinanceAdapter
from .reporting.export import export_all
from .search.orchestrator import run_search
from .search.walk_forward import WFOConfig


def synthetic_gold(n: int = 3000, seed: int = 7) -> pd.DataFrame:
    """Deterministic gold-like daily OHLCV with alternating trend/chop regimes."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2012-01-01", periods=n, freq="B", tz="UTC")
    # Regime drift: switch sign every ~250 bars, plus noise.
    regime = np.sign(np.sin(np.arange(n) / 250.0))
    drift = regime * 0.0004
    rets = drift + rng.normal(0.0, 0.009, size=n)
    close = 1500.0 * np.exp(np.cumsum(rets))
    open_ = close * (1 + rng.normal(0, 0.001, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    vol = rng.integers(1000, 9000, n).astype(float)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx
    )
    df.index.name = "timestamp"
    return df


def load_data(source: str) -> tuple[pd.DataFrame, bool]:
    """Return (frame, is_synthetic)."""
    s = get_settings()
    if source == "synthetic":
        return synthetic_gold(), True
    try:
        adapter = YFinanceAdapter()
        df = adapter.fetch(s.instrument.yahoo_symbol, "1d", start=s.history_start)
        if len(df) < 1200:
            print(f"yfinance returned only {len(df)} bars; falling back to synthetic", file=sys.stderr)
            return synthetic_gold(), True
        return df, False
    except Exception as exc:  # noqa: BLE001
        print(f"data fetch failed ({exc}); falling back to synthetic", file=sys.stderr)
        return synthetic_gold(), True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="gold-bot research pipeline")
    parser.add_argument("--source", choices=["yfinance", "synthetic"], default="yfinance")
    parser.add_argument("--out", default="docs/data")
    parser.add_argument("--max-trials", type=int, default=200)
    args = parser.parse_args(argv)

    df, synthetic = load_data(args.source)
    print(f"data: {len(df)} bars {df.index[0].date()} -> {df.index[-1].date()} (synthetic={synthetic})")

    outcome = run_search(df, wfo_cfg=WFOConfig(), max_trials_per_family=args.max_trials)
    print(
        f"trials={outcome.n_trials} accepted={outcome.accepted} note={outcome.note!r} "
        f"wfe={outcome.walk_forward_efficiency:.2f}"
    )
    if outcome.best is not None:
        print("best:", json.dumps(outcome.best.summary()))

    written = export_all(outcome, df, args.out, timeframe="1d")

    # If running synthetic, annotate status.json so it is never mistaken for real.
    if synthetic:
        status_path = written.get("status.json")
        if status_path:
            data = json.loads(status_path.read_text())
            data["note"] = (data.get("note", "") + " [SYNTHETIC DEMO DATA]").strip()
            data["data_source"] = "synthetic"
            status_path.write_text(json.dumps(data, indent=2))

    print(f"wrote {len(written)} artifacts to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
