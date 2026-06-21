"""Research entry point: data -> search -> gate -> holdout -> export artifacts.

    python -m gold_bot.run_research --out docs/data            # auto source
    python -m gold_bot.run_research --source local --out docs/data
    python -m gold_bot.run_research --source synthetic --out docs/data

Data sources (auto order): committed real MGC 5m futures CSV -> Alpha Vantage
intraday (GLD proxy) -> synthetic. Everything is resampled to ``settings.timeframe``
(default 15min) and annualisation is inferred from the bars-per-day in the data.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from config.settings import Settings, get_settings

from .data.local import infer_periods_per_year, load_local, resample_ohlcv
from .reporting.export import export_all
from .search.gating import GateConfig
from .search.orchestrator import run_search
from .search.walk_forward import WFOConfig


def synthetic_5m(days: int = 40, seed: int = 7) -> pd.DataFrame:
    """Deterministic gold-like 5-minute OHLCV with trend/chop regimes."""
    rng = np.random.default_rng(seed)
    bars_per_day = 276
    sessions = pd.bdate_range("2026-04-01", periods=days, tz="UTC")
    stamps = [d + pd.Timedelta(minutes=5 * i) for d in sessions for i in range(bars_per_day)]
    idx = pd.DatetimeIndex(stamps, name="timestamp")
    n = len(idx)
    regime = np.sign(np.sin(np.arange(n) / 1500.0))
    rets = regime * 0.00003 + rng.normal(0.0, 0.0008, size=n)
    close = 4000.0 * np.exp(np.cumsum(rets))
    open_ = close * (1 + rng.normal(0, 0.0002, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0006, n)))
    vol = rng.integers(50, 3000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx
    )


def _from_alphavantage(s: Settings) -> pd.DataFrame:
    from .data.alphavantage_adapter import AlphaVantageAdapter

    adapter = AlphaVantageAdapter(
        api_key=s.alphavantage_api_key, scale=s.gold_proxy_scale, intraday_months=s.av_intraday_months
    )
    return adapter.fetch(s.gold_proxy_symbol, s.timeframe, start=s.history_start)


def load_data(source: str) -> tuple[pd.DataFrame, str]:
    """Return (frame at settings.timeframe, source_label)."""
    s = get_settings()
    tf = s.timeframe

    if source == "synthetic":
        return resample_ohlcv(synthetic_5m(), tf) if tf != "5min" else synthetic_5m(), "synthetic"

    if source == "auto":
        order = ["local"] + (["alphavantage"] if s.alphavantage_api_key else [])
    else:
        order = [source]

    for src in order:
        try:
            if src == "local":
                df = load_local("mgc", tf)
                label = f"local:MGC{tf}"
            elif src == "alphavantage":
                df = _from_alphavantage(s)
                label = f"alphavantage:{s.gold_proxy_symbol}x{s.gold_proxy_scale:g}"
            else:
                raise ValueError(f"unknown source {src!r}")
            if len(df) < 500:
                print(f"{src} returned only {len(df)} bars; trying next", file=sys.stderr)
                continue
            return df, label
        except Exception as exc:  # noqa: BLE001
            print(f"{src} fetch failed ({exc}); trying next", file=sys.stderr)

    print("all real sources failed; falling back to synthetic", file=sys.stderr)
    base = synthetic_5m()
    return (resample_ohlcv(base, tf) if tf != "5min" else base), "synthetic"


def intraday_wfo(search_len: int) -> WFOConfig:
    """Walk-forward windows sized to the available (intraday) history."""
    train = max(400, int(0.30 * search_len))
    test = max(150, int(0.12 * search_len))
    return WFOConfig(scheme="rolling", train_bars=train, test_bars=test, step_bars=test)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="gold-bot research pipeline")
    parser.add_argument(
        "--source", choices=["auto", "local", "alphavantage", "synthetic"], default="auto"
    )
    parser.add_argument("--out", default="docs/data")
    parser.add_argument("--max-trials", type=int, default=200)
    args = parser.parse_args(argv)

    s = get_settings()
    df, source_label = load_data(args.source)
    synthetic = source_label == "synthetic"
    ppy = infer_periods_per_year(df)
    print(
        f"data: {len(df)} bars @ {s.timeframe} {df.index[0]} -> {df.index[-1]} "
        f"(source={source_label}, periods/yr={ppy})"
    )

    search_len = int(len(df) * 0.8)
    # Min-trades is adapted to the ~2-month intraday sample (more history -> raise it).
    gate_cfg = GateConfig(min_trades=40)
    outcome = run_search(
        df,
        wfo_cfg=intraday_wfo(search_len),
        gate_cfg=gate_cfg,
        periods_per_year=ppy,
        max_trials_per_family=args.max_trials,
    )
    print(
        f"trials={outcome.n_trials} accepted={outcome.accepted} note={outcome.note!r} "
        f"wfe={outcome.walk_forward_efficiency:.2f}"
    )
    if outcome.best is not None:
        print("best:", json.dumps(outcome.best.summary()))

    written = export_all(outcome, df, args.out, timeframe=s.timeframe)

    status_path = written.get("status.json")
    if status_path:
        data = json.loads(status_path.read_text())
        data["data_source"] = source_label
        data["timeframe"] = s.timeframe
        if synthetic:
            data["note"] = (data.get("note", "") + " [SYNTHETIC DEMO DATA]").strip()
        status_path.write_text(json.dumps(data, indent=2))

    print(f"wrote {len(written)} artifacts to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
