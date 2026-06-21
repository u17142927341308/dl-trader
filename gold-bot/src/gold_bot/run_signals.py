"""Lightweight signals refresh (runs more frequently than the full research).

Reads the currently accepted strategy from ``docs/data/current_strategy.json``
(written by the research job), fetches the latest bars, and re-emits
``signals.json`` + bumps ``status.json``'s timestamp. If no strategy is
accepted, it publishes a FLAT signal. It does NOT re-run the search.

    python -m gold_bot.run_signals --out docs/data
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from config.settings import get_settings

from .run_research import load_data
from .signals.generator import signal_from_params
from .signals.schema import SignalArtifact


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="gold-bot signals refresh")
    parser.add_argument(
        "--source", choices=["auto", "alphavantage", "yfinance", "synthetic"], default="auto"
    )
    parser.add_argument("--out", default="docs/data")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    s = get_settings()
    df, source_label = load_data(args.source)

    strat_path = out / "current_strategy.json"
    if strat_path.exists():
        cs = json.loads(strat_path.read_text())
        sig = signal_from_params(cs["family"], cs["params"], df, timeframe="1d", settings=s)
    else:
        sig = SignalArtifact(
            generated_at=_now_iso(),
            instrument=s.instrument.symbol,
            timeframe="1d",
            signal="FLAT",
            account_headroom_to_trailing_dd=s.account_rules.trailing_drawdown,
            confidence_notes="No accepted strategy on file.",
            auto_execution=False,
        )

    (out / "signals.json").write_text(json.dumps(sig.model_dump(), indent=2, default=str))

    status_path = out / "status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text())
        status["generated_at"] = _now_iso()
        status["data_as_of"] = df.index[-1].isoformat() if len(df) else None
        status_path.write_text(json.dumps(status, indent=2, default=str))

    print(f"signals: {sig.signal} (source={source_label})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
