"""Assemble validated artifacts and write ``docs/data/*.json`` for the dashboard.

Every artifact is built through its pydantic model first, so anything written to
disk is guaranteed to match the schema the front-end expects.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from config.settings import Settings, get_settings

from ..backtest.metrics import Metrics
from ..risk.prop_rules import TrailingDrawdown
from ..signals.generator import build_signal
from ..signals.schema import (
    BestCandidateArtifact,
    CurrentStrategyArtifact,
    EquityCurveArtifact,
    FoldSummary,
    HoldoutSummary,
    MetricRow,
    MetricsArtifact,
    StatusArtifact,
    TradesArtifact,
    WalkForwardArtifact,
)
from ..strategies.base import make_strategy_id


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _trailing_floor(equity: pd.Series, amount: float) -> list[float]:
    tdd = TrailingDrawdown(float(equity.iloc[0]), amount)
    out: list[float] = []
    for v in equity.to_numpy(dtype=float):
        out.append(tdd.floor)
        tdd.on_day_close(v)
    return out


def _status(outcome: object, data_asof: str, settings: Settings) -> StatusArtifact:
    accepted = getattr(outcome, "accepted", False)
    state = "accepted" if accepted else (getattr(outcome, "note", None) or "searching")
    headroom = settings.account_rules.trailing_drawdown
    if getattr(outcome, "best_oos_result", None) is not None:
        eq = outcome.best_oos_result.equity
        floor0 = settings.account_rules.account_size - settings.account_rules.trailing_drawdown
        headroom = float(eq.iloc[-1] - floor0)
    return StatusArtifact(
        generated_at=_now_iso(),
        data_as_of=data_asof,
        search_state=state,
        account_headroom_to_trailing_dd=headroom,
        daily_loss_state="ok",
        n_trials=getattr(outcome, "n_trials", 0),
        note=getattr(outcome, "note", ""),
    )


def _metrics_rows(oos: Metrics | None, holdout: Metrics | None) -> MetricsArtifact:
    names = [
        ("Sharpe", "sharpe"),
        ("Sortino", "sortino"),
        ("Calmar", "calmar"),
        ("Total return %", "total_return_pct"),
        ("CAGR", "cagr"),
        ("Profit factor", "profit_factor"),
        ("Win rate", "win_rate"),
        ("Payoff ratio", "payoff_ratio"),
        ("Expectancy $", "expectancy"),
        ("Max DD $", "max_dd_dollars"),
        ("Max DD %", "max_dd_pct"),
        ("Trailing-DD headroom $", "trailing_dd_min_headroom"),
        ("Worst daily loss $", "worst_daily_loss"),
        ("Exposure", "exposure"),
        ("# Trades", "n_trades"),
        ("Longest losing streak", "longest_losing_streak"),
        ("Deflated Sharpe", "deflated_sharpe"),
        ("MC bust prob", "monte_carlo_bust_prob"),
    ]

    def fmt(metrics: Metrics | None, attr: str):
        if metrics is None:
            return None
        val = getattr(metrics, attr)
        if isinstance(val, float):
            return round(val, 4)
        return val

    rows = [MetricRow(name=label, oos=fmt(oos, attr), holdout=fmt(holdout, attr)) for label, attr in names]
    return MetricsArtifact(rows=rows)


def _walkforward(outcome: object) -> WalkForwardArtifact:
    ledger = getattr(outcome, "ledger", None)
    best_metrics = getattr(outcome, "best_oos_metrics", None)
    folds: list[FoldSummary] = []
    top: list[dict] = []
    if ledger is not None:
        # Best trial per family, for a compact per-family view.
        by_family: dict[str, object] = {}
        for t in ledger.trials:
            cur = by_family.get(t.family)
            if cur is None or t.score > cur.score:
                by_family[t.family] = t
        for fam, t in by_family.items():
            folds.append(
                FoldSummary(
                    family=fam,
                    oos_sharpe=round(t.oos_metrics.sharpe, 3),
                    oos_profit_factor=round(t.oos_metrics.profit_factor, 3),
                    n_trades=t.oos_metrics.n_trades,
                )
            )
        ranked = sorted(ledger.trials, key=lambda x: x.score, reverse=True)[:10]
        top = [t.summary() for t in ranked]
    return WalkForwardArtifact(
        total_trials=getattr(getattr(outcome, "ledger", None), "n_trials", 0),
        deflated_sharpe=round(best_metrics.deflated_sharpe, 4) if best_metrics else None,
        walk_forward_efficiency=round(getattr(outcome, "walk_forward_efficiency", 0.0), 4),
        folds=folds,
        top_trials=top,
    )


def _equity(outcome: object, settings: Settings) -> EquityCurveArtifact:
    result = getattr(outcome, "best_oos_result", None)
    if result is None or len(result.equity) == 0:
        return EquityCurveArtifact(timestamps=[], equity=[], trailing_dd_floor=[])
    eq = result.equity
    return EquityCurveArtifact(
        timestamps=[t.isoformat() for t in eq.index],
        equity=[round(float(v), 2) for v in eq.to_numpy()],
        trailing_dd_floor=[round(v, 2) for v in _trailing_floor(eq, settings.account_rules.trailing_drawdown)],
    )


def _trades(outcome: object) -> TradesArtifact:
    result = getattr(outcome, "best_oos_result", None)
    if result is None or not result.trades:
        return TradesArtifact(rows=[])
    rows = []
    for t in result.trades[-300:]:
        rows.append(
            {
                "entry_time": t.entry_time.isoformat(),
                "exit_time": t.exit_time.isoformat(),
                "dir": "LONG" if t.direction > 0 else "SHORT",
                "size": t.size,
                "entry": round(t.entry_price, 2),
                "exit": round(t.exit_price, 2),
                "net_pnl": round(t.net_pnl, 2),
                "reason": t.reason,
            }
        )
    return TradesArtifact(rows=rows)


def _current_strategy(outcome: object) -> CurrentStrategyArtifact | None:
    cs = getattr(outcome, "current_strategy", None)
    if not cs:
        return None
    return CurrentStrategyArtifact(
        strategy_id=cs["strategy_id"],
        family=cs["family"],
        params=cs["params"],
        holdout=HoldoutSummary(**cs["holdout"]),
    )


def _best_candidate(outcome: object) -> BestCandidateArtifact | None:
    best = getattr(outcome, "best", None)
    if best is None:
        return None
    gate = getattr(outcome, "gate_result", None)
    failures = gate.failures if gate is not None else []
    return BestCandidateArtifact(
        strategy_id=make_strategy_id(best.family, best.params),
        family=best.family,
        params=best.params,
        accepted=bool(getattr(outcome, "accepted", False)),
        oos_sharpe=round(best.oos_metrics.sharpe, 3),
        oos_profit_factor=round(best.oos_metrics.profit_factor, 3),
        oos_return_pct=round(best.oos_metrics.total_return_pct, 4),
        n_trades=best.oos_metrics.n_trades,
        gate_failures=failures,
    )


def _write(path: Path, model_or_dict: object) -> None:
    if hasattr(model_or_dict, "model_dump"):
        payload = model_or_dict.model_dump()
    else:
        payload = model_or_dict
    path.write_text(json.dumps(payload, indent=2, default=str))


def export_all(
    outcome: object,
    df: pd.DataFrame,
    out_dir: str | Path,
    *,
    timeframe: str = "1d",
    settings: Settings | None = None,
) -> dict[str, Path]:
    """Write every dashboard artifact and return the paths written."""
    s = settings or get_settings()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_asof = df.index[-1].isoformat() if len(df) else None

    holdout_metrics = getattr(outcome, "holdout_metrics", None)
    artifacts: dict[str, object] = {
        "status.json": _status(outcome, data_asof, s),
        "equity_curve.json": _equity(outcome, s),
        "metrics.json": _metrics_rows(getattr(outcome, "best_oos_metrics", None), holdout_metrics),
        "trades.json": _trades(outcome),
        "walkforward.json": _walkforward(outcome),
        "signals.json": build_signal(outcome, df, timeframe=timeframe, settings=s),
    }
    cs = _current_strategy(outcome)
    if cs is not None:
        artifacts["current_strategy.json"] = cs
    bc = _best_candidate(outcome)
    if bc is not None:
        artifacts["best_candidate.json"] = bc

    written: dict[str, Path] = {}
    for name, artifact in artifacts.items():
        p = out / name
        _write(p, artifact)
        written[name] = p
    return written
