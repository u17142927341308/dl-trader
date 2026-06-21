"""Tests for the walk-forward search, gating, orchestrator, and JSON export."""

from __future__ import annotations

import json

from gold_bot.backtest.metrics import Metrics
from gold_bot.reporting.export import export_all
from gold_bot.run_research import synthetic_5m
from gold_bot.search.gating import evaluate_gate
from gold_bot.search.orchestrator import run_search
from gold_bot.search.walk_forward import WFOConfig, make_windows
from gold_bot.signals.schema import (
    EquityCurveArtifact,
    MetricsArtifact,
    SignalArtifact,
    StatusArtifact,
    TradesArtifact,
    WalkForwardArtifact,
)


def test_make_windows_rolling_and_anchored() -> None:
    rolling = make_windows(2000, WFOConfig("rolling", train_bars=750, test_bars=250, step_bars=250))
    assert len(rolling) == 5
    # rolling train windows slide; anchored ones all start at 0.
    anchored = make_windows(2000, WFOConfig("anchored", train_bars=750, test_bars=250, step_bars=250))
    assert all(w.train.start == 0 for w in anchored)
    assert rolling[1].train.start == 250


def _metrics(**over) -> Metrics:
    base = dict(
        total_return_dollars=1000.0, total_return_pct=0.1, cagr=0.1, sharpe=1.5,
        sortino=2.0, calmar=1.0, max_dd_dollars=500.0, max_dd_pct=0.05,
        trailing_dd_min_headroom=900.0, worst_daily_loss=400.0, profit_factor=1.6,
        win_rate=0.55, payoff_ratio=1.3, expectancy=12.0, avg_trade_duration_hours=48.0,
        exposure=0.6, n_trades=150, longest_losing_streak=5, deflated_sharpe=0.99,
        walk_forward_efficiency=0.7, monte_carlo_bust_prob=0.01, dd_breached=False,
    )
    base.update(over)
    return Metrics(**base)


def test_gate_accepts_strong_and_rejects_weak() -> None:
    good = evaluate_gate(_metrics(), n_trials=20, trial_sharpe_std=0.1, walk_forward_efficiency=0.7)
    assert good.passed

    weak = evaluate_gate(
        _metrics(sharpe=0.3, profit_factor=1.0), n_trials=20, trial_sharpe_std=0.1,
        walk_forward_efficiency=0.7,
    )
    assert not weak.passed
    assert "oos_sharpe" in weak.failures


def test_gate_rejects_dd_breach() -> None:
    res = evaluate_gate(
        _metrics(dd_breached=True), n_trials=5, trial_sharpe_std=0.1, walk_forward_efficiency=0.7
    )
    assert not res.passed
    assert "trailing_dd_survival" in res.failures


def test_gate_rejects_bust_probability() -> None:
    res = evaluate_gate(
        _metrics(monte_carlo_bust_prob=0.2), n_trials=5, trial_sharpe_std=0.1,
        walk_forward_efficiency=0.7,
    )
    assert "monte_carlo_survival" in res.failures


def test_run_search_smoke_and_export(tmp_path) -> None:
    # Small intraday synthetic dataset + small windows so the smoke test is fast.
    df = synthetic_5m(days=12, seed=3)
    outcome = run_search(
        df,
        wfo_cfg=WFOConfig("rolling", train_bars=700, test_bars=300, step_bars=300),
        periods_per_year=69552,
        max_trials_per_family=8,
    )
    assert outcome.n_trials > 0
    assert isinstance(outcome.accepted, bool)
    # Every trial was logged.
    assert len(outcome.ledger.trials) == outcome.n_trials

    written = export_all(outcome, df, tmp_path, timeframe="15min")
    assert "status.json" in written and "signals.json" in written

    # Exported JSON must validate against the shared schema.
    StatusArtifact(**json.loads((tmp_path / "status.json").read_text()))
    EquityCurveArtifact(**json.loads((tmp_path / "equity_curve.json").read_text()))
    MetricsArtifact(**json.loads((tmp_path / "metrics.json").read_text()))
    TradesArtifact(**json.loads((tmp_path / "trades.json").read_text()))
    WalkForwardArtifact(**json.loads((tmp_path / "walkforward.json").read_text()))
    SignalArtifact(**json.loads((tmp_path / "signals.json").read_text()))


def test_search_no_windows_is_honest() -> None:
    df = synthetic_5m(days=2, seed=1)
    outcome = run_search(df, wfo_cfg=WFOConfig(train_bars=5000, test_bars=2000))
    assert not outcome.accepted
    assert "insufficient" in outcome.note
