"""Metric tests — each checked against a hand-computed fixture."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from config.settings import AccountRules
from scipy.stats import norm

from gold_bot.backtest import metrics as m


def _trade(net_pnl: float, entry: str = "2021-01-01 09:00", exit: str = "2021-01-01 10:00"):
    return SimpleNamespace(
        net_pnl=net_pnl,
        entry_time=pd.Timestamp(entry, tz="UTC"),
        exit_time=pd.Timestamp(exit, tz="UTC"),
        direction=1,
    )


# --- ratio / curve metrics ------------------------------------------------- #
def test_total_return_and_drawdown() -> None:
    eq = pd.Series([100.0, 120.0, 90.0, 130.0, 80.0])
    dollars, pct = m.total_return(eq)
    assert dollars == pytest.approx(-20.0)
    assert pct == pytest.approx(-0.20)

    mdd_d, mdd_pct = m.max_drawdown(eq)
    assert mdd_d == pytest.approx(50.0)  # 130 -> 80
    assert mdd_pct == pytest.approx(50.0 / 130.0)


def test_cagr_one_year() -> None:
    eq = pd.Series(np.linspace(100.0, 121.0, 252))
    assert m.cagr(eq, periods_per_year=252) == pytest.approx(0.21, rel=1e-3)


def test_sharpe_zero_when_no_variance() -> None:
    assert m.sharpe_ratio(pd.Series([0.01, 0.01, 0.01])) == 0.0


def test_sharpe_matches_formula() -> None:
    r = pd.Series([0.01, 0.02, -0.01, 0.03])
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert m.sharpe_ratio(r) == pytest.approx(expected)


def test_sortino_matches_formula() -> None:
    r = pd.Series([0.02, -0.01, 0.02, -0.01])
    downside = np.minimum(r.to_numpy(), 0.0)
    dd = np.sqrt(np.mean(downside**2))
    expected = r.mean() / dd * np.sqrt(252)
    assert m.sortino_ratio(r) == pytest.approx(expected)


def test_trailing_dd_headroom() -> None:
    eq = pd.Series([50_000.0, 50_500.0, 49_000.0])
    # floor rises to 48,500 after the 50,500 close; 49,000 - 48,500 = 500.
    assert m.trailing_dd_min_headroom(eq, amount=2_000) == pytest.approx(500.0)
    # A breach shows up as negative headroom.
    breach = pd.Series([50_000.0, 47_500.0])
    assert m.trailing_dd_min_headroom(breach, amount=2_000) == pytest.approx(-500.0)


def test_worst_daily_loss() -> None:
    idx = pd.to_datetime(
        ["2021-01-01", "2021-01-01", "2021-01-02", "2021-01-02"], utc=True
    )
    pnl = pd.Series([-10.0, -20.0, 5.0, 5.0], index=idx)
    assert m.worst_daily_loss(pnl) == pytest.approx(30.0)


# --- trade metrics --------------------------------------------------------- #
def test_trade_metrics() -> None:
    trades = [_trade(10), _trade(-5), _trade(20), _trade(-5)]
    assert m.profit_factor(trades) == pytest.approx(30.0 / 10.0)
    assert m.win_rate(trades) == pytest.approx(0.5)
    assert m.payoff_ratio(trades) == pytest.approx(15.0 / 5.0)
    assert m.expectancy(trades) == pytest.approx(5.0)


def test_longest_losing_streak() -> None:
    trades = [_trade(10), _trade(-5), _trade(-5), _trade(-5), _trade(10), _trade(-5)]
    assert m.longest_losing_streak(trades) == 3


def test_avg_trade_duration_hours() -> None:
    trades = [
        _trade(1, "2021-01-01 09:00", "2021-01-01 10:00"),  # 1h
        _trade(1, "2021-01-01 09:00", "2021-01-01 12:00"),  # 3h
    ]
    assert m.avg_trade_duration_hours(trades) == pytest.approx(2.0)


def test_exposure() -> None:
    pos = pd.Series([0, 1, 1, 0, -1], dtype=float)
    assert m.exposure(pos) == pytest.approx(0.6)


def test_profit_factor_no_losses_is_inf() -> None:
    assert m.profit_factor([_trade(10), _trade(5)]) == float("inf")


# --- (de)flated Sharpe ----------------------------------------------------- #
def test_probabilistic_sharpe_ratio() -> None:
    sr_hat, n, skew, kurt = 0.1, 101, 0.0, 3.0
    denom = 1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat**2
    z = (sr_hat - 0.0) * np.sqrt(n - 1.0) / np.sqrt(denom)
    assert m.probabilistic_sharpe_ratio(sr_hat, n, skew, kurt) == pytest.approx(norm.cdf(z))


def test_expected_max_sharpe() -> None:
    gamma = 0.5772156649015329
    std, n = 0.1, 10
    z1 = norm.ppf(1 - 1 / n)
    z2 = norm.ppf(1 - 1 / (n * np.e))
    expected = std * ((1 - gamma) * z1 + gamma * z2)
    assert m.expected_max_sharpe(std, n) == pytest.approx(expected)


def test_deflation_reduces_psr() -> None:
    psr0 = m.probabilistic_sharpe_ratio(0.2, 200, 0.0, 3.0, sr_benchmark=0.0)
    dsr = m.deflated_sharpe_ratio(0.2, 200, 0.0, 3.0, n_trials=50, trial_sharpe_std=0.1)
    assert dsr < psr0  # trying many strategies deflates the result
    # With a single trial the benchmark is 0, so DSR == PSR vs 0.
    dsr1 = m.deflated_sharpe_ratio(0.2, 200, 0.0, 3.0, n_trials=1, trial_sharpe_std=0.1)
    assert dsr1 == pytest.approx(psr0)


# --- Monte-Carlo bust probability ------------------------------------------ #
def test_mc_never_busts_when_all_wins() -> None:
    p = m.monte_carlo_bust_probability([100, 100, 100], 50_000, 2_000, n_paths=500)
    assert p == 0.0


def test_mc_always_busts_on_big_loss() -> None:
    p = m.monte_carlo_bust_probability([-3_000], 50_000, 2_000, n_paths=200)
    assert p == 1.0


def test_mc_probability_in_unit_interval() -> None:
    p = m.monte_carlo_bust_probability(
        [200, -150, 300, -2_100, 100], 50_000, 2_000, n_paths=1_000, method="permutation"
    )
    assert 0.0 <= p <= 1.0


def test_walk_forward_efficiency() -> None:
    assert m.walk_forward_efficiency(0.2, 0.1) == pytest.approx(0.5)
    assert m.walk_forward_efficiency(0.0, 0.1) == 0.0


# --- aggregator ------------------------------------------------------------ #
def test_compute_metrics_end_to_end() -> None:
    from gold_bot.backtest.event_engine import EventConfig, run_event_backtest
    from gold_bot.strategies.base import StrategyResult

    idx = pd.date_range("2021-01-01", periods=60, freq="D", tz="UTC")
    closes = np.linspace(1800, 1860, 60)
    df = pd.DataFrame(
        {"open": closes, "high": closes + 1, "low": closes - 1, "close": closes, "volume": 1.0},
        index=idx,
    )
    sig = StrategyResult(
        signal=pd.Series(1.0, index=idx),
        atr=pd.Series(5.0, index=idx),
        atr_stop_mult=2.0,
        family="test",
        params={},
    )
    res = run_event_backtest(df, sig, EventConfig.from_settings(use_stops=False))
    mets = m.compute_metrics(res, account_rules=AccountRules(), n_trials=20, trial_sharpe_std=0.1)

    assert mets.n_trades >= 1
    assert mets.exposure > 0
    assert not mets.dd_breached
    assert 0.0 <= mets.monte_carlo_bust_prob <= 1.0
    assert isinstance(mets.to_dict(), dict)
    assert mets.total_return_dollars > 0
