"""Event engine tests + reconciliation with the fast vectorised runner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from config.settings import AccountRules, CostModel, get_settings

from gold_bot.backtest.event_engine import EventConfig, run_event_backtest
from gold_bot.backtest.vectorbt_runner import run_fast_backtest
from gold_bot.strategies import build
from gold_bot.strategies.base import StrategyResult

MGC = get_settings().instrument


def _const_signal(df: pd.DataFrame, value: float, atr_val: float = 5.0) -> StrategyResult:
    return StrategyResult(
        signal=pd.Series(value, index=df.index),
        atr=pd.Series(atr_val, index=df.index),
        atr_stop_mult=2.0,
        family="test",
        params={},
    )


def _frame(prices: list[tuple[float, float, float, float]], freq: str = "D") -> pd.DataFrame:
    idx = pd.date_range("2021-01-01", periods=len(prices), freq=freq, tz="UTC")
    arr = np.array(prices, dtype=float)
    return pd.DataFrame(
        {"open": arr[:, 0], "high": arr[:, 1], "low": arr[:, 2], "close": arr[:, 3], "volume": 1.0},
        index=idx,
    )


def test_long_in_uptrend_is_profitable() -> None:
    # Steady uptrend, always long, no breaches.
    closes = np.linspace(1800, 1850, 60)
    df = _frame([(c, c + 1, c - 1, c) for c in closes])
    res = run_event_backtest(df, _const_signal(df, 1.0), EventConfig.from_settings(use_stops=False))
    assert not res.dd_dead
    assert res.total_net > 0
    assert res.final_equity > get_settings().account_rules.account_size


def test_costs_make_net_below_gross() -> None:
    closes = np.linspace(1800, 1850, 30)
    df = _frame([(c, c + 1, c - 1, c) for c in closes])
    res = run_event_backtest(df, _const_signal(df, 1.0), EventConfig.from_settings(use_stops=False))
    assert res.total_costs > 0
    assert res.total_net < res.total_gross


def test_trailing_drawdown_kills_account() -> None:
    # Long position, price gaps down 210 points -> $2,100 loss on 1 MGC,
    # below the $2,000 trailing floor (daily limit lifted to isolate the DD).
    rules = AccountRules(account_size=50_000, trailing_drawdown=2_000, daily_loss_limit=10_000_000)
    cfg = EventConfig(instrument=MGC, costs=CostModel(), rules=rules, use_stops=False)
    df = _frame([
        (1800, 1801, 1799, 1800),  # bar0: establish, signal will be long next
        (1800, 1800, 1590, 1595),  # bar1: gap down through the floor
        (1595, 1596, 1594, 1595),
    ])
    res = run_event_backtest(df, _const_signal(df, 1.0), cfg)
    assert res.dd_dead
    assert res.dd_breach_time == df.index[1]
    # Equity is frozen after death.
    assert res.equity.iloc[-1] == res.equity.iloc[1]


def test_daily_loss_halts_trading() -> None:
    # Intraday (hourly) bars in one day; long position loses $1,000+ -> halt.
    rules = AccountRules(account_size=50_000, trailing_drawdown=10_000_000, daily_loss_limit=1_000)
    cfg = EventConfig(instrument=MGC, costs=CostModel(), rules=rules, use_stops=False)
    df = _frame(
        [
            (1800, 1801, 1799, 1800),
            (1800, 1801, 1799, 1800),  # go long here
            (1800, 1800, 1690, 1695),  # -105 pts => -$1,050 on 1 MGC -> breach
            (1695, 1696, 1694, 1695),
        ],
        freq="h",
    )
    res = run_event_backtest(df, _const_signal(df, 1.0), cfg)
    assert res.daily_halt_days == 1
    assert not res.dd_dead
    # A daily-halt trade exists and trading is flat afterwards in the same day.
    assert any(t.reason == "daily_halt" for t in res.trades)


@pytest.mark.parametrize("family", ["ema_cross", "donchian_breakout", "macd_trend"])
def test_reconcile_event_vs_fast(ohlcv: pd.DataFrame, family: str) -> None:
    # With stops and prop rules disabled, the event engine must reduce to the
    # fast vectorised model — to the cent.
    strat = build(family).generate(ohlcv)
    cfg = EventConfig.from_settings(use_stops=False, enforce_prop=False, fixed_size=1)
    ev = run_event_backtest(ohlcv, strat, cfg)
    fast = run_fast_backtest(ohlcv, strat, size=1)

    assert ev.n_trades == fast.n_trades
    assert ev.total_net == pytest.approx(fast.total_net, abs=1e-6)
    np.testing.assert_allclose(ev.equity.to_numpy(), fast.equity.to_numpy(), atol=1e-6)
