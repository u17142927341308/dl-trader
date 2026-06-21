"""Tests for position sizing and the DynamicRiskManager circuit breaker."""

from __future__ import annotations

import numpy as np
import pandas as pd
from config.settings import AccountRules, get_settings

from gold_bot.backtest.event_engine import EventConfig, run_event_backtest
from gold_bot.risk.manager import DynamicRiskManager
from gold_bot.risk.position_sizing import atr_risk_per_contract, contracts_for_budget
from gold_bot.strategies.base import StrategyResult

MGC = get_settings().instrument


def _mgr(**over) -> DynamicRiskManager:
    rules = AccountRules(account_size=50_000, trailing_drawdown=2_000, daily_loss_limit=1_000, max_contracts=10)
    return DynamicRiskManager(rules=rules, instrument=MGC, **over)


def test_position_sizing_primitives() -> None:
    # ATR 20, 2x stop, $10/pt -> $400 risk per MGC.
    assert atr_risk_per_contract(20.0, 2.0, 10.0) == 400.0
    assert contracts_for_budget(500.0, 400.0, 10) == 1
    assert contracts_for_budget(1000.0, 400.0, 10) == 2
    assert contracts_for_budget(1000.0, 400.0, max_contracts=1) == 1  # plan cap
    assert contracts_for_budget(100.0, 400.0, 10) == 0  # too volatile to size


def test_vol_target_size_full_headroom() -> None:
    mgr = _mgr()  # budget = 0.5 * 1000 = $500
    # ATR 20 -> rpc 400 -> 1 contract at full headroom.
    assert mgr.size(20.0, 2.0, equity=50_000, headroom=2_000) == 1
    # Lower ATR -> smaller risk per contract -> more contracts.
    assert mgr.size(5.0, 2.0, equity=50_000, headroom=2_000) == 5


def test_circuit_breaker_flat_on_low_headroom() -> None:
    mgr = _mgr()  # flat threshold = 0.25 * 2000 = 500
    assert mgr.size(5.0, 2.0, equity=48_400, headroom=400) == 0


def test_headroom_scaling_reduces_size() -> None:
    mgr = _mgr()  # reduce below 2000, flat below 500
    full = mgr.size(5.0, 2.0, equity=50_000, headroom=2_000)  # factor 1.0 -> 5
    # headroom 1250 -> factor = (1250-500)/(2000-500) = 0.5 -> 2
    half = mgr.size(5.0, 2.0, equity=49_250, headroom=1_250)
    assert half < full
    assert half == 2


def test_daily_loss_scaling() -> None:
    mgr = _mgr()
    # Half the daily limit already lost -> size scaled by ~0.5.
    full = mgr.size(5.0, 2.0, equity=50_000, headroom=2_000, day_loss=0.0)
    reduced = mgr.size(5.0, 2.0, equity=50_000, headroom=2_000, day_loss=500.0)
    assert reduced < full


def test_zero_atr_is_flat() -> None:
    assert _mgr().size(0.0, 2.0, equity=50_000, headroom=2_000) == 0


def test_engine_uses_risk_manager_for_sizing() -> None:
    # An always-long signal with a fixed ATR should size via the manager.
    idx = pd.date_range("2021-01-01", periods=40, freq="D", tz="UTC")
    closes = np.linspace(1800, 1820, 40)
    df = pd.DataFrame(
        {"open": closes, "high": closes + 1, "low": closes - 1, "close": closes, "volume": 1.0},
        index=idx,
    )
    sig = StrategyResult(
        signal=pd.Series(1.0, index=idx), atr=pd.Series(5.0, index=idx),
        atr_stop_mult=2.0, family="t", params={},
    )
    mgr = _mgr()
    cfg = EventConfig(
        instrument=MGC, costs=get_settings().cost_model, rules=mgr.rules,
        use_stops=False, enforce_prop=False, risk_manager=mgr,
    )
    res = run_event_backtest(df, sig, cfg)
    # rpc = 5*2*10 = 100, budget 500 -> 5 contracts at full headroom.
    assert res.trades[0].size == 5
