"""Tests for strategy families, the registry, and search-space expansion.

The critical test is ``test_no_lookahead_signal``: a strategy's target-position
series must be prefix-stable, exactly like the indicators it is built from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold_bot.strategies import (
    REGISTRY,
    build,
    iter_search_space,
    signal_to_trades,
)

FAMILIES = ["ema_cross", "rsi_bollinger", "donchian_breakout", "macd_trend"]


def test_registry_has_all_families() -> None:
    for fam in FAMILIES:
        assert fam in REGISTRY


def test_build_unknown_family_raises() -> None:
    with pytest.raises(KeyError):
        build("does_not_exist")


def test_unknown_param_raises() -> None:
    with pytest.raises(ValueError):
        build("ema_cross", {"nonsense": 5})


@pytest.mark.parametrize("family", FAMILIES)
def test_signal_in_valid_set(ohlcv: pd.DataFrame, family: str) -> None:
    res = build(family).generate(ohlcv)
    assert res.signal.index.equals(ohlcv.index)
    assert not res.signal.isna().any()
    assert set(np.unique(res.signal.to_numpy())).issubset({-1.0, 0.0, 1.0})


@pytest.mark.parametrize("family", FAMILIES)
def test_no_lookahead_signal(ohlcv: pd.DataFrame, family: str) -> None:
    cutoff = 400
    strat = build(family)
    full = strat.generate(ohlcv).signal
    prefix = strat.generate(ohlcv.iloc[:cutoff]).signal
    pd.testing.assert_series_equal(
        full.iloc[:cutoff], prefix, check_names=False, check_freq=False
    )


def test_ema_cross_follows_trend() -> None:
    idx = pd.date_range("2020-01-01", periods=300, freq="D", tz="UTC")
    up = pd.Series(np.linspace(1000, 2000, 300), index=idx)
    df_up = pd.DataFrame(
        {"open": up, "high": up + 1, "low": up - 1, "close": up, "volume": 1.0}
    )
    sig_up = build("ema_cross", {"fast": 10, "slow": 50}).generate(df_up).signal
    assert sig_up.iloc[-1] == 1.0  # rising market -> long

    down = pd.Series(np.linspace(2000, 1000, 300), index=idx)
    df_down = df_up.assign(open=down, high=down + 1, low=down - 1, close=down)
    sig_down = build("ema_cross", {"fast": 10, "slow": 50}).generate(df_down).signal
    assert sig_down.iloc[-1] == -1.0  # falling market -> short


def test_degenerate_ema_params_flat() -> None:
    idx = pd.date_range("2020-01-01", periods=120, freq="D", tz="UTC")
    px = pd.Series(np.linspace(1000, 1100, 120), index=idx)
    df = pd.DataFrame({"open": px, "high": px + 1, "low": px - 1, "close": px, "volume": 1.0})
    sig = build("ema_cross", {"fast": 50, "slow": 20}).generate(df).signal
    assert (sig == 0.0).all()  # fast >= slow -> no signal


def test_signal_to_trades_transitions() -> None:
    sig = pd.Series([0, 1, 1, 0, -1, -1, 0], dtype=float)
    t = signal_to_trades(sig)
    assert t["long_entries"].tolist() == [False, True, False, False, False, False, False]
    assert t["long_exits"].tolist() == [False, False, False, True, False, False, False]
    assert t["short_entries"].tolist() == [False, False, False, False, True, False, False]
    assert t["short_exits"].tolist() == [False, False, False, False, False, False, True]


def test_iter_search_space_expands_grids() -> None:
    combos = list(iter_search_space())
    # Every combo must be buildable and reference a known family.
    families_seen = set()
    for fam, params in combos:
        families_seen.add(fam)
        strat = build(fam, params)
        assert strat.name == fam
    assert families_seen == set(FAMILIES)

    # ema_cross grid in the YAML: fast[4] x slow[3] x atr_period[1] x stop[2] = 24.
    ema_count = sum(1 for fam, _ in combos if fam == "ema_cross")
    assert ema_count == 24


def test_strategy_id_is_stable(ohlcv: pd.DataFrame) -> None:
    res1 = build("donchian_breakout", {"entry_lookback": 40}).generate(ohlcv)
    res2 = build("donchian_breakout", {"entry_lookback": 40}).generate(ohlcv)
    assert res1.strategy_id == res2.strategy_id
    assert res1.strategy_id.startswith("donchian_breakout__")
