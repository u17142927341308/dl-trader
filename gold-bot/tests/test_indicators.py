"""Tests for indicators: NO LOOK-AHEAD + correctness against hand fixtures.

The headline test is ``test_no_lookahead_*``: it proves prefix-stability. If
appending future bars changes a past indicator value, the indicator leaks the
future and the test fails.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from gold_bot.features import indicators as ind

# All single-series indicators that map a price series -> series/frame.
PREFIX_CASES = {
    "sma": lambda s: ind.sma(s, 20),
    "ema": lambda s: ind.ema(s, 20),
    "rolling_std": lambda s: ind.rolling_std(s, 20),
    "rsi": lambda s: ind.rsi(s, 14),
    "bollinger": lambda s: ind.bollinger_bands(s, 20, 2.0),
    "macd": lambda s: ind.macd(s, 12, 26, 9),
}


@pytest.mark.parametrize("name", list(PREFIX_CASES))
def test_no_lookahead_price_indicators(ohlcv: pd.DataFrame, name: str) -> None:
    """Indicator over data[0:t] must equal full-series indicator restricted to [0:t]."""
    fn = PREFIX_CASES[name]
    close = ohlcv["close"]
    cutoff = 400

    full = fn(close)
    prefix = fn(close.iloc[:cutoff])

    full_head = full.iloc[:cutoff]
    # Compare ignoring leading NaNs that both share.
    pd.testing.assert_frame_equal(
        pd.DataFrame(full_head), pd.DataFrame(prefix), check_dtype=False
    )


def test_no_lookahead_ohlc_indicators(ohlcv: pd.DataFrame) -> None:
    """ATR / true_range / donchian use H/L/C; verify prefix-stability too."""
    cutoff = 400
    h, low, c = ohlcv["high"], ohlcv["low"], ohlcv["close"]

    cases = {
        "true_range": lambda h, low, c: ind.true_range(h, low, c),
        "atr": lambda h, low, c: ind.atr(h, low, c, 14),
        "donchian": lambda h, low, c: ind.donchian(h, low, 20),
    }
    for fn in cases.values():
        full = fn(h, low, c)
        prefix = fn(h.iloc[:cutoff], low.iloc[:cutoff], c.iloc[:cutoff])
        pd.testing.assert_frame_equal(
            pd.DataFrame(full.iloc[:cutoff]), pd.DataFrame(prefix), check_dtype=False
        )


def test_sma_correctness() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = ind.sma(s, 3)
    assert math.isnan(out.iloc[0]) and math.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(2.0)  # (1+2+3)/3
    assert out.iloc[3] == pytest.approx(3.0)  # (2+3+4)/3
    assert out.iloc[4] == pytest.approx(4.0)  # (3+4+5)/3


def test_ema_correctness() -> None:
    # adjust=False recursion: ema[t] = a*x[t] + (1-a)*ema[t-1], a = 2/(span+1).
    s = pd.Series([1.0, 2.0, 3.0])
    out = ind.ema(s, 2)  # span=2 -> alpha = 2/3
    a = 2.0 / 3.0
    # min_periods=2 so first value is NaN; seed at index1 = mean of first 2? No:
    # ewm with adjust=False seeds at the first observation, min_periods masks it.
    assert math.isnan(out.iloc[0])
    expected1 = a * 2.0 + (1 - a) * 1.0
    assert out.iloc[1] == pytest.approx(expected1)
    expected2 = a * 3.0 + (1 - a) * expected1
    assert out.iloc[2] == pytest.approx(expected2)


def test_rsi_all_gains_is_100() -> None:
    s = pd.Series(np.arange(1.0, 30.0))  # strictly increasing -> RSI saturates to 100
    out = ind.rsi(s, 14)
    assert out.dropna().iloc[-1] == pytest.approx(100.0)


def test_true_range_uses_prev_close() -> None:
    high = pd.Series([10.0, 12.0])
    low = pd.Series([8.0, 9.0])
    close = pd.Series([9.0, 11.0])
    tr = ind.true_range(high, low, close)
    # bar0: no prev close -> just H-L = 2
    assert tr.iloc[0] == pytest.approx(2.0)
    # bar1: max(12-9, |12-9|, |9-9|) = max(3,3,0) = 3
    assert tr.iloc[1] == pytest.approx(3.0)


def test_donchian_excludes_current_bar() -> None:
    high = pd.Series([1.0, 2.0, 3.0, 10.0])
    low = pd.Series([0.0, 1.0, 2.0, 0.5])
    dc = ind.donchian(high, low, period=2, exclude_current=True)
    # At index 3 the upper channel is max of bars [1,2] highs = max(2,3)=3,
    # NOT including the current bar's 10. That's what lets 10 be a breakout.
    assert dc["upper"].iloc[3] == pytest.approx(3.0)
    assert dc["lower"].iloc[3] == pytest.approx(1.0)


def test_bollinger_band_ordering(ohlcv: pd.DataFrame) -> None:
    bb = ind.bollinger_bands(ohlcv["close"], 20, 2.0).dropna()
    assert (bb["upper"] >= bb["mid"]).all()
    assert (bb["mid"] >= bb["lower"]).all()
