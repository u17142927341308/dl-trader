"""Turn the accepted strategy + latest bars into an advisory signal.

Architected so a future Tradovate-API execution module could consume the same
:class:`SignalArtifact` object. It does NOT place orders — signals are advisory
and displayed only.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pandas as pd
from config.settings import Settings, get_settings

from ..risk.manager import DynamicRiskManager
from ..strategies.base import make_strategy_id
from ..strategies.registry import build
from .schema import SignalArtifact

_TF_HOURS = {"1d": 24, "1h": 1, "1wk": 168}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def build_signal(
    outcome: object,  # search.orchestrator.SearchOutcome
    df: pd.DataFrame,
    timeframe: str = "1d",
    settings: Settings | None = None,
) -> SignalArtifact:
    """Produce the latest signal for the accepted strategy (FLAT if none)."""
    s = settings or get_settings()
    inst = s.instrument
    headroom = s.account_rules.account_size - (
        s.account_rules.account_size - s.account_rules.trailing_drawdown
    )  # = trailing_drawdown at the start; refined below if equity known

    if not getattr(outcome, "accepted", False) or outcome.best is None:
        return SignalArtifact(
            generated_at=_now_iso(),
            instrument=inst.symbol,
            timeframe=timeframe,
            signal="FLAT",
            account_headroom_to_trailing_dd=headroom,
            confidence_notes="No accepted strategy — search ongoing or none found.",
            auto_execution=False,
        )
    return signal_from_params(
        outcome.best.family, outcome.best.params, df, timeframe=timeframe, settings=s
    )


def signal_from_params(
    family: str,
    params: dict,
    df: pd.DataFrame,
    timeframe: str = "1d",
    settings: Settings | None = None,
) -> SignalArtifact:
    """Build a signal from a stored accepted strategy (used by the signals job)."""
    s = settings or get_settings()
    inst = s.instrument
    headroom = s.account_rules.trailing_drawdown

    strat = build(family, params).generate(df)
    direction = int(strat.signal.iloc[-1])
    atr = float(strat.atr.iloc[-1])
    entry = float(df["close"].iloc[-1])
    side = "LONG" if direction > 0 else ("SHORT" if direction < 0 else "FLAT")

    if direction == 0 or atr <= 0 or math.isnan(atr):
        return SignalArtifact(
            generated_at=_now_iso(),
            instrument=inst.symbol,
            timeframe=timeframe,
            signal="FLAT",
            entry=entry,
            account_headroom_to_trailing_dd=headroom,
            strategy_id=make_strategy_id(family, params),
            confidence_notes=f"Strategy flat at last bar. ATR={atr:.2f}.",
            auto_execution=False,
        )

    stop_dist = atr * strat.atr_stop_mult
    stop = entry - direction * stop_dist
    take_profit = entry + direction * stop_dist * 2.0  # 2R target
    risk_per_contract = stop_dist * inst.point_value
    # Same DynamicRiskManager the backtest uses (what you test is what you trade).
    # At signal time we assume a fresh account (full trailing-DD headroom).
    manager = DynamicRiskManager.from_settings(s)
    size = manager.size(
        atr,
        strat.atr_stop_mult,
        equity=s.account_rules.account_size,
        headroom=s.account_rules.trailing_drawdown,
        day_loss=0.0,
    )
    valid_hours = _TF_HOURS.get(timeframe, 24)

    return SignalArtifact(
        generated_at=_now_iso(),
        instrument=inst.symbol,
        timeframe=timeframe,
        signal=side,
        entry=round(entry, 2),
        stop_loss=round(stop, 2),
        take_profit=round(take_profit, 2),
        position_size_contracts=size,
        risk_dollars=round(size * risk_per_contract, 2),
        account_headroom_to_trailing_dd=headroom,
        confidence_notes=(
            f"{family} fired {side}. ATR={atr:.2f}, "
            f"stop={strat.atr_stop_mult:g}xATR, 2R target."
        ),
        strategy_id=make_strategy_id(family, params),
        valid_until=(datetime.now(UTC) + timedelta(hours=valid_hours))
        .isoformat()
        .replace("+00:00", "Z"),
        auto_execution=False,
    )
