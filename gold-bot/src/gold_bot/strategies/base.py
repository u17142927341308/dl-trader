"""Strategy abstraction.

Every strategy maps OHLCV bars to a **target position direction** series in
``{-1, 0, +1}`` (short / flat / long). Two hard rules make backtests honest:

1. **Causality.** The decision at bar ``t`` may use indicator values up to and
   including bar ``t`` only. Strategies never peek at the future. (Each family
   is tested for prefix-stability, exactly like the indicators.)

2. **Execution lag is the backtester's job, not the strategy's.** The signal at
   bar ``t`` means "this is the position I want, decided at the close of bar
   ``t``". The event-driven engine (Phase 3) acts on it at the NEXT bar's open.
   Strategies therefore must NOT pre-shift their own signals — doing so twice
   would hide look-ahead. Keeping the lag in one place keeps it auditable.

The ATR series and the ``atr_stop_mult`` parameter travel alongside the signal
so the risk/event layer can size stops and targets without recomputing them.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..features import indicators as ind

# Parameter spec entry: (python_type, default). Used for light validation so a
# typo'd parameter from the YAML search space fails loudly instead of silently.
ParamSpec = dict[str, tuple[type, Any]]


@dataclass
class StrategyResult:
    """Output of a strategy over a full price frame."""

    signal: pd.Series  # target direction in {-1, 0, +1}, indexed like the input
    atr: pd.Series  # ATR aligned to the same index (for stop sizing)
    atr_stop_mult: float  # multiple of ATR for protective stop
    family: str
    params: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def strategy_id(self) -> str:
        parts = "_".join(f"{k}{v}" for k, v in sorted(self.params.items()))
        return f"{self.family}__{parts}"


def signal_to_trades(signal: pd.Series) -> pd.DataFrame:
    """Derive long/short entry & exit booleans from a target-position series.

    Provided for vectorbt compatibility (Phase 3). A transition into a state is
    an entry; a transition out of it is an exit. Uses ``shift(1)`` (past), so it
    stays causal.
    """
    prev = signal.shift(1).fillna(0)
    return pd.DataFrame(
        {
            "long_entries": (signal == 1) & (prev != 1),
            "long_exits": (signal != 1) & (prev == 1),
            "short_entries": (signal == -1) & (prev != -1),
            "short_exits": (signal != -1) & (prev == -1),
        }
    )


class Strategy(abc.ABC):
    """Abstract base for a parametrised strategy family."""

    #: family name, must match the registry key and the search_space.yaml key.
    name: str = "base"
    #: declared parameters with (type, default).
    param_spec: ParamSpec = {}

    def __init__(self, **params: Any) -> None:
        merged: dict[str, Any] = {}
        for key, (typ, default) in self.param_spec.items():
            value = params.get(key, default)
            try:
                merged[key] = typ(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{self.name}: parameter {key!r} expected {typ.__name__}, got {value!r}"
                ) from exc
        unknown = set(params) - set(self.param_spec)
        if unknown:
            raise ValueError(f"{self.name}: unknown parameters {sorted(unknown)}")
        self.params = merged

    # --- public API -------------------------------------------------------
    def generate(self, df: pd.DataFrame) -> StrategyResult:
        """Run the strategy over ``df`` and return a :class:`StrategyResult`."""
        signal = self.compute_signal(df).astype(float)
        signal = signal.reindex(df.index).fillna(0.0)
        atr_period = int(self.params.get("atr_period", 14))
        atr = ind.atr(df["high"], df["low"], df["close"], atr_period)
        return StrategyResult(
            signal=signal,
            atr=atr,
            atr_stop_mult=float(self.params.get("atr_stop_mult", 2.0)),
            family=self.name,
            params=dict(self.params),
        )

    # --- to implement -----------------------------------------------------
    @abc.abstractmethod
    def compute_signal(self, df: pd.DataFrame) -> pd.Series:
        """Return the target-position series in {-1, 0, +1}. MUST be causal."""
        raise NotImplementedError

    # --- helpers ----------------------------------------------------------
    @staticmethod
    def _stateful_fill(
        enter_long: pd.Series,
        exit_long: pd.Series,
        enter_short: pd.Series,
        exit_short: pd.Series,
    ) -> pd.Series:
        """Walk forward turning entry/exit triggers into a held position series.

        A pure forward pass: position at bar ``t`` depends only on triggers at
        ``<= t`` and the prior position. Guaranteed causal. Long has priority on
        simultaneous opposite triggers (deterministic tie-break).
        """
        el = enter_long.to_numpy(dtype=bool)
        xl = exit_long.to_numpy(dtype=bool)
        es = enter_short.to_numpy(dtype=bool)
        xs = exit_short.to_numpy(dtype=bool)
        pos = np.zeros(len(el), dtype=np.int8)
        cur = 0
        for i in range(len(el)):
            if cur == 1 and xl[i]:
                cur = 0
            elif cur == -1 and xs[i]:
                cur = 0
            if cur == 0:
                if el[i]:
                    cur = 1
                elif es[i]:
                    cur = -1
            pos[i] = cur
        return pd.Series(pos, index=enter_long.index, dtype=float)
