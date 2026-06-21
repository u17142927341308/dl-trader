"""Walk-forward optimisation (WFO) splitter + candidate evaluation.

A candidate (family + fixed params) is scored by its **out-of-sample**
performance, never in-sample. We slide rolling (or anchored) windows over the
data; for each window the strategy is evaluated on the test segment only, and
the test segments are chained into one aggregate OOS equity curve. The in-sample
(train) segments are chained too, purely so we can measure walk-forward
efficiency (OOS return / IS return) — a curve-fit detector.

Why this is honest:
    * signals are causal, so slicing a window never leaks the future;
    * the candidate is judged on data its parameters were not chosen to fit;
    * the prop-firm trailing-DD is applied to the *chained* OOS equity (in
      :func:`gold_bot.backtest.metrics`), not reset per window, so a strategy
      cannot hide a blow-up by spreading it across windows.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..backtest.event_engine import BacktestResult, EventConfig, run_event_backtest
from ..backtest.metrics import trailing_dd_min_headroom
from ..strategies.base import StrategyResult
from ..strategies.registry import build


@dataclass
class WFOConfig:
    scheme: str = "rolling"  # "rolling" | "anchored"
    train_bars: int = 750
    test_bars: int = 250
    step_bars: int = 250


@dataclass
class Window:
    train: slice
    test: slice


def make_windows(n: int, cfg: WFOConfig) -> list[Window]:
    """Build train/test index-position windows over ``n`` bars."""
    windows: list[Window] = []
    start = 0
    while start + cfg.train_bars + cfg.test_bars <= n:
        train_start = 0 if cfg.scheme == "anchored" else start
        train_end = start + cfg.train_bars
        test_end = train_end + cfg.test_bars
        windows.append(Window(slice(train_start, train_end), slice(train_end, test_end)))
        start += cfg.step_bars
    return windows


def _slice_result(result: StrategyResult, sl: slice, idx: pd.Index) -> StrategyResult:
    sub = idx[sl]
    return StrategyResult(
        signal=result.signal.loc[sub],
        atr=result.atr.loc[sub],
        atr_stop_mult=result.atr_stop_mult,
        family=result.family,
        params=result.params,
    )


def _chain(
    df: pd.DataFrame,
    result: StrategyResult,
    slices: list[slice],
    event_cfg: EventConfig,
    trailing_dd: float,
) -> BacktestResult:
    """Run the strategy on each window slice and chain the segments together."""
    eqs, pnls, poss = [], [], []
    trades = []
    account = event_cfg.rules.account_size
    cum = 0.0
    for sl in slices:
        sub_df = df.iloc[sl]
        if len(sub_df) < 2:
            continue
        sub_res = _slice_result(result, sl, df.index)
        bt = run_event_backtest(sub_df, sub_res, event_cfg)
        # Chain equity so segments continue from where the last left off.
        seg_pnl = bt.pnl
        eqs.append(account + cum + seg_pnl.cumsum())
        pnls.append(seg_pnl)
        poss.append(bt.position)
        trades.extend(bt.trades)
        cum += seg_pnl.sum()

    if not eqs:
        empty = pd.Series(dtype=float)
        return BacktestResult(
            equity=pd.Series([account], dtype=float),
            pnl=empty,
            position=empty,
            trades=[],
            final_equity=account,
        )

    equity = pd.concat(eqs)
    pnl = pd.concat(pnls)
    position = pd.concat(poss)
    min_headroom = trailing_dd_min_headroom(equity, trailing_dd)
    return BacktestResult(
        equity=equity,
        pnl=pnl,
        position=position,
        trades=trades,
        dd_dead=bool(min_headroom < 0),
        daily_halt_days=0,
        final_equity=float(equity.iloc[-1]),
        total_gross=float(sum(t.gross_pnl for t in trades)),
        total_costs=float(sum(t.costs for t in trades)),
        total_net=float(equity.iloc[-1] - account),
    )


@dataclass
class CandidateEval:
    family: str
    params: dict
    oos: BacktestResult
    is_: BacktestResult


def evaluate_candidate(
    df: pd.DataFrame,
    family: str,
    params: dict,
    windows: list[Window],
    event_cfg: EventConfig | None = None,
) -> CandidateEval:
    """Evaluate one (family, params) candidate across all walk-forward windows.

    Windows are run with prop enforcement OFF (clean strategy PnL); the
    trailing-DD survival check is applied to the *chained* OOS equity instead,
    which is stricter than per-window resets.
    """
    cfg = event_cfg or EventConfig.from_settings()
    cfg = EventConfig(
        instrument=cfg.instrument,
        costs=cfg.costs,
        rules=cfg.rules,
        fixed_size=cfg.fixed_size,
        use_stops=True,
        enforce_prop=False,
    )
    strat = build(family, params).generate(df)
    oos = _chain(df, strat, [w.test for w in windows], cfg, cfg.rules.trailing_drawdown)
    is_ = _chain(df, strat, [w.train for w in windows], cfg, cfg.rules.trailing_drawdown)
    return CandidateEval(family=family, params=params, oos=oos, is_=is_)
