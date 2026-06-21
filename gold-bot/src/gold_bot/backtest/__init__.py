"""Backtesting: event-driven verifier + fast vectorised runner + metrics."""

from .event_engine import BacktestResult, EventConfig, Trade, run_event_backtest
from .vectorbt_runner import FastResult, run_fast_backtest

__all__ = [
    "EventConfig",
    "BacktestResult",
    "Trade",
    "run_event_backtest",
    "FastResult",
    "run_fast_backtest",
]
