"""Strategy families and the registry that exposes them to the search."""

from .base import Strategy, StrategyResult, signal_to_trades
from .registry import REGISTRY, build, iter_search_space, register

__all__ = [
    "Strategy",
    "StrategyResult",
    "signal_to_trades",
    "REGISTRY",
    "register",
    "build",
    "iter_search_space",
]
