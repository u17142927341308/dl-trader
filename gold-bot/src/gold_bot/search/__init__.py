"""Disciplined, overfitting-resistant strategy search."""

from .gating import GateConfig, GateResult, evaluate_gate
from .optimizer import TrialResult, run_grid_search
from .orchestrator import SearchOutcome, run_search
from .walk_forward import WFOConfig, evaluate_candidate, make_windows

__all__ = [
    "WFOConfig",
    "make_windows",
    "evaluate_candidate",
    "TrialResult",
    "run_grid_search",
    "GateConfig",
    "GateResult",
    "evaluate_gate",
    "SearchOutcome",
    "run_search",
]
