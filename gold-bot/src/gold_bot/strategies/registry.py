"""Strategy registry + search-space expansion.

Families self-register via the ``@register`` decorator. The search/orchestrator
(Phase 5) iterates :func:`iter_search_space` to get every (family, params)
combination defined in ``config/search_space.yaml`` and builds a concrete
strategy with :func:`build`.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from .base import Strategy

# Populated by the @register decorator on import of the family modules.
REGISTRY: dict[str, type[Strategy]] = {}

# config/search_space.yaml relative to the repo root (this file is at
# src/gold_bot/strategies/registry.py -> repo root is parents[3]).
_DEFAULT_SEARCH_SPACE = Path(__file__).resolve().parents[3] / "config" / "search_space.yaml"

# Non-family top-level keys in the YAML that must be skipped during expansion.
_RESERVED_KEYS = {"search"}


def register(cls: type[Strategy]) -> type[Strategy]:
    """Class decorator: add a strategy family to the registry by its ``name``."""
    if not getattr(cls, "name", None) or cls.name == "base":
        raise ValueError(f"{cls.__name__} must define a unique class-level `name`")
    if cls.name in REGISTRY and REGISTRY[cls.name] is not cls:
        raise ValueError(f"duplicate strategy family name: {cls.name!r}")
    REGISTRY[cls.name] = cls
    return cls


def build(family: str, params: dict[str, Any] | None = None) -> Strategy:
    """Instantiate a registered strategy family with the given parameters."""
    _ensure_loaded()
    if family not in REGISTRY:
        raise KeyError(f"unknown strategy family {family!r}; have {sorted(REGISTRY)}")
    return REGISTRY[family](**(params or {}))


def load_search_space(path: str | Path | None = None) -> dict[str, Any]:
    """Parse the search-space YAML into a dict."""
    p = Path(path) if path else _DEFAULT_SEARCH_SPACE
    with open(p, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _expand_grid(params: dict[str, list[Any]]) -> Iterator[dict[str, Any]]:
    """Cartesian product of a {param: [values]} grid into individual dicts."""
    keys = list(params)
    for combo in itertools.product(*(params[k] for k in keys)):
        yield dict(zip(keys, combo, strict=True))


def iter_search_space(
    path: str | Path | None = None,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield (family, params) for every combination in the search space.

    Only families that are actually registered are expanded; an unknown family
    in the YAML raises, so typos surface loudly.
    """
    _ensure_loaded()
    space = load_search_space(path)
    for family, block in space.items():
        if family in _RESERVED_KEYS:
            continue
        if family not in REGISTRY:
            raise KeyError(f"search_space.yaml references unknown family {family!r}")
        params = (block or {}).get("params", {})
        grid = {k: (v if isinstance(v, list) else [v]) for k, v in params.items()}
        for combo in _expand_grid(grid):
            yield family, combo


def _ensure_loaded() -> None:
    """Import family modules so their @register side effects run."""
    if REGISTRY:
        return
    from . import (  # noqa: F401  (imported for registration side effects)
        donchian_breakout,
        ema_cross,
        macd_trend,
        rsi_bollinger,
    )


# Trigger registration on import of this module.
_ensure_loaded()
