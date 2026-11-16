"""Reading score and forecast rows: numbers and a model's source."""

from __future__ import annotations

from typing import Any


def num(x: Any) -> float | None:
    """Return `x` as a float, or None if it is not a number."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def source(model: str) -> str | None:
    """Return a model's release source (daily or weekly), if it has one."""
    if "daily" in model:
        return "daily"
    if "weekly" in model:
        return "weekly"
    return None


def sources(model: str) -> list[str]:
    """Return the sources a model is drawn on: its own, or both if none."""
    own = source(model)
    return [own] if own else ["daily", "weekly"]
