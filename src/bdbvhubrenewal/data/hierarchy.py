# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Location hierarchies read from ISO 3166-2 style codes."""

from __future__ import annotations

__all__ = ["depth", "parent"]


def parent(code: str) -> str | None:
    """Return the parent of a location code."""
    raise NotImplementedError


def depth(code: str) -> int:
    """Return how far a location code sits below the root."""
    raise NotImplementedError
