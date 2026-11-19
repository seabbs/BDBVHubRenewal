# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""An offloaded checkpoint's held-out losses, kept beside its settings.

The workflow's `evaluate_checkpoint` rule writes them as `SIDECAR`
(workflow/flowlib/evaluations.py has the layout); readers take a
checkpoint's settings through `with_sidecar`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

SIDECAR: Final = "held-out.json"
"""The sidecar's file name, beside `settings.json` and in the release."""

RESULT_ONLY: Final = ("kinds", "device", "seconds", "n_outbreaks", "step")
"""Fields of an evaluation's result about the run, not losses."""


def with_sidecar(
    settings: Mapping[str, Any], sidecar: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Return `settings` with the sidecar's losses added, settings first."""
    if not sidecar:
        return dict(settings)
    losses = {k: v for k, v in sidecar.items() if k not in RESULT_ONLY}
    return {**losses, **settings}
