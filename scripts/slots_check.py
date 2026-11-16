# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Check that a per-series slots checkpoint ties lags to their place.

Stub: the tests in `tests/test_slots_check.py` pin the behaviour.
"""

from __future__ import annotations


def model_ids(model, block):
    """Return each row's slot as inference assigns it, None without slots."""
    raise NotImplementedError


def predict(model, block):
    """Return the loss `[n, P]` and the scaled quantiles `[n, Q, P]`."""
    raise NotImplementedError


def output_change(given, other, d, h):
    """Return the max and mean absolute change of the scaled quantiles."""
    raise NotImplementedError


def synthetic_adapter(model, out, n_slots, seed, scale):
    """Save an adapter whose LoRA is a no-op and whose slots are random."""
    raise NotImplementedError


def germany_blocks(path, per_source, max_rows, seed):
    """Return a sample of the German states blocks with start and horizon."""
    raise NotImplementedError
