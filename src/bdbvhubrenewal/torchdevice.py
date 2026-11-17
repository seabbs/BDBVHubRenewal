# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""The torch device, threads and Chronos-2 batch every model run uses.

One place for the machine's compute settings, so training, scoring, the
paper and submissions run Chronos-2 the same way. Each is a machine
setting in config/local.env (`workflow/flowlib/localconfig.py`), which
the workflow and scripts export to the process environment:

- `DEVICE`: `auto` (the first of CUDA, MPS and the CPU there is), or a
  torch device such as `cuda`, `mps` or `cpu`.
- `TORCH_THREADS`: torch's CPU threads; unset leaves torch's default.
- `CHRONOS_BATCH`: series in one Chronos-2 forward pass.

Settings are machine settings: none enters a checkpoint's hash.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Final

CHRONOS_BATCH: Final = 256
"""Series in one Chronos-2 forward when `CHRONOS_BATCH` is unset. On an
8 GB GPU, 256 fit and 1,024 do not (group attention grows with the
square of the series in a group)."""

AUTO_ORDER: Final = ("cuda", "mps", "cpu")
"""The order `auto` tries the devices in."""


def pick_device(device: str | None = None) -> str:
    """Return the torch device to run on.

    Parameters
    ----------
    device : str, optional
        A device, or `auto`. None reads `DEVICE` (default `auto`).

    Returns
    -------
    str
        `device` as given, or under `auto` the first of `AUTO_ORDER`
        torch has.
    """
    device = device or os.environ.get("DEVICE") or "auto"
    if device != "auto":
        return device
    import torch

    have: dict[str, Callable[[], bool]] = {
        "cuda": torch.cuda.is_available,
        "mps": torch.backends.mps.is_available,
        "cpu": lambda: True,
    }
    return next(d for d in AUTO_ORDER if have[d]())


def _count(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    n = int(value)
    if n < 1:
        raise ValueError(f"{name} must be a positive count, not {value}")
    return n


def torch_threads() -> int | None:
    """Return torch's CPU threads (`TORCH_THREADS`), None if unset."""
    return _count("TORCH_THREADS")


def apply_threads() -> None:
    """Set torch's CPU threads from `TORCH_THREADS`, if set."""
    n = torch_threads()
    if n is not None:
        import torch

        torch.set_num_threads(n)


def chronos_batch() -> int:
    """Return the series in one Chronos-2 forward (`CHRONOS_BATCH`)."""
    return _count("CHRONOS_BATCH") or CHRONOS_BATCH
