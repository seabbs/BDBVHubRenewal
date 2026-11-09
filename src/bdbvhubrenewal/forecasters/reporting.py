# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""A renewal fit's own counts as reported by a later release.

A stored fit is keyed by its inputs, so it cannot be asked for the counts
as reported by releases after its origin. It stores instead, per
trajectory, the cases it expects to be reported $j$ days after its
cut-off (`bdbvhubrenewal.model.reporting_curves`), and scoring draws the
counts any later release shows from them (`as_reported_samples`), as the
model's own observation process does:

- known onsets: the vintage count, or the trajectory's draw of it where
  the vintage has none, plus $\text{NB}(\mu_j, \phi)$, with $\mu_j$ the
  known cases expected between the cut-off and $j$ days on;
- imputed onsets: $\text{NB}(v + \mu_j, \phi)$ around the vintage
  count $v$, or the trajectory's draw plus $\text{NB}(\mu_j, \phi)$
  where the vintage has none.

This keeps each trajectory's delay, publication lag and missing-onset
fraction, and the fitted negative binomial noise. NumPy only: scoring
does not need the model.

The expected cases are stored as the share of the eventual count
reported by each day, in 1/255 steps, with the eventual count, so a
fit's block is about the size of its samples.
"""

from __future__ import annotations

import base64
import datetime as dt
import zlib
from collections.abc import Mapping
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

__all__ = [
    "FLOOR",
    "NEGLIGIBLE",
    "as_reported_samples",
    "decode",
    "draw_reported",
    "encode",
]

FLOOR: Final = 1e-6
"""Smallest negative binomial mean, as the observation process."""

_STEPS: Final = 255

NEGLIGIBLE: Final = 0.01
"""Expected cases still to come below which a cell is stored as reported
in full at once: its share curve would cost as much as any other."""
_STREAMS: Final = ("known", "imputed")


def _neg_binomial(
    mean: NDArray[np.float64],
    phi: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Draw NB2 counts (variance `mean + mean**2 / phi`), gamma-Poisson."""
    mean = np.maximum(mean, FLOOR)
    rate = rng.gamma(phi, mean / phi)
    return rng.poisson(rate).astype(float)


def draw_reported(
    curves: Mapping[str, Any], day: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Draw the counts reported `day` days after the fit's cut-off.

    Parameters
    ----------
    curves : mapping
        `reporting_curves` output, or `decode` of a stored block.
    day : int
        Days after the cut-off. Past the curves' last day, every report
        is in.
    rng : numpy.random.Generator
        Random numbers.

    Returns
    -------
    numpy.ndarray
        Counts by trajectory, week and place, `(n, W, P)`.
    """
    n_days = np.shape(curves["known"])[1] - 1
    j = n_days if day > n_days else max(day, 0) - 1
    total = np.zeros(np.shape(curves["draw_known"]), dtype=float)
    for s in _STREAMS:
        rest = np.zeros_like(total) if j < 0 else np.asarray(curves[s])[:, j]
        phi = np.asarray(curves[f"phi_{s}"], dtype=float)[:, None, None]
        phi = np.broadcast_to(phi, total.shape)
        seen = np.broadcast_to(curves[f"vintage_{s}"], total.shape)
        drawn = np.asarray(curves[f"draw_{s}"], dtype=float)
        unseen = np.isnan(seen)
        if s == "known":
            later = _neg_binomial(rest, phi, rng)
            total += np.where(unseen, drawn, seen) + later
        else:
            centre = np.where(unseen, rest, np.nan_to_num(seen) + rest)
            later = _neg_binomial(centre, phi, rng)
            total += np.where(unseen, drawn + later, later)
    return total


def _pack(x: NDArray[Any]) -> str:
    return base64.b64encode(zlib.compress(x.tobytes(), 9)).decode()


def _unpack(text: str, dtype: Any, shape: tuple[int, ...]) -> NDArray[Any]:
    raw = zlib.decompress(base64.b64decode(text))
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def encode(curves: Mapping[str, Any]) -> dict[str, Any]:
    """Encode `reporting_curves` output compactly for the fit store.

    Each stream's expected cases are stored as the eventual count and the
    share of it reported by each day, rounded to 1/255 and differenced
    over days, up to the last day any share is below 1. A cell expecting
    fewer than `NEGLIGIBLE` more cases is reported in full on the first
    day. The counts and concentrations are stored as float32.

    Parameters
    ----------
    curves : mapping
        `reporting_curves` output.

    Returns
    -------
    dict
        JSON-ready block, read back by `decode`.
    """
    n, n_days1, n_w, n_p = np.shape(curves["known"])
    out: dict[str, Any] = {
        "cutoff": int(curves["cutoff"]),
        "shape": [n, n_days1 - 1, n_w, n_p],
    }
    last = 0
    shares = {}
    for s in _STREAMS:
        x = np.asarray(curves[s], dtype=float)
        final = x[:, -1]
        share = np.divide(
            x[:, :-1],
            final[:, None],
            out=np.ones_like(x[:, :-1]),
            where=final[:, None] > NEGLIGIBLE,
        )
        q = np.rint(np.clip(share, 0, 1) * _STEPS).astype(np.uint8)
        q = np.maximum.accumulate(q, axis=1)
        below = np.nonzero((q < _STEPS).any(axis=(0, 2, 3)))[0]
        last = max(last, int(below[-1]) + 1 if below.size else 0)
        shares[s] = q
        out[f"final_{s}"] = _pack(final.astype(np.float32))
        for key in (f"vintage_{s}", f"draw_{s}", f"phi_{s}"):
            out[key] = _pack(np.asarray(curves[key], dtype=np.float32))
    out["days"] = last
    for s in _STREAMS:
        q = shares[s][:, :last]
        step = np.diff(q, axis=1, prepend=np.zeros_like(q[:, :1]))
        out[f"share_{s}"] = _pack(step.astype(np.uint8))
    return out


def decode(block: Mapping[str, Any]) -> dict[str, Any]:
    """Read back a block written by `encode`.

    Parameters
    ----------
    block : mapping
        `encode` output.

    Returns
    -------
    dict
        As `reporting_curves`, with the expected cases rounded to 1/255
        of their eventual count, and every share 1 past the stored days.
    """
    n, n_days, n_w, n_p = block["shape"]
    last = block["days"]
    out: dict[str, Any] = {"cutoff": block["cutoff"]}
    for s in _STREAMS:
        f32 = np.float32
        final = _unpack(block[f"final_{s}"], f32, (n, n_w, n_p))
        step = _unpack(block[f"share_{s}"], np.uint8, (n, last, n_w, n_p))
        share = np.ones((n, n_days + 1, n_w, n_p))
        share[:, :last] = np.cumsum(step, axis=1, dtype=float) / _STEPS
        out[s] = share * final[:, None].astype(float)
        out[f"vintage_{s}"] = _unpack(block[f"vintage_{s}"], f32, (n_w, n_p))
        out[f"draw_{s}"] = _unpack(block[f"draw_{s}"], f32, (n, n_w, n_p))
        out[f"phi_{s}"] = _unpack(block[f"phi_{s}"], f32, (n,))
    return out


def as_reported_samples(
    block: Mapping[str, Any], cutoff: dt.date, rng: np.random.Generator
) -> pd.DataFrame:
    """Return a fit's samples of the counts a later release shows.

    Parameters
    ----------
    block : mapping
        One run's stored block: `start_date` (ISO), `places`, `weeks`
        (hub week ends, one per curve week) and `curves` (`encode`).
    cutoff : datetime.date
        The later release's cut-off.
    rng : numpy.random.Generator
        Random numbers.

    Returns
    -------
    pandas.DataFrame
        Columns `location`, `target_end_date`, `sample` and `value`.
    """
    curves = decode(block["curves"])
    start = dt.date.fromisoformat(block["start_date"])
    day = (cutoff - start).days - int(curves["cutoff"])
    values = draw_reported(curves, day, rng)
    n, n_w, n_p = values.shape
    return pd.DataFrame(
        {
            "location": np.tile(np.repeat(block["places"], 1), n * n_w),
            "target_end_date": np.tile(np.repeat(block["weeks"], n_p), n),
            "sample": np.repeat(np.arange(1, n + 1), n_w * n_p),
            "value": values.ravel(),
        }
    )
