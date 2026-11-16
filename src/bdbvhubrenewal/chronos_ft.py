# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Use our fine-tuned Chronos-2 on your own reporting triangle.

`load` downloads a `chronos-ft-<hash>` release of seabbs/BDBVHubRenewal,
the newest or a pinned one, and checks each adapter file against the
release's `sha256sums.txt`. `forecast_triangle` gives a reporting
triangle to Chronos-2 as the `fm_chronos_*_ragged_ft` models do and
returns quantiles of each row's final count, for the rows still being
reported and for rows ahead.
See the "Using the fine-tuned Chronos-2" docs page.
Needs the Chronos-2 dependency group (`uv sync --group chronos`).

Nothing in the backtest or the fit store imports this module, so it does
not change any stored fit's key.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bdbvhubrenewal.forecasters import foundation
from bdbvhubrenewal.forecasters.foundation import Checkpoint, HubTriangle

__all__ = ["FineTunedChronos", "forecast_triangle", "load"]

COLUMNS = ("place", "reference", "step", "quantile_level", "value")
"""Columns of `forecast_triangle`'s output."""


@dataclass(frozen=True)
class FineTunedChronos:
    """A fine-tuned checkpoint, downloaded and checked.

    Attributes
    ----------
    checkpoint : Checkpoint
        Its settings hash and the SHA-256 of each adapter file.
    adapter : pathlib.Path
        The LoRA adapter's directory in the local cache.
    """

    checkpoint: Checkpoint
    adapter: Path


def load(spec: str = "latest") -> FineTunedChronos:
    """Download a fine-tuned checkpoint and check its SHA-256.

    Parameters
    ----------
    spec : str
        `"latest"`, the newest `chronos-ft-*` release by publish date, or
        a settings hash such as `"e8a1f5ae338b"`, the release
        `chronos-ft-e8a1f5ae338b`.

    Returns
    -------
    FineTunedChronos

    Raises
    ------
    LookupError
        If there is no `chronos-ft-*` release.
    FileNotFoundError
        If the release lacks an asset.
    ValueError
        If an adapter file's SHA-256 is not the one in the release's
        `sha256sums.txt`, or the release holds another checkpoint.
    ConnectionError
        If GitHub cannot be reached. Set `GH_TOKEN` or `GITHUB_TOKEN` to
        avoid the anonymous rate limit.
    """
    checkpoint = foundation.resolve_checkpoint(spec)
    return FineTunedChronos(
        checkpoint, foundation.finetuned_adapter(checkpoint)
    )


def _pipeline(checkpoint: Checkpoint | None) -> Any:
    """Return Chronos-2, with the checkpoint's adapter merged in if any."""
    return foundation._chronos(checkpoint)


def _as_triangle(
    triangles: Mapping[str, pd.DataFrame], horizon: int
) -> tuple[HubTriangle, pd.Index]:
    """Lay out reporting triangles as the hub target is laid out.

    Row t is reported up to delay T - 1 - t, so the latest cut-off is the
    last row, and the count at lag d is the sum of delays 0 to d.
    """
    first = next(iter(triangles.values()))
    index, delays = first.index, first.columns
    for place, tri in triangles.items():
        if not (tri.index.equals(index) and tri.columns.equals(delays)):
            raise ValueError(
                f"The triangle of {place} has other rows or delays than the "
                "first triangle's."
            )
    counts = np.stack(
        [t.to_numpy(dtype=float) for t in triangles.values()]
    )  # [L, T, D]
    n_places, n_rows, n_delays = counts.shape
    if n_delays < 2:
        raise ValueError("The triangle needs at least two delays.")
    if n_rows - n_delays + 1 < 2:
        raise ValueError(
            f"The triangle needs at least two rows reported at every delay; "
            f"it has {max(n_rows - n_delays + 1, 0)} ({n_rows} rows, "
            f"{n_delays} delays)."
        )
    rows, lags = np.indices((n_rows, n_delays))
    known = rows + lags <= n_rows - 1
    gaps = np.isnan(counts) & known
    if gaps.any():
        place, row, delay = np.argwhere(gaps)[0]
        raise ValueError(
            f"The triangle of {list(triangles)[place]} is missing the count "
            f"of row {index[row]} at delay {delays[delay]}, which the last "
            "row's cut-off has reported. Give 0 for no reports."
        )
    # The last delay's count is final, so the lags below it are the
    # covariates, as `MAX_LAG` is the hub's last delay.
    cumulative = np.where(known, np.nancumsum(counts, axis=2), np.nan)
    n_times = n_rows + horizon
    partial = np.full((n_places, n_times, n_delays - 1), np.nan)
    partial[:, :n_rows] = cumulative[:, :, :-1]
    final = np.zeros((n_places, n_times))
    final[:, :n_rows] = np.nansum(np.where(known, counts, 0.0), axis=2)
    times = [str(i) for i in range(n_times)]
    return HubTriangle(times, final, partial, n_rows - 1), index


def _references(index: pd.Index, horizon: int) -> list[Any]:
    """Return the triangle's row labels and `horizon` more, if spaced."""
    labels = list(index)
    try:
        width = index[-1] - index[-2]
        labels += [index[-1] + width * (k + 1) for k in range(horizon)]
    except TypeError:
        labels += [None] * horizon
    return labels


def forecast_triangle(
    triangle: pd.DataFrame | Mapping[str, pd.DataFrame],
    horizon: int = 0,
    model: str | FineTunedChronos | None = "latest",
    quantile_levels: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Nowcast and forecast a reporting triangle with Chronos-2.

    The rows reported at every delay are the context. Each place's count
    at every delay is a covariate, known up to the last row's cut-off, as
    in the ragged layout of the `fm_chronos_*_ragged_ft` models; the
    last delay is taken as final. Our checkpoints were fine-tuned with
    last delays of 2 to 7 weeks or 14 to 49 days, so triangles whose last
    delay is in those ranges match their training best.

    Parameters
    ----------
    triangle : pandas.DataFrame or mapping of str to pandas.DataFrame
        One row per reference time, consecutive and equally spaced, and
        one column per delay from 0, holding the counts first reported at
        that delay. Row t of T is reported up to delay T - 1 - t; later
        cells are ignored and may be NaN. A mapping gives several places,
        each the same rows and delays, forecast as one group.
    horizon : int
        Rows to forecast past the last.
    model : str, FineTunedChronos or None
        A checkpoint (`load`), a spec for `load`, or None for the
        pretrained Chronos-2.
    quantile_levels : sequence of float, optional
        Levels to return, Chronos-2's own by default.

    Returns
    -------
    pandas.DataFrame
        `place`, `reference` (the row label, extended past the last row
        when the labels are dates or numbers), `step` (rows after the
        last; 0 is the last, below 0 the rows still being reported),
        `quantile_level` and `value`, the quantile of the row's final
        count, at least its count reported so far.

    Raises
    ------
    ValueError
        If a reported cell is NaN, places differ in rows or delays, or
        fewer than two rows are reported at every delay.
    ImportError
        If Chronos-2 is not installed (`uv sync --group chronos`).
    """
    triangles = (
        {"series": triangle}
        if isinstance(triangle, pd.DataFrame)
        else dict(triangle)
    )
    tri, index = _as_triangle(triangles, horizon)
    places = list(triangles)
    start = tri.n_context
    group = foundation.layout_block(tri, "ragged", places, start)
    checkpoint = load(model).checkpoint if isinstance(model, str) else model
    pipeline = _pipeline(
        checkpoint.checkpoint
        if isinstance(checkpoint, FineTunedChronos)
        else checkpoint
    )
    levels = list(
        pipeline.quantiles if quantile_levels is None else quantile_levels
    )
    length = len(tri.times) - start
    (q,), _ = pipeline.predict_quantiles(
        [dict(group)], prediction_length=length, quantile_levels=levels
    )
    q = q.float().cpu().numpy() if hasattr(q, "cpu") else np.asarray(q)
    q = np.maximum(np.sort(q, axis=-1), tri.final[:, start:, None])
    n_rows = len(index)
    refs = _references(index, horizon)[start:]
    steps = np.arange(start, len(tri.times)) - (n_rows - 1)
    n_q = len(levels)
    out = pd.DataFrame(
        {
            "place": np.repeat(places, length * n_q),
            "reference": np.tile(
                np.repeat(np.array(refs, dtype=object), n_q), len(places)
            ),
            "step": np.tile(np.repeat(steps, n_q), len(places)),
            "quantile_level": np.tile(levels, len(places) * length),
            "value": q.reshape(-1),
        }
    )
    if isinstance(index, pd.DatetimeIndex):
        out["reference"] = pd.to_datetime(out["reference"])
    out.attrs["checkpoint"] = getattr(checkpoint, "checkpoint", checkpoint)
    return out[list(COLUMNS)]
