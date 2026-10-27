# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Posterior predictive checks: the `ppc.json` a fit writes, and its plot.

`bdbv-hub-renewal fit` writes `<fit dir>/ppc.json` beside
`diagnostics.json`, and the submission gate (`ci/`) reads it to check
interval coverage and calibration in sample.
The schema (version 1) is documented in `ci/INTEGRATION.md`.
It has one row per location, stream and in-sample onset week, with the
observed count, the predictive median, the central intervals and the
randomised probability integral transform (PIT) of the observation.

A model with no posterior, such as persistence, writes the
`not_available` form instead.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import matplotlib as mpl

mpl.use("Agg")

import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from bdbv_hub_renewal.plots import _grid, model_colour, theme

__all__ = [
    "SCHEMA_VERSION",
    "STREAMS",
    "not_available",
    "plot_ppc",
    "read_ppc",
    "summarise_draws",
    "write_ppc",
]

SCHEMA_VERSION: Final = 1
STREAMS: Final = ("observed", "imputed", "total")
"""Known onsets, the hub's imputed onsets, and their sum."""

_COLUMNS: Final = ("location", "stream", "target_end_date", "observed")


def _pct(level: float) -> int:
    return round(100 * level)


def not_available(
    reason: str, model: str | None = None, config: str | None = None
) -> dict[str, Any]:
    """Return the `ppc.json` payload of a model with no posterior.

    Parameters
    ----------
    reason : str
        Why there are no posterior predictive checks, for the report.
    model, config : str, optional
        Model and configuration, recorded when given.

    Returns
    -------
    dict
        `{"schema_version": 1, "available": false, "reason": ...}`.
    """
    out: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "available": False,
        "reason": reason,
    }
    if model is not None:
        out["model"] = model
    if config is not None:
        out["config"] = config
    return out


def summarise_draws(
    observed: pd.DataFrame,
    draws: ArrayLike,
    *,
    as_of: str | dt.date,
    model: str = "renewal",
    config: str = "main",
    complete_lag_weeks: int = 4,
    intervals: Sequence[float] = (0.5, 0.9),
    seed: int = 1,
) -> dict[str, Any]:
    """Summarise posterior predictive draws against the observed counts.

    Parameters
    ----------
    observed : pandas.DataFrame
        One row per location, stream and onset week, with columns
        `location`, `stream` (one of `STREAMS`), `target_end_date`
        (ISO date) and `observed` (the count in the data vintage).
    draws : array_like
        Posterior predictive draws of the observed counts, of shape
        `(n_draws, len(observed))`, with column `j` matching row `j` of
        `observed`.
    as_of : str or datetime.date
        Data vintage. Weeks ending at least `complete_lag_weeks` weeks
        before it are marked complete.
    model, config : str
        Model and configuration, recorded in the payload.
    complete_lag_weeks : int
        Completeness lag in weeks.
    intervals : sequence of float
        Central interval levels, such as `(0.5, 0.9)`. Each adds
        `lower_<pct>` and `upper_<pct>` to the rows.
    seed : int
        Seed of the uniform draws in the randomised PIT, so the payload
        is reproducible.

    Returns
    -------
    dict
        The `ppc.json` payload, with `available` true.

    Raises
    ------
    ValueError
        If a column is missing, a stream is unknown, or the draws do
        not match the rows.

    Notes
    -----
    The PIT is the randomised PIT for counts:
    `F(y - 1) + v * (F(y) - F(y - 1))` with `v ~ Uniform(0, 1)` and `F`
    the empirical predictive distribution function, using `P(X < y)` for
    `F(y - 1)`. It is uniform on (0, 1) under a calibrated model, even
    though the counts are discrete.
    """
    missing = [c for c in _COLUMNS if c not in observed.columns]
    if missing:
        raise ValueError(f"observed lacks columns {missing}")
    unknown = sorted(set(observed["stream"]) - set(STREAMS))
    if unknown:
        raise ValueError(f"unknown streams {unknown}; use {list(STREAMS)}")
    x = np.asarray(draws, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(observed):
        raise ValueError(
            f"draws have shape {x.shape}, not (n_draws, {len(observed)})"
        )
    levels = [float(level) for level in intervals]
    if not all(0 < level < 1 for level in levels):
        raise ValueError(f"interval levels must be in (0, 1): {levels}")
    y = observed["observed"].to_numpy(dtype=float)
    below = (x < y).mean(axis=0)
    at_or_below = (x <= y).mean(axis=0)
    v = np.random.default_rng(seed).uniform(size=len(y))
    pit = below + v * (at_or_below - below)
    probs = [0.5]
    for level in levels:
        probs += [(1 - level) / 2, (1 + level) / 2]
    q = np.quantile(x, probs, axis=0) if len(y) else np.empty((len(probs), 0))
    vintage = dt.date.fromisoformat(str(as_of))
    cutoff = (vintage - dt.timedelta(weeks=complete_lag_weeks)).isoformat()
    rows: list[dict[str, Any]] = []
    weeks = pd.to_datetime(observed["target_end_date"]).dt.date.astype(str)
    columns = zip(
        observed["location"].astype(str),
        observed["stream"].astype(str),
        weeks,
        y,
        strict=True,
    )
    for j, (location, stream, week, obs) in enumerate(columns):
        row: dict[str, Any] = {
            "location": location,
            "stream": stream,
            "target_end_date": week,
            "observed": int(obs) if float(obs).is_integer() else float(obs),
            "median": float(q[0, j]),
        }
        for k, level in enumerate(levels):
            row[f"lower_{_pct(level)}"] = float(q[1 + 2 * k, j])
            row[f"upper_{_pct(level)}"] = float(q[2 + 2 * k, j])
        row["pit"] = float(pit[j])
        row["complete"] = week <= cutoff
        rows.append(row)
    return {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "model": model,
        "config": config,
        "intervals": levels,
        "rows": rows,
    }


def write_ppc(ppc: Mapping[str, Any], path: Path | str) -> Path:
    """Write a `ppc.json` payload.

    Parameters
    ----------
    ppc : mapping
        Output of `summarise_draws` or `not_available`.
    path : pathlib.Path or str
        File to write, usually `<fit dir>/ppc.json`.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ppc, indent=1) + "\n", encoding="utf-8")
    return out


def read_ppc(path: Path | str) -> dict[str, Any]:
    """Read and check a `ppc.json` file.

    Parameters
    ----------
    path : pathlib.Path or str
        The file.

    Returns
    -------
    dict
        The payload.

    Raises
    ------
    ValueError
        If the file is not a version 1 payload, or an available payload
        has no rows list.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: not a JSON object")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: schema_version {data.get('schema_version')!r}, "
            f"expected {SCHEMA_VERSION}"
        )
    if "available" not in data:
        raise ValueError(f"{path}: no 'available' field")
    if data["available"] and not isinstance(data.get("rows"), list):
        raise ValueError(f"{path}: no 'rows' list")
    return data


def _coverage(rows: pd.DataFrame, pct: int) -> float:
    lo, hi = f"lower_{pct}", f"upper_{pct}"
    if rows.empty or lo not in rows or hi not in rows:
        return float("nan")
    inside = (rows["observed"] >= rows[lo]) & (rows["observed"] <= rows[hi])
    return float(inside.mean())


def plot_ppc(
    ppc: Mapping[str, Any], stream: str = "total", ncol: int = 3
) -> Figure:
    """Plot the posterior predictive intervals against the observed counts.

    Small multiples by location, largest first, in light mode on a white
    background, for the submission gate's `ppc.png` artefact.

    Parameters
    ----------
    ppc : mapping
        A `ppc.json` payload with `available` true.
    stream : str
        Stream to show. Falls back to the first stream present.
    ncol : int
        Panels per row.

    Returns
    -------
    matplotlib.figure.Figure
        Per location: the predictive median with the widest and
        narrowest intervals, and the observed counts, hollow for
        incomplete weeks and marked where outside the widest interval.

    Raises
    ------
    ValueError
        If the payload is not available or has no rows.
    """
    if not ppc.get("available"):
        raise ValueError(f"no posterior predictive checks: {ppc.get('reason')}")
    table = pd.DataFrame(ppc.get("rows") or [])
    if table.empty:
        raise ValueError("ppc.json has no rows")
    if stream not in set(table["stream"]):
        stream = str(table["stream"].iloc[0])
    table = table.loc[table["stream"] == stream].copy()
    table["date"] = pd.to_datetime(table["target_end_date"])
    sizes = table.groupby("location")["observed"].sum()
    locations = list(sizes.sort_values(ascending=False).index)
    pcts = sorted(_pct(level) for level in ppc.get("intervals", []))
    wide, narrow = (pcts[-1], pcts[0]) if pcts else (None, None)
    with theme("light") as ink:
        colour = model_colour("model", ["model"], "light")
        miss = model_colour("miss", ["model", "miss"], "light")
        fig, axes = _grid(len(locations), ncol, width=3.6, height=2.4)
        fig.set_facecolor("white")
        for ax, loc in zip(axes, locations, strict=True):
            d = table.loc[table["location"] == loc].sort_values("date")
            ax.set_facecolor("white")
            for pct, alpha in ((wide, 0.18), (narrow, 0.35)):
                if pct is None or f"lower_{pct}" not in d:
                    continue
                ax.fill_between(
                    d["date"],
                    d[f"lower_{pct}"],
                    d[f"upper_{pct}"],
                    color=colour,
                    alpha=alpha,
                    lw=0,
                    label=f"{pct}% interval",
                )
            ax.plot(
                d["date"], d["median"], color=colour, lw=1.2, label="Median"
            )
            done = d["complete"].astype(bool)
            out = np.zeros(len(d), dtype=bool)
            if wide is not None and f"lower_{wide}" in d:
                out = (
                    (d["observed"] < d[f"lower_{wide}"])
                    | (d["observed"] > d[f"upper_{wide}"])
                ).to_numpy()
            ax.scatter(
                d["date"][done],
                d["observed"][done],
                s=10,
                color=ink["data"],
                zorder=3,
                label="Observed (complete week)",
            )
            ax.scatter(
                d["date"][~done],
                d["observed"][~done],
                s=10,
                facecolors="white",
                edgecolors=ink["data"],
                zorder=3,
                label="Observed (incomplete week)",
            )
            ax.scatter(
                d["date"][out],
                d["observed"][out],
                s=28,
                facecolors="none",
                edgecolors=miss,
                lw=1.2,
                zorder=4,
                label=f"Outside the {wide}% interval",
            )
            cov = _coverage(d, wide) if wide is not None else float("nan")
            label = f", {wide}% coverage {cov:.2f}" if cov == cov else ""
            ax.set_title(f"{loc}{label}", fontsize=9)
            # matplotlib's date formatters are untyped.
            fmt = DateFormatter("%d %b")  # type: ignore[no-untyped-call]
            ax.xaxis.set_major_formatter(fmt)
            ax.tick_params(axis="x", labelrotation=30)
        axes[0].set_ylabel("Cases by onset week")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="outside lower center", ncol=3, fontsize=7
        )
        fig.suptitle(
            f"Posterior predictive check: {ppc.get('model', 'model')} "
            f"({ppc.get('config', '')}), {stream} onsets",
            color=ink["ink"],
            fontsize=10,
        )
    return fig
