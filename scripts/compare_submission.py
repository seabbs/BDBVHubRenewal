# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Compare hub submission files of several models, one panel per place.

Each panel shows a model's median with 50% and 90% intervals for weekly
confirmed cases, over the target data of one vintage. Files may hold
`sample` rows (as this repository writes them) or `quantile` rows (as
hub model outputs). Samples are summarised by their empirical
quantiles.

Usage::

    uv run python scripts/compare_submission.py \\
        randomwalknowcast=path/to/2026-11-07-seabbs-rwnowcast.csv \\
        renewal=path/to/2026-11-07-seabbs-renewal.csv \\
        --vintage 2026-11-01 --out output/compare.png

A bare path is labelled by its file name. `--locations` sets the panels
(default: national and the main provinces).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

TARGET = "wk inc confirmed cases"
LOCATIONS = ("CD", "CD-IT", "CD-NK", "CD-HU", "CD-TO")
LEVELS = {0.5: (0.25, 0.75), 0.9: (0.05, 0.95)}
"""Interval level: its lower and upper quantile."""
# Okabe-Ito, which stays distinct for colour-blind readers.
COLOURS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
)


def read_summary(path: Path, locations: Sequence[str]) -> pd.DataFrame:
    """Median and interval quantiles by place and week, from one file.

    Returns columns `location`, `target_end_date`, and the quantiles
    0.05, 0.25, 0.5, 0.75 and 0.95.
    """
    df = pd.read_csv(path, dtype={"output_type_id": str})
    df = df[(df["target"] == TARGET) & df["location"].isin(locations)]
    if df.empty:
        raise ValueError(f"{path}: no `{TARGET}` rows for {list(locations)}")
    want = sorted({q for lo_hi in LEVELS.values() for q in lo_hi} | {0.5})
    kind = set(df["output_type"])
    if "sample" in kind:
        rows = df[df["output_type"] == "sample"]
        out = (
            rows.groupby(["location", "target_end_date"])["value"]
            .quantile(want)
            .unstack()
        )
    elif "quantile" in kind:
        rows = df[df["output_type"] == "quantile"].copy()
        rows["q"] = rows["output_type_id"].astype(float).round(4)
        rows = rows[rows["q"].isin(want)]
        out = rows.pivot_table(
            index=["location", "target_end_date"],
            columns="q",
            values="value",
            aggfunc="first",
        )
    else:
        raise ValueError(f"{path}: needs sample or quantile rows, has {kind}")
    missing = [q for q in want if q not in out.columns]
    if missing:
        raise ValueError(f"{path}: no quantiles {missing}")
    out = out.reset_index()
    out["target_end_date"] = pd.to_datetime(out["target_end_date"])
    return out


def read_observed(target: Path, vintage: str) -> pd.DataFrame:
    """Observed plus imputed weekly cases at a vintage (`latest` allowed)."""
    df = pd.read_csv(target, dtype={"vintage": str})
    if vintage == "latest":
        vintage = str(df["vintage"].max())
    df = df[df["vintage"] == vintage].copy()
    if df.empty:
        raise ValueError(f"{target}: no rows for vintage {vintage}")
    df["target_end_date"] = pd.to_datetime(df["target_end_date"])
    return df[["location", "target_end_date", "total"]]


def plot(
    summaries: dict[str, pd.DataFrame],
    observed: pd.DataFrame,
    locations: Sequence[str],
    vintage: str,
    reference_date: str = "",
) -> plt.Figure:
    """One panel per location, each model as median and 50%/90% bands."""
    n = len(locations)
    cols = min(n, 3)
    rows = -(-n // cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(5 * cols, 3.4 * rows), squeeze=False, sharex=True
    )
    first = min(s["target_end_date"].min() for s in summaries.values())
    for ax, loc in zip(axes.ravel(), locations, strict=False):
        obs = observed[observed["location"] == loc]
        obs = obs[obs["target_end_date"] >= first]
        ax.plot(
            obs["target_end_date"],
            obs["total"],
            "o",
            color="black",
            ms=4,
            label=f"data {vintage}",
            zorder=5,
        )
        for (name, s), colour in zip(summaries.items(), COLOURS, strict=False):
            s = s[s["location"] == loc].sort_values("target_end_date")
            if s.empty:
                continue
            x = s["target_end_date"]
            for level, (lo, hi) in LEVELS.items():
                ax.fill_between(
                    x,
                    s[lo],
                    s[hi],
                    color=colour,
                    alpha=0.15 if level == 0.9 else 0.3,
                    linewidth=0,
                )
            ax.plot(x, s[0.5], color=colour, lw=1.6, label=name)
        if reference_date:
            ax.axvline(pd.Timestamp(reference_date), color="grey", ls=":", lw=1)
        ax.set_title(loc, loc="left", fontsize=10)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels))
    fig.suptitle(
        "Weekly confirmed cases: median, 50% and 90% intervals",
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    return fig


def parse_files(items: Sequence[str]) -> dict[str, Path]:
    """`label=path` or `path` arguments, in order."""
    out: dict[str, Path] = {}
    for item in items:
        label, sep, path = item.partition("=")
        p = Path(path if sep else item)
        if not p.is_file():
            raise FileNotFoundError(p)
        out[label if sep else p.stem] = p
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Read the files, write the figure."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("files", nargs="+", help="label=path or path, per model")
    p.add_argument("--vintage", default="latest", help="Data vintage")
    p.add_argument("--target", type=Path, default=Path("data/target.csv"))
    p.add_argument("--locations", nargs="+", default=list(LOCATIONS))
    p.add_argument("--reference-date", default="", help="Marked on each panel")
    p.add_argument("--out", type=Path, default=Path("output/compare.png"))
    args = p.parse_args(argv)
    files = parse_files(args.files)
    summaries = {k: read_summary(v, args.locations) for k, v in files.items()}
    observed = read_observed(args.target, args.vintage)
    vintage = args.vintage
    if vintage == "latest":
        vintage = str(
            pd.read_csv(args.target, dtype={"vintage": str})["vintage"].max()
        )
    fig = plot(
        summaries, observed, args.locations, vintage, args.reference_date
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
