"""Example simulated outbreaks, with their final counts and releases.

`examples_figure` draws a few contrasting simulations of the grid, one
panel each: the final daily counts by onset date, and the counts known at
three release dates, so the reporting delay is visible. The workflow's
`sim_examples` rule runs it on pilot simulations of the grid
(`synthetic.examples` in the workflow config).

Usage: python -m paperfigs.sim_examples --sims DIR --picks JSON --out PNG
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from paperfigs.style import shared_legend

__all__ = ["RELEASES", "as_of", "examples_figure"]

RELEASES: Final = (0.4, 0.65, 0.9)
"""Release dates drawn, as shares of the way through each outbreak."""


def as_of(counts: pd.DataFrame, release: pd.Timestamp | None) -> pd.Series:
    """Return daily counts by onset date, summed over strata.

    With a `release` date, only reports up to it and onsets up to it are
    counted; without one, every report (the final counts).
    """
    rows = counts.assign(
        reference_date=pd.to_datetime(counts["reference_date"]),
        report_date=pd.to_datetime(counts["report_date"]),
    )
    if release is not None:
        rows = rows.loc[
            (rows["report_date"] <= release)
            & (rows["reference_date"] <= release)
        ]
    return rows.groupby("reference_date")["count"].sum()


def _subtitle(row: pd.Series) -> str:
    keys = ("trajectory", "delay_change", "observation", "strata")
    return " · ".join(str(row[k]) for k in keys if k in row)


def examples_figure(
    counts: pd.DataFrame,
    settings: pd.DataFrame,
    picks: Sequence[tuple[str, str]],
) -> Figure:
    """Plot each picked simulation's final counts and three releases.

    Parameters
    ----------
    counts : pandas.DataFrame
        The simulator's counts: `outbreak` (`<simulation>-<draw>`),
        `location`, `reference_date`, `report_date` and `count`.
    settings : pandas.DataFrame
        One row per simulation (`outbreak`), with its grid cell.
    picks : sequence of (str, str)
        Each simulation's id and the label its panel is titled with.
    """
    ncol = min(3, len(picks))
    nrow = -(-len(picks) // ncol)
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(3.6 * ncol, 2.6 * nrow),
        layout="constrained",
        squeeze=False,
    )
    cell = settings.set_index("outbreak")
    colours = ("#9ecae1", "#4292c6", "#08519c")
    for ax, (sim, label) in zip(axes.ravel(), picks, strict=False):
        one = counts.loc[counts["outbreak"] == f"{sim}-1"]
        final = as_of(one, None)
        ax.plot(final.index, final.values, color="black", lw=1, label="final")
        start, end = final.index.min(), final.index.max()
        for share, colour in zip(RELEASES, colours, strict=True):
            release = start + (end - start) * share
            known = as_of(one, release)
            ax.plot(
                known.index,
                known.values,
                color=colour,
                lw=1,
                ls="--",
                label=f"release at {share:.0%}",
            )
        title = label
        if sim in cell.index:
            title += f"\n{_subtitle(cell.loc[sim])}"
        ax.set_title(title, fontsize=8)
        ax.set_ylabel("Daily cases", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.tick_params(axis="x", rotation=30)
    for ax in axes.ravel()[len(picks) :]:
        ax.set_visible(False)
    shared_legend(fig, axes, ncol=4, fontsize=7)
    return fig


def main() -> None:
    """Draw the figure from the pilot simulations' files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sims", type=Path, required=True)
    parser.add_argument(
        "--picks", required=True, help="[[simulation, label], ...] as JSON."
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    counts = pd.concat(
        pd.read_csv(p) for p in sorted(args.sims.glob("counts-*.csv.gz"))
    )
    settings = pd.concat(
        pd.read_csv(p) for p in sorted(args.sims.glob("settings-*.csv"))
    )
    picks = [tuple(p) for p in json.loads(args.picks)]
    fig = examples_figure(counts, settings, picks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
