# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""The pairing sensitivity page: how much the walk models' copula matters.

`pairing_page` reads the `pairing-sensitivity` results and writes one
compact table per study to `docs/generated/pairing.md`, which
`docs/pairing.md` includes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bdbvhubrenewal.evaluation.pairing import summarise_pairing

__all__ = ["pairing_page"]

_STUDIES = {
    "bdbv": "BDBV release backtest",
    "germany": "German development half",
}
_TARGETS = {
    "cell": "each week",
    "sum_weeks": "each place over its weeks",
    "sum_zones": "a province's zones each week",
    "joint": "all places and weeks (energy)",
}


def _table(summary: pd.DataFrame) -> str:
    """Return one study's table: CRPS or energy relative to no pairing."""
    rows = summary.loc[
        summary["metric"].isin(["crps", "energy"])
        & (summary["variant"] == "rolling")
    ]
    wide = rows.pivot_table(
        index=["model", "level", "target"],
        columns="variant",
        values="relative",
    ).reset_index()
    cover = summary.loc[
        (summary["metric"] == "cover90")
        & summary["variant"].isin(["off", "rolling"])
    ].pivot_table(
        index=["model", "level", "target"], columns="variant", values="value"
    )
    lines = [
        "| model | level | target | rolling | 90% cover, off | "
        "90% cover, rolling |",
        "|---|---|---|---|---|---|",
    ]
    cov = cover.to_dict("index")

    def fmt(v: object) -> str:
        try:
            x = float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return ""
        return "" if x != x else f"{x:.2f}"

    for r in wide.to_dict("records"):
        model, level, target = (
            str(r["model"]),
            str(r["level"]),
            str(r["target"]),
        )
        c = cov.get((model, level, target), {})
        lines.append(
            f"| `{model}` | {level} | {_TARGETS.get(target, target)} "
            f"| {fmt(r.get('rolling'))} "
            f"| {fmt(c.get('off'))} | {fmt(c.get('rolling'))} |"
        )
    return "\n".join(lines)


def pairing_page(results: Path | str, generated: Path | str) -> None:
    """Write the pairing sensitivity tables.

    Parameters
    ----------
    results : pathlib.Path or str
        The `pairing-sensitivity` output directory.
    generated : pathlib.Path or str
        Where `pairing.md` goes.
    """
    gen = Path(generated)
    gen.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(results).glob("pairing-*.csv.gz"))
    frames = [pd.read_csv(f) for f in files]
    frames = [f for f in frames if not f.empty]
    if not frames:
        text = "The pairing sensitivity was not run in this build."
    else:
        summary = summarise_pairing(pd.concat(frames, ignore_index=True))
        parts = []
        for study, label in _STUDIES.items():
            own = summary.loc[summary["study"] == study]
            if own.empty:
                continue
            parts += [f"### {label}", "", _table(own), ""]
        text = "\n".join(parts)
    (gen / "pairing.md").write_text(text.rstrip() + "\n")
