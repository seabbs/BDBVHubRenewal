"""The synthetic scenario grid tables and SI notes, read from the grid config.

The grid config (`config/finetune/grids/<grid>.yaml`, the one the main
workflow's `simulate.grid_file` names) defines the grid: categorical
`factors` (levels with weights, drawn independently for each simulation),
each placed in a layer by `factor_layers`, and named sampling
`distributions`, each with its layer and the reason for it. The engine
(`packages/nowcastfoundation/r`) reads the same file, so the tables cannot
drift from what is simulated.

- `layer_tables`: the SI's two grid tables, strata and infections, and
  observation, delays and the reporting system, with each factor's levels
  and each layer's sampling distributions;
- `distribution_table`: every sampling distribution with its reason;
- `model_note` and `heldout_note`: the SI notes on the simulator and on
  the held-out set (the `synthetic` block of the paper workflow config).

The workflow's `grid_tables` rule runs it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Final

import pandas as pd
import yaml

__all__ = [
    "LAYERS",
    "TABLES",
    "distribution",
    "distribution_table",
    "factor_shares",
    "heldout_note",
    "layer_tables",
    "model_note",
    "read_grid",
    "to_markdown",
]

LAYERS: Final = ("strata", "infections", "observation", "delays", "reporting")
"""The grid's layers, in the order the SI describes them."""

TABLES: Final = {
    "epidemic": ("strata", "infections"),
    "reporting": ("observation", "delays", "reporting"),
}
"""The layers of each SI grid table."""

LAYER_NAMES: Final = {
    "strata": "Strata and aggregation",
    "infections": "Infections",
    "observation": "Observation",
    "delays": "Reporting delays",
    "reporting": "Reporting system",
}
"""Each layer's name in the tables."""

HEADERS: Final = {
    "layer": "Layer",
    "setting": "Setting",
    "values": "Levels or sampling distribution",
}
"""Headers of the SI grid tables."""

DISTRIBUTION_HEADERS: Final = {
    "setting": "Setting",
    "distribution": "Sampling distribution",
    "why": "Reason",
}
"""Headers of the SI table of sampling distributions."""


def read_grid(path: Path | str) -> dict[str, Any]:
    """Return a grid config."""
    return yaml.safe_load(Path(path).read_text())


def factor_shares(grid: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Return each factor's levels with their shares (weights normalised)."""
    out = {}
    for name, levels in grid["factors"].items():
        total = sum(levels.values())
        out[name] = {str(k): v / total for k, v in levels.items()}
    return out


def _number(x: float) -> str:
    """Return a number as short text: 7, 0.55, 1e-04."""
    x = float(x)
    if x == int(x) and abs(x) < 1e6:
        return str(int(x))
    if abs(x) < 1e-3:
        return f"{x:.0e}"
    return f"{x:.3g}"


def distribution(entry: dict[str, Any]) -> str:
    """Return a sampling distribution as text, as LogNormal(7, 0.55)."""
    d = entry["dist"]
    n = _number
    if d == "lognormal":
        return (
            f"LogNormal(median {n(entry['median'])}, sdlog {n(entry['sdlog'])})"
        )
    if d == "normal":
        return f"Normal({n(entry['mean'])}, {n(entry['sd'])})"
    if d == "halfnormal":
        return f"HalfNormal({n(entry['scale'])})"
    if d == "gamma":
        return f"Gamma(mean {n(entry['mean'])}, sd {n(entry['sd'])})"
    if d == "uniform":
        return f"Uniform({n(entry['lower'])}, {n(entry['upper'])})"
    if d == "beta":
        text = f"Beta({n(entry['a'])}, {n(entry['b'])})"
        lower, upper = float(entry["lower"]), float(entry["upper"])
        if (lower, upper) != (0.0, 1.0):
            text += f" on [{n(lower)}, {n(upper)}]"
        return text
    raise ValueError(f"unknown sampling distribution {d!r}")


def _levels(shares: dict[str, float]) -> str:
    """Return levels as `a`, `b`, with shares where they are not even."""
    even = len({round(s, 6) for s in shares.values()}) == 1
    return ", ".join(
        f"`{k}`" if even else f"`{k}` ({s:.0%})" for k, s in shares.items()
    )


def layer_tables(grid: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Return the grid's two SI tables, by layer.

    Each layer's factors (levels, with their shares where uneven), then
    its sampling distributions. A factor or distribution whose layer is
    not one of `LAYERS` fails, so a new one is placed.
    """
    rows: dict[str, list[dict[str, str]]] = {k: [] for k in LAYERS}
    shares = factor_shares(grid)
    for name, layer in grid["factor_layers"].items():
        if layer not in rows:
            raise KeyError(f"factor {name} has layer {layer}, not in LAYERS")
        rows[layer].append({"setting": name, "values": _levels(shares[name])})
    missing = set(shares) - set(grid["factor_layers"])
    if missing:
        raise KeyError(f"factors with no layer: {sorted(missing)}")
    for name, entry in grid["distributions"].items():
        if entry["layer"] not in rows:
            raise KeyError(f"{name} has layer {entry['layer']}, not in LAYERS")
        rows[entry["layer"]].append(
            {"setting": name, "values": distribution(entry)}
        )
    return {
        table: pd.DataFrame(
            [
                {
                    "layer": LAYER_NAMES[layer],
                    "setting": f"`{r['setting']}`",
                    "values": r["values"],
                }
                for layer in layers
                for r in rows[layer]
            ]
        )
        for table, layers in TABLES.items()
    }


def distribution_table(grid: dict[str, Any]) -> pd.DataFrame:
    """Return every sampling distribution with its reason, by layer."""
    order = {layer: i for i, layer in enumerate(LAYERS)}
    entries = sorted(
        grid["distributions"].items(), key=lambda kv: order[kv[1]["layer"]]
    )
    return pd.DataFrame(
        [
            {
                "setting": f"`{name}`",
                "distribution": distribution(entry),
                "why": entry.get("why", ""),
            }
            for name, entry in entries
        ]
    )


def to_markdown(table: pd.DataFrame, headers: dict[str, str]) -> str:
    """Return the table as Markdown, with `headers` as its columns."""
    cols = list(headers)
    lines = [
        "| " + " | ".join(headers[c] for c in cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for _, row in table.iterrows():
        cells = [str(row[c]).replace("|", "\\|") for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def model_note(grid: dict[str, Any], grid_file: str) -> str:
    """Return the SI note on the simulator, from its grid config."""
    floors = grid["floors"]
    lines = [
        "#### The simulator",
        "",
        "We simulated each synthetic outbreak in five layers: strata and"
        " aggregation, infections, observation, reporting delays and the"
        " reporting system.",
        "Infections follow a renewal process with susceptible depletion in"
        " each stratum, mixing between places, imports, shifts in"
        " transmission and immune-escape variants.",
        "Expected reference events are the infections convolved with the"
        " incubation period and scaled by the ascertainment.",
        "The release triangle is drawn from the generative model of"
        " epinowcast @placeholder with `enw_simulate()`, which runs the"
        " epinowcast Stan model with fixed parameters and no likelihood.",
        f"epinowcast is pinned at commit `{grid['epinowcast_sha']}`.",
        "The simulator is the R package `packages/nowcastfoundation/r`,"
        f" run on the grid config `{grid_file}` (grid `{grid['name']}`).",
        "Each simulation draws every categorical setting independently from"
        " its levels' shares, and every continuous setting from its named"
        " sampling distribution, as the SI tables list.",
        "A parameter we did not set is drawn from its epinowcast prior.",
        f"Seed infections are at least {floors['seed']} and series at least"
        f" {floors['weeks']} weeks long.",
        f"A draw that errors, reports fewer than {floors['min_cases']} cases"
        " or fails the simulation checks is drawn again, up to"
        f" {grid['tries']} times, then from a new design.",
        "",
        "##### Factors",
        "",
    ]
    for name, shares in factor_shares(grid).items():
        layer = LAYER_NAMES[grid["factor_layers"][name]]
        lines.append(f"- `{name}` ({layer.lower()}): {_levels(shares)}.")
    return "\n".join(lines) + "\n"


def _seeds(seeds: list[int]) -> str:
    """Return seeds as `seed 7001` or `seeds 8016 to 8025`."""
    if len(seeds) == 1:
        return f"seed {seeds[0]}"
    return f"seeds {seeds[0]} to {seeds[-1]}"


def _ordinal(n: int) -> str:
    """Return `n` as an ordinal, such as 6th."""
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    if 10 <= n % 100 <= 20:
        suffix = "th"
    return f"{n}{suffix}"


def heldout_note(config: dict) -> str:
    """Return the SI note on the held-out set, from its config.

    `config` is the workflow config's `synthetic` block with the run's
    `heldout_seeds` and `development_seeds`.
    """
    protocol = config["protocol"]
    min_releases = protocol["min_releases"]
    horizon_weeks = protocol["horizon_weeks"]
    horizons = protocol["horizons"]
    seeds = config["heldout_seeds"]
    n = config["simulations_per_seed"]
    blocks = config.get("blocks") or []
    dev = config["development_seeds"]
    cover = (
        [
            f"These cover the complete balanced blocks {blocks[0]} to"
            f" {blocks[-1]}, and simulations outside them are not used."
        ]
        if blocks
        else []
    )
    lines = [
        "#### The held-out synthetic set",
        "",
        f"The held-out set is simulated on grid `{config['grid']}` with"
        f" {_seeds(seeds)}, {n} simulations each.",
        *cover,
        f"We evaluate draw {config['draw']} of each simulation, so the"
        " outbreaks are independent.",
        "The fine-tune never trains on these seeds.",
        f"The simulations are written to `{config['simulations']['paper']}`,"
        " and the workflow config records them.",
        "The synthetic runner runs the models on the set.",
        "",
        "The held-out set was reserved for the final evaluation.",
        "No model was scored on it during development.",
        f"We checked the pipeline on the r3 pilot {_seeds(dev)}, whose"
        " forecasts are not results.",
        "",
        "##### Protocol",
        "",
        "Each outbreak is released weekly, on one weekday from Wednesday to"
        " Saturday drawn from its id, and a release's cut-off is its origin.",
        "Each outbreak has four origins, spread evenly from its"
        f" {_ordinal(min_releases)} release to the last release at least its"
        f" longest delay plus {horizon_weeks} weeks before its last report.",
        f"Each forecast covers weeks {horizons[0]} to {horizons[-1]} from"
        " the week of the cut-off, as weekly totals by onset week for the"
        " country and each stratum.",
        "Scores use rolling truth, the count reported D days after the end"
        " of a target week, with D the outbreak's own 98% reporting delay in"
        " whole days, and are relative to `randomwalknowcast`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Write the grid tables and the SI notes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid", type=Path, required=True, help="The grid config file."
    )
    parser.add_argument(
        "--grid-name", required=True, help="The grid file's repository path."
    )
    parser.add_argument(
        "--config", required=True, help="The `synthetic` block as JSON."
    )
    parser.add_argument("--si-dir", type=Path, required=True)
    args = parser.parse_args()
    grid = read_grid(args.grid)
    args.si_dir.mkdir(parents=True, exist_ok=True)
    for name, table in layer_tables(grid).items():
        (args.si_dir / f"grid-{name}.md").write_text(
            to_markdown(table, HEADERS)
        )
    (args.si_dir / "grid-distributions.md").write_text(
        to_markdown(distribution_table(grid), DISTRIBUTION_HEADERS)
    )
    (args.si_dir / "epinowcast-model.md").write_text(
        model_note(grid, args.grid_name)
    )
    config = json.loads(args.config)
    (args.si_dir / "heldout-set.md").write_text(heldout_note(config))


if __name__ == "__main__":
    main()
