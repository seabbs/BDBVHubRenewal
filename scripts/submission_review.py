# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Write a review page of a hub track's dry-run submission.

Reads a dry run of `ci/steps/track.sh` (`--track`, the track's WORK
folder), and writes a Markdown page with the gate's checks and hub
validation, fan charts of the submission beside other hub files, the
fine-tuned checkpoint it loaded, and any backtest tables given. Run from
the repo root after `prepare.sh` and `track.sh`:

    uv run python scripts/submission_review.py --track WORK/track-MODEL \\
        --observed WORK/inputs/time-series.csv \\
        --compare randomwalknowcast=WORK/candidates/randomwalknowcast \\
        --compare seabbs-renewal=PREVIOUS.csv --out docs/submission/review
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RELEASES = "https://github.com/seabbs/BDBVHubRenewal/releases/download"
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6")


def _hub_file(path: Path) -> Path:
    """Return a hub CSV: `path` itself, or the first under it."""
    if path.is_file():
        return path
    return sorted(path.rglob("*.csv"))[0]


def _quantiles(path: Path) -> pd.DataFrame:
    """Quantiles of each location and week of a hub sample file."""
    rows = pd.read_csv(_hub_file(path))
    q = (
        rows.groupby(["location", "target_end_date"])["value"]
        .quantile(list(LEVELS))
        .unstack()
    )
    return q.reset_index()


def _fan(ax, q: pd.DataFrame, colour: str, label: str) -> None:  # type: ignore[no-untyped-def]
    q = q.sort_values("target_end_date")
    x = pd.to_datetime(q["target_end_date"])
    ax.fill_between(x, q[0.05], q[0.95], color=colour, alpha=0.15, lw=0)
    ax.fill_between(x, q[0.25], q[0.75], color=colour, alpha=0.3, lw=0)
    ax.plot(x, q[0.5], color=colour, lw=1.5, label=label)


def fan_charts(
    series: dict[str, pd.DataFrame],
    observed: pd.DataFrame,
    locations: list[str],
    path: Path,
    title: str,
) -> None:
    """One panel per location, each model's 50% and 90% intervals."""
    n = len(locations)
    cols = min(n, 3)
    rows = -(-n // cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.2 * cols, 3 * rows), squeeze=False
    )
    for ax, loc in zip(axes.flat, locations, strict=False):
        obs = observed.loc[observed["location"] == loc]
        if not obs.empty:
            ax.plot(
                pd.to_datetime(obs["target_end_date"]),
                obs["observation"],
                "k.",
                ms=4,
                label="reported",
            )
        for (name, q), colour in zip(series.items(), COLOURS, strict=False):
            mine = q.loc[q["location"] == loc]
            if not mine.empty:
                _fan(ax, mine, colour, name)
        ax.set_title(loc, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.tick_params(axis="x", rotation=45)
    for ax in axes.flat[n:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _markdown(table: pd.DataFrame) -> str:
    """Return a data frame as a Markdown table."""
    head = "| " + " | ".join(map(str, table.columns)) + " |"
    rule = "|" + "---|" * len(table.columns)
    body = [
        "| " + " | ".join(map(str, r)) + " |"
        for r in table.itertuples(index=False)
    ]
    return "\n".join([head, rule, *body])


def _settings(tag: str) -> dict:
    """Return the checkpoint release's `settings.json`."""
    url = f"{RELEASES}/{tag}/settings.json"
    with urllib.request.urlopen(url, timeout=60) as response:
        return dict(json.loads(response.read()))


def main() -> None:
    """Write the review page."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--track", type=Path, required=True)
    p.add_argument("--observed", type=Path, required=True)
    p.add_argument(
        "--compare",
        action="append",
        default=[],
        help="NAME=PATH of another hub file (or a folder holding one)",
    )
    p.add_argument(
        "--table",
        action="append",
        default=[],
        help="TITLE=PATH of a Markdown or CSV backtest table",
    )
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--log", type=Path, default=None, help="track.sh's log")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    gate = args.track / "gate"
    if not (gate / "provenance.json").exists():
        # The track stopped before its gate: show why.
        log = args.log.read_text() if args.log and args.log.exists() else ""
        errors = [
            line for line in log.splitlines() if "ERROR" in line or "::" in line
        ]
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "index.md").write_text(
            "# Submission review\n\nThe dry run stopped before its gate, so "
            "nothing would be submitted.\n\n```text\n"
            + "\n".join(errors[-20:] or log.splitlines()[-40:])
            + "\n```\n"
        )
        print(f"Wrote {args.out / 'index.md'} (no gate).")
        return
    prov = json.loads((gate / "provenance.json").read_text())
    selected = gate / "selected"
    run = prov.get("run") or {}
    model_id = (args.track / "metadata").glob("*.yml")
    model_id = next(iter(sorted(model_id)), Path("unknown")).stem
    figures = args.out / "figures"
    lines = [
        f"# Submission review: {model_id}",
        "",
        f"Dry run of the `{model_id}` track for the round "
        f"{run.get('reference_date', 'n/a')}, from data as of "
        f"{run.get('as_of', 'n/a')}. Nothing was pushed.",
        "{ .page-subtitle }",
        "",
        "## What would be submitted",
        "",
        f"- Model: `{run.get('model', 'n/a')}`.",
        f"- Outcome: {'selected' if prov.get('selected') else 'round skipped'}"
        f". {prov.get('reason', '')}",
    ]
    if args.seconds is not None:
        lines.append(
            f"- The track ran in {args.seconds / 60:.1f} min on a GitHub "
            "runner's CPU."
        )
    checkpoint = run.get("checkpoint")
    if checkpoint:
        settings = _settings(checkpoint["tag"])
        lines += [
            f"- Checkpoint: release `{checkpoint['tag']}`, adapter SHA-256 "
            f"`{checkpoint['sha256']['adapter_model.safetensors'][:12]}`.",
            f"- Held-out loss: {settings.get('held_out_loss_start', 'n/a'):.3f}"
            f" before fine-tuning, {settings.get('held_out_loss', 'n/a'):.3f}"
            f" after, on {settings.get('n_validation', 'n/a')} held-out "
            f"blocks from {settings.get('n_outbreaks', 'n/a')} outbreaks.",
        ]
    lines += ["", "## Checks and hub validation", ""]
    lines.append((gate / "provenance.md").read_text().replace("## ", "### "))
    for name in ("checks.md", "fit-summary.md", "ppc-summary.md"):
        path = selected / name
        if path.exists():
            text = path.read_text().replace("## ", "### ")
            lines.append(text.split("-->", 1)[-1])
    log = selected / "hub-validation.log"
    if log.exists():
        lines += [
            "",
            "Hub validation log:",
            "",
            "```text",
            log.read_text().strip()[-3000:],
            "```",
        ]

    if selected.exists():
        sub = _hub_file(selected / "submission")
        mine = _quantiles(sub)
        series = {model_id: mine}
        for spec in args.compare:
            name, path = spec.split("=", 1)
            if Path(path).exists():
                series[name] = _quantiles(Path(path))
        observed = pd.read_csv(args.observed)
        observed = observed.loc[observed["as_of"] == observed["as_of"].max()]
        # The last 16 weeks, and the places with a case in them.
        start = pd.to_datetime(mine["target_end_date"]).min() - pd.Timedelta(
            weeks=12
        )
        observed = observed.loc[
            pd.to_datetime(observed["target_end_date"]) >= start
        ]
        active = set(observed.loc[observed["observation"] > 0, "location"])
        locs = sorted(mine["location"].unique())
        provinces = [
            loc for loc in locs if loc.count("-") == 1 and loc in active
        ]
        zones = [loc for loc in locs if loc.count("-") == 2]
        recent = (
            observed.loc[observed["location"].isin(zones)]
            .groupby("location")["observation"]
            .sum()
            .sort_values(ascending=False)
        )
        sample = list(recent.index[:6]) or zones[:6]
        lines += [
            "",
            "## Fan charts",
            "",
            f"Medians with 50% and 90% intervals: `{model_id}`, "
            + ", ".join(f"`{n}`" for n in list(series)[1:])
            + f". Dots are the counts reported as of "
            f"{observed['as_of'].max()}, over the last 16 weeks. Shown are "
            f"the {len(provinces)} provinces with a case in them, and the six "
            f"zones with the most. The file has "
            f"{sum(loc.count('-') == 1 for loc in locs)} provinces and "
            f"{len(zones)} health zones.",
            "",
        ]
        for name, places in (
            ("national", ["CD"]),
            ("provinces", provinces),
            ("zones", sample),
        ):
            path = figures / f"{name}.png"
            fan_charts(series, observed, places, path, name.capitalize())
            lines += [f"![{name}](figures/{name}.png)", ""]

    if args.table:
        lines += ["## Backtest against randomwalknowcast", ""]
        for spec in args.table:
            title, path = spec.split("=", 1)
            lines += [f"### {title}", ""]
            p_ = Path(path)
            if not p_.exists():
                lines += ["Pending.", ""]
            elif p_.suffix == ".csv":
                lines += [_markdown(pd.read_csv(p_)), ""]
            else:
                lines += [p_.read_text(), ""]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "index.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.out / 'index.md'}.")


if __name__ == "__main__":
    main()
