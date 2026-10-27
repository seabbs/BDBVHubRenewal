# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Docs pages for the submitted rounds, from the submission ledger.

The submission workflow records every live round on the
`submission-ledger` branch: `state.json` holds each round's outcome, and
`archive/<reference_date>/<run>/` holds the files of each run (the hub
file, the plot, the provenance and the pull request body).

`make_round_pages` turns that into one page per round at
`docs/submission/rounds/<reference_date>.md`, served at
`/submission/rounds/<reference_date>/`, which the hub pull request links to,
an index of rounds, and a snippet on the latest submission for the
Nowcast page. The pages are built with the docs and not committed.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["latest_run", "make_round_pages", "round_page"]


def latest_run(round_dir: Path) -> Path | None:
    """Return the newest run folder of a round's archive.

    Parameters
    ----------
    round_dir : pathlib.Path
        `archive/<reference_date>/` on the ledger branch.

    Returns
    -------
    pathlib.Path or None
        The run folder with the largest run id, or None if there is none.
    """
    runs = [p for p in round_dir.iterdir() if p.is_dir()]
    if not runs:
        return None

    def key(p: Path) -> tuple[int, str]:
        return (int(p.name), p.name) if p.name.isdigit() else (-1, p.name)

    return max(runs, key=key)


def _national_table(csv: Path, national: str = "CD") -> str:
    table = pd.read_csv(csv)
    cd = table.loc[table["location"] == national]
    if cd.empty:
        return ""
    q = (
        cd.groupby("target_end_date")["value"]
        .quantile(np.array([0.05, 0.25, 0.5, 0.75, 0.95]))
        .unstack()
        .round()
        .astype(int)
    )
    lines = [
        "| week ending | median | 50% interval | 90% interval |",
        "| --- | --- | --- | --- |",
    ]
    lines += [
        f"| {week} | {r[0.5]} | {r[0.25]} to {r[0.75]} | "
        f"{r[0.05]} to {r[0.95]} |"
        for week, r in q.iterrows()
    ]
    return "\n".join(lines) + "\n"


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def round_page(
    reference_date: str,
    entry: dict[str, Any],
    run_dir: Path | None,
    image: str | None,
) -> str:
    """Return the Markdown page of one round.

    Parameters
    ----------
    reference_date : str
        The round.
    entry : dict
        The round's record in the ledger's `state.json`.
    run_dir : pathlib.Path or None
        The newest archived run of the round.
    image : str or None
        File name of the plot, beside the page.

    Returns
    -------
    str
        The page.
    """
    prov = _read(run_dir / "provenance.json") if run_dir else {}
    run = _read(run_dir / "run.json") if run_dir else {}
    lines = [f"# Round {reference_date}", ""]
    lines.append(
        f"Nowcast and forecast submitted to the BDBV Modeling Hub for the "
        f"round with reference date {reference_date}."
    )
    lines.append("{ .page-subtitle }")
    lines.append("")
    facts = [
        ("Status", str(entry.get("status", "unknown"))),
        ("Produced by", str(entry.get("produced_by") or "n/a")),
        ("Data vintage (`as_of`)", str(entry.get("as_of") or "n/a")),
        ("Model", f"{run.get('model', 'n/a')} (`{run.get('config', 'n/a')}`)"),
    ]
    if entry.get("pr_url"):
        facts.append(("Hub pull request", f"<{entry['pr_url']}>"))
    if entry.get("run"):
        facts.append(("Workflow run", f"<{entry['run']}>"))
    if entry.get("updated"):
        facts.append(("Updated", str(entry["updated"])[:16].replace("T", " ")))
    lines += ["| | |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in facts]
    lines.append("")
    if prov.get("stand_in"):
        lines.append(
            f'!!! note "Stand-in"\n    {prov["stand_in"][0].upper()}'
            f"{prov['stand_in'][1:]}."
        )
        lines.append("")
    if prov.get("reason"):
        lines += [str(prov["reason"]), ""]
    if image:
        lines += [
            f"![Nowcast and forecast submitted for {reference_date}]({image})",
            "",
            "Bars are the published counts by onset week in the data used. "
            "Lines and bands are the submitted median, sample trajectories "
            "and 50% and 90% intervals.",
            "",
        ]
    csvs = sorted(run_dir.glob("*.csv")) if run_dir else []
    if csvs:
        nat = _national_table(csvs[0])
        if nat:
            lines += ["## National", "", nat]
    if prov.get("attempts"):
        lines += ["## Fallback chain", ""]
        lines += [
            "| step | outcome | why |",
            "| --- | --- | --- |",
        ]
        for a in prov["attempts"]:
            if a.get("status") in ("not needed", "not run"):
                continue
            why = a["reasons"][0] if a.get("reasons") else ""
            out = (
                "submitted" if a["id"] == prov.get("selected") else a["status"]
            )
            lines.append(f"| {a['label']} | {out} | {why} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def make_round_pages(
    ledger: Path | str | None,
    docs: Path | str = "docs",
    generated: Path | str = "docs/generated",
) -> list[str]:
    """Write the round pages, their index and the latest-submission snippet.

    Parameters
    ----------
    ledger : pathlib.Path or str or None
        A checkout of the `submission-ledger` branch. None, or a folder
        without `state.json`, means no round has been submitted yet.
    docs : pathlib.Path or str
        The docs folder. Pages go to `submission/rounds/` inside it.
    generated : pathlib.Path or str
        Where the `latest-submission.md` snippet goes.

    Returns
    -------
    list of str
        The reference dates with a page, newest first.
    """
    out = Path(docs) / "submission" / "rounds"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    state = _read(Path(ledger) / "state.json") if ledger else {}
    rounds: dict[str, Any] = state.get("rounds", {})
    done = []
    for ref in sorted(rounds, reverse=True):
        entry = rounds[ref]
        round_dir = Path(ledger or ".") / "archive" / ref
        run_dir = latest_run(round_dir) if round_dir.is_dir() else None
        image = None
        if run_dir and (run_dir / "nowcast.png").exists():
            image = f"{ref}.png"
            shutil.copy(run_dir / "nowcast.png", out / image)
        (out / f"{ref}.md").write_text(round_page(ref, entry, run_dir, image))
        done.append(ref)
    index = ["# Submitted rounds", ""]
    if done:
        index += ["| round | status | produced by |", "| --- | --- | --- |"]
        index += [
            f"| [{ref}]({ref}.md) | {rounds[ref].get('status', '')} | "
            f"{rounds[ref].get('produced_by', '')} |"
            for ref in done
        ]
    else:
        index.append("No round has been submitted yet.")
    (out / "index.md").write_text("\n".join(index) + "\n")
    snippet = Path(generated) / "latest-submission.md"
    snippet.parent.mkdir(parents=True, exist_ok=True)
    if done:
        ref = done[0]
        entry = rounds[ref]
        text = (
            f"**Latest submission:** "
            f"[round {ref}](submission/rounds/{ref}.md), "
            f"{entry.get('status', '')}, produced by "
            f"{entry.get('produced_by') or 'n/a'}. "
            "[All submitted rounds](submission/rounds/index.md)."
        )
    else:
        text = (
            "**Latest submission:** none yet. The submission workflow runs "
            "in dry-run mode until it is switched on."
        )
    snippet.write_text(text + "\n")
    return done
