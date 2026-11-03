# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Docs pages for the submitted rounds, from the submission ledger.

The submission workflow records every live round on the
`submission-ledger` branch: `state.json` holds each round's outcome, and
`archive/<reference_date>/<run>/` holds the files of each run (the hub
file, the plot, the provenance and the pull request body).

Each round records every hub model it submitted under `models`, keyed by
hub model id, with the folder of that model's archived files.
`make_round_pages` turns that into one page per round at
`docs/submission/rounds/<reference_date>.md`, served at
`/submission/rounds/<reference_date>/`, which the hub pull requests link
to, with one tab per hub model id. The index is the submission log, one
table per hub model id, and a snippet on the latest submission goes to
the Nowcast page. A new hub model id appears without code changes.

A round's page is frozen once its hub pull request has merged:
`freeze_round` writes it to `pages/` on the ledger branch, once, and
later docs builds copy that file as it is. Rounds not yet frozen are
built from the ledger at each docs build.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "freeze_round",
    "gh_pr_state",
    "latest_run",
    "make_round_pages",
    "round_page",
]

FROZEN = "pages"
"""Folder on the ledger branch that holds the frozen round pages."""


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


def gh_pr_state(url: str) -> str:
    """Return a pull request's state (open, closed or merged) from `gh`."""
    out = subprocess.run(
        ["gh", "pr", "view", url, "--json", "state", "-q", ".state"],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip().lower()


def _link(url: str, text: str) -> str:
    return f"[{text}]({url})" if url else "n/a"


def _commit(model: dict[str, Any]) -> str:
    """Link the model's commit in the repo its workflow run belongs to."""
    sha = str(model.get("code_commit") or "")
    repo = str(model.get("run") or "").split("/actions/")[0]
    return f"[`{sha[:7]}`]({repo}/commit/{sha})" if sha and repo else "n/a"


def _files(ledger: Path, model: dict[str, Any]) -> dict[str, Any]:
    """Return a model's archived run, provenance and checks."""
    d = ledger / str(model.get("archive", ""))
    return {
        "dir": d,
        "run": _read(d / "run.json"),
        "prov": _read(d / "provenance.json"),
        "checks": _read(d / "checks.json").get("checks", []),
    }


def _model_section(
    ref: str, model_id: str, model: dict[str, Any], ledger: Path, out: Path
) -> list[str]:
    """Return one hub model's part of a round page, and copy its plot."""
    f = _files(ledger, model)
    run, prov = f["run"], f["prov"]
    facts = [
        ("Status", str(model.get("status", "unknown"))),
        ("Produced by", str(model.get("produced_by") or "n/a")),
        ("Model", f"{run.get('model', 'n/a')} (`{run.get('config', 'n/a')}`)"),
        ("Gate status", str(prov.get("status", "n/a"))),
        ("Hub pull request", _link(str(model.get("pr_url", "")), "link")),
        ("Workflow run", _link(str(model.get("run", "")), "link")),
        ("Commit", _commit(model)),
    ]
    lines = ["| | |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in facts]
    lines.append("")
    if prov.get("reason"):
        lines += [str(prov["reason"]), ""]
    if (f["dir"] / "nowcast.png").exists():
        image = f"{ref}-{model_id}.png"
        shutil.copy(f["dir"] / "nowcast.png", out / image)
        lines += [
            f"![Nowcast and forecast submitted by {model_id} for {ref}]"
            f"({image})",
            "",
            "Bars are the published counts by onset week in the data used. "
            "Lines and bands are the submitted median, sample trajectories "
            "and 50% and 90% intervals.",
            "",
        ]
    csvs = sorted(f["dir"].glob(f"*-{model_id}.csv"))
    nat = _national_table(csvs[0]) if csvs else ""
    if nat:
        lines += ["**National**", "", nat]
    if f["checks"]:
        lines += ["**Checks**", "", "| check | status | summary |"]
        lines.append("| --- | --- | --- |")
        lines += [
            f"| {c['name']} | {c['status']} | {c.get('summary', '')} |"
            for c in f["checks"]
        ]
        lines.append("")
    tried = [
        a
        for a in prov.get("attempts", [])
        if a.get("status") not in ("not needed", "not run")
    ]
    if tried:
        lines += ["**Candidates**", "", "| step | outcome | why |"]
        lines.append("| --- | --- | --- |")
        for a in tried:
            why = a["reasons"][0] if a.get("reasons") else ""
            outcome = (
                "submitted" if a["id"] == prov.get("selected") else a["status"]
            )
            lines.append(f"| {a['label']} | {outcome} | {why} |")
        lines.append("")
    return lines


def round_page(
    reference_date: str, entry: dict[str, Any], ledger: Path, out: Path
) -> str:
    """Return the Markdown page of one round, with one tab per hub model.

    Parameters
    ----------
    reference_date : str
        The round.
    entry : dict
        The round's record in the ledger's `state.json`.
    ledger : pathlib.Path
        A checkout of the `submission-ledger` branch.
    out : pathlib.Path
        Folder of the page, where each model's plot is copied.

    Returns
    -------
    str
        The page.
    """
    lines = [f"# Round {reference_date}", ""]
    lines.append(
        f"Nowcasts and forecasts submitted to the BDBV Modeling Hub for the "
        f"round with reference date {reference_date}."
    )
    lines.append("{ .page-subtitle }")
    lines.append("")
    lines.append(f"Data vintage (`as_of`): {entry.get('as_of') or 'n/a'}.")
    lines.append("")
    for model_id, model in sorted(entry.get("models", {}).items()):
        lines += [f'=== "{model_id}"', ""]
        section = _model_section(reference_date, model_id, model, ledger, out)
        text = "\n".join(section).splitlines()
        lines += [f"    {x}" if x else "" for x in text]
    return "\n".join(lines) + "\n"


def _render(ledger: Path, ref: str, entry: dict[str, Any], out: Path) -> None:
    """Write a round's page, and its plots, into `out`."""
    (out / f"{ref}.md").write_text(round_page(ref, entry, ledger, out))


def freeze_round(ledger: Path | str, reference_date: str) -> bool:
    """Write a round's page to the ledger's `pages/`, unless it is there.

    Parameters
    ----------
    ledger : pathlib.Path or str
        A checkout of the `submission-ledger` branch.
    reference_date : str
        The round, which must be in the ledger's `state.json`.

    Returns
    -------
    bool
        True if the page was written, False if it was already frozen.
    """
    ledger = Path(ledger)
    out = ledger / FROZEN
    if (out / f"{reference_date}.md").exists():
        return False
    entry = _read(ledger / "state.json")["rounds"][reference_date]
    out.mkdir(exist_ok=True)
    _render(ledger, reference_date, entry, out)
    return True


def _log(
    rounds: dict[str, Any], ledger: Path, pr_state: Callable[[str], str]
) -> list[str]:
    """Return the submission log, one table per hub model id."""
    ids = sorted({m for r in rounds.values() for m in r.get("models", {})})
    lines = []
    for model_id in ids:
        lines += [f"## {model_id}", ""]
        lines += [
            "| round | run | commit | chosen model | gate status | hub PR "
            "| PR state |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for ref in sorted(rounds, reverse=True):
            model = rounds[ref].get("models", {}).get(model_id)
            if model is None:
                continue
            f = _files(ledger, model)
            run_url = str(model.get("run", ""))
            pr = str(model.get("pr_url", ""))
            lines.append(
                f"| [{ref}]({ref}.md) "
                f"| {_link(run_url, run_url.rsplit('/', 1)[-1])} "
                f"| {_commit(model)} "
                f"| {f['run'].get('model', 'n/a')} "
                f"| {f['prov'].get('status', 'n/a')} "
                f"| {_link(pr, '#' + pr.rsplit('/', 1)[-1])} "
                f"| {pr_state(pr) if pr else 'n/a'} |"
            )
        lines.append("")
    return lines


def make_round_pages(
    ledger: Path | str | None,
    docs: Path | str = "docs",
    generated: Path | str = "docs/generated",
    pr_state: Callable[[str], str] = gh_pr_state,
) -> list[str]:
    """Write the round pages, the submission log and the latest snippet.

    Parameters
    ----------
    ledger : pathlib.Path or str or None
        A checkout of the `submission-ledger` branch. None, or a folder
        without `state.json`, means no round has been submitted.
    docs : pathlib.Path or str
        The docs folder. Pages go to `submission/rounds/` inside it.
    generated : pathlib.Path or str
        Where the `latest-submission.md` snippet goes.
    pr_state : callable
        Returns a hub pull request's state from its URL.

    Returns
    -------
    list of str
        The reference dates with a page, newest first.
    """
    out = Path(docs) / "submission" / "rounds"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    root = Path(ledger or ".")
    state = _read(root / "state.json") if ledger else {}
    rounds: dict[str, Any] = state.get("rounds", {})
    done = sorted(rounds, reverse=True)
    frozen = root / FROZEN
    for ref in done:
        if (frozen / f"{ref}.md").exists():
            for f in frozen.glob(f"{ref}*"):
                shutil.copy(f, out / f.name)
        else:
            _render(root, ref, rounds[ref], out)
    index = ["# Submission log", ""]
    if done:
        index += [
            "Every round each hub model was submitted for, newest first.",
            "The round links to its page.",
            "",
        ]
        index += _log(rounds, root, pr_state)
    (out / "index.md").write_text("\n".join(index) + "\n")
    snippet = Path(generated) / "latest-submission.md"
    snippet.parent.mkdir(parents=True, exist_ok=True)
    if done:
        ref = done[0]
        models = rounds[ref].get("models", {})
        each = ", ".join(
            f"{model_id} {m.get('status', '')} "
            f"({m.get('produced_by') or 'n/a'})"
            for model_id, m in sorted(models.items())
        )
        snippet.write_text(
            f"**Latest submission:** "
            f"[round {ref}](submission/rounds/{ref}.md): {each}. "
            "[Submission log](submission/rounds/index.md).\n"
        )
    else:
        snippet.write_text("")
    return done
