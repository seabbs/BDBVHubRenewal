"""Make a GitHub release of each submitted round, for Zenodo to archive.

A round is released once its window has closed and every hub pull
request it opened is merged or closed. The tag is `round-<reference
date>`, at the commit the ledger records the round was submitted from,
and the notes list each hub model's submission. A round whose tag exists
is skipped, so the job runs on every trigger and releases each round
once. Releases are not marked latest, so the package's own releases
stay latest. Zenodo archives each release and gives it a DOI.
"""

from __future__ import annotations

import argparse
import datetime as dt
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .common import (
    Runner,
    append_summary,
    read_json,
    set_outputs,
    subprocess_runner,
)
from .window import parse_now, window_for

DONE = {"MERGED", "CLOSED"}
"""Hub pull request states that let a round be released."""
FOOTER = "This was opened by a bot. Please ping @seabbs for any questions."


@dataclass(frozen=True)
class Release:
    """One round's release."""

    tag: str
    title: str
    target: str
    notes: str


@dataclass(frozen=True)
class Plan:
    """What to do with one ledger round."""

    reference_date: str
    release: Release | None
    reason: str


def submissions(entry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Each hub model's submission in a ledger round, by hub model id."""
    return {
        mid: m
        for mid, m in sorted(entry.get("models", {}).items())
        if m.get("status") == "submitted" and m.get("pr_url")
    }


def notes(
    ref: str,
    subs: Mapping[str, Mapping[str, Any]],
    states: Mapping[str, str],
    commit: str,
) -> str:
    """Release notes: one row per hub model, then the commit."""
    rows = [
        "| Hub model id | Chosen model | Run | Hub pull request | State |",
        "|---|---|---|---|---|",
    ]
    for mid, m in subs.items():
        run = m.get("run", "")
        pr = m["pr_url"]
        rows.append(
            f"| {mid} | {m.get('produced_by', '')} "
            f"| [{run.rsplit('/', 1)[-1]}]({run}) "
            f"| [#{pr.rsplit('/', 1)[-1]}]({pr}) "
            f"| {states[pr].lower()} |"
        )
    return "\n".join(
        [
            "Forecasts submitted to the BDBV Modeling Hub for the round "
            f"with reference date {ref}.",
            "",
            *rows,
            "",
            f"Submitted from commit {commit}.",
            "",
            FOOTER,
            "",
        ]
    )


def plan(
    state: Mapping[str, Any],
    now: dt.datetime,
    pr_state: Callable[[str], str],
    tag_exists: Callable[[str], bool],
) -> list[Plan]:
    """Decide which ledger rounds to release.

    Parameters
    ----------
    state : mapping
        The ledger's `state.json`.
    now : datetime.datetime
        The current time.
    pr_state : callable
        Returns a hub pull request's state (`OPEN`, `MERGED` or
        `CLOSED`) from its URL.
    tag_exists : callable
        Whether a tag exists in this repository.

    Returns
    -------
    list of Plan
        One per round, with the release to make or why there is none.
    """
    out = []
    for ref, entry in sorted(state.get("rounds", {}).items()):
        tag = f"round-{ref}"
        subs = submissions(entry)
        models = list(subs.values())
        commit = entry.get("code_commit") or (
            models[0].get("code_commit", "") if models else ""
        )
        if not subs:
            out.append(Plan(ref, None, "no submission"))
            continue
        if tag_exists(tag):
            out.append(Plan(ref, None, f"{tag} exists"))
            continue
        if now < window_for(dt.date.fromisoformat(ref)).closes:
            out.append(Plan(ref, None, "the window has not closed"))
            continue
        states = {m["pr_url"]: pr_state(m["pr_url"]) for m in models}
        still_open = [u for u, s in states.items() if s not in DONE]
        if still_open:
            out.append(Plan(ref, None, f"open: {', '.join(still_open)}"))
            continue
        if not commit:
            out.append(Plan(ref, None, "no submitted commit in the ledger"))
            continue
        release = Release(
            tag,
            f"BDBV Hub round {ref}",
            commit,
            notes(ref, subs, states, commit),
        )
        out.append(Plan(ref, release, "ready"))
    return out


def gh_pr_state(runner: Runner) -> Callable[[str], str]:
    """Read a pull request's state with `gh`."""
    return lambda url: runner(
        ["gh", "pr", "view", url, "--json", "state", "-q", ".state"]
    ).stdout.strip()


def gh_tag_exists(runner: Runner, repo: str) -> Callable[[str], bool]:
    """Check for a tag with `gh api`."""
    return lambda tag: (
        runner(
            ["gh", "api", f"repos/{repo}/git/ref/tags/{tag}"], check=False
        ).returncode
        == 0
    )


def create(runner: Runner, repo: str, release: Release) -> None:
    """Make the release, tagging its target commit."""
    runner(
        [
            "gh", "release", "create", release.tag,
            "-R", repo,
            "--target", release.target,
            "--title", release.title,
            "--notes", release.notes,
            "--latest=false",
        ]
    )  # fmt: skip


def main(
    argv: Sequence[str] | None = None, runner: Runner = subprocess_runner
) -> int:
    """Release the rounds that are ready, or print them on a dry run."""
    p = argparse.ArgumentParser(prog="release-rounds")
    p.add_argument("--state", required=True, help="Ledger state.json.")
    p.add_argument("--repo", required=True, help="owner/name.")
    p.add_argument("--now", help="ISO time, for testing.")
    p.add_argument("--dry-run", action="store_true", help="Only print.")
    args = p.parse_args(argv)
    plans = plan(
        read_json(args.state),
        parse_now(args.now),
        gh_pr_state(runner),
        gh_tag_exists(runner, args.repo),
    )
    lines = ["| Round | Release | Reason |", "|---|---|---|"]
    released = 0
    for pl in plans:
        tag = pl.release.tag if pl.release else ""
        print(f"{pl.reference_date}: {tag or 'skip'} ({pl.reason})")
        if pl.release and args.dry_run:
            print(f"  would release {tag} at {pl.release.target}")
            print(pl.release.notes)
        elif pl.release:
            create(runner, args.repo, pl.release)
            released += 1
        lines.append(f"| {pl.reference_date} | {tag} | {pl.reason} |")
    append_summary("### Round releases\n\n" + "\n".join(lines) + "\n")
    set_outputs({"released": released})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
