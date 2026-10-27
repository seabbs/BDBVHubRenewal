"""Decide whether this scheduled run should forecast and submit.

The hub publishes a new data vintage by committing to `target-data/` and
`auxiliary-data/`, usually between Friday and Monday. This compares the
newest commit touching those paths, and the newest `as_of` in
`time-series.csv`, with the ledger of what earlier runs used. The ledger
(`state.json`) lives on the `submission-ledger` branch of this repo.

Rules, for the round whose window is open now:

- The round must be open in the hub's `tasks.json`.
- Run when the hub data changed since this round last ran (a new round,
  or a data fix during the window, which updates the round's PR).
- On a new round, only run on *fresh* data: an `as_of` no earlier round
  used. Last week's data are not resubmitted on Sunday morning, but if
  nothing new lands by the last `stale_fallback_hours` of the window the
  round runs on what there is rather than being missed.
- `force` (manual dispatch) skips the data checks but never the window,
  unless `ignore_window` is also set, which only a dry run may use.

Subcommands: `detect` writes the decision to `$GITHUB_OUTPUT`; `record`
updates the ledger after a run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import sys
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .common import (
    Runner,
    append_summary,
    read_json,
    set_outputs,
    subprocess_runner,
    write_json,
)
from .hubdata import TARGET, max_as_of
from .window import current_window, parse_now, window_for

HUB = "InsightNet-US/BDBV-Modeling-Hub"
WATCHED = (
    "target-data/time-series.csv",
    "auxiliary-data/daily-confirmed-case.csv",
    "auxiliary-data/weekly-reporting-delays-by-province.csv",
)


@dataclass(frozen=True)
class DataVersion:
    """The hub data as of now."""

    data_commit: str  # newest commit touching a watched path
    committed_at: str
    hub_ref: str  # hub `main` HEAD, which pins every input file
    as_of: str


class HubSource(Protocol):
    """Where the hub's state is read from."""

    def data_version(self) -> DataVersion:
        """Return the current data version."""
        ...

    def tasks(self, ref: str) -> dict[str, Any]:
        """Return `hub-config/tasks.json` at `ref`."""
        ...


class GhHubSource:
    """Read the hub through `gh api` and raw.githubusercontent.com."""

    def __init__(
        self,
        repo: str = HUB,
        paths: Sequence[str] = WATCHED,
        runner: Runner = subprocess_runner,
    ) -> None:
        self.repo, self.paths, self.runner = repo, paths, runner

    def _api(self, endpoint: str) -> Any:
        out = self.runner(["gh", "api", endpoint]).stdout
        return json.loads(out)

    def _raw(self, ref: str, path: str) -> str:
        url = f"https://raw.githubusercontent.com/{self.repo}/{ref}/{path}"
        with urllib.request.urlopen(url, timeout=120) as resp:
            return str(resp.read().decode("utf-8"))

    def data_version(self) -> DataVersion:
        """Return the newest data commit, `main` HEAD and latest `as_of`."""
        head = self._api(f"repos/{self.repo}/commits/main")["sha"]
        newest = ("", "")
        for path in self.paths:
            commits = self._api(
                f"repos/{self.repo}/commits?sha={head}&path={path}&per_page=1"
            )
            if commits:
                c = commits[0]
                stamp = c["commit"]["committer"]["date"]
                newest = max(newest, (stamp, c["sha"]))
        if not newest[1]:
            raise RuntimeError("no commits found for the watched hub paths")
        text = self._raw(head, "target-data/time-series.csv")
        return DataVersion(
            data_commit=newest[1],
            committed_at=newest[0],
            hub_ref=head,
            as_of=max_as_of(io.StringIO(text)),
        )

    def tasks(self, ref: str) -> dict[str, Any]:
        """Return `tasks.json` at `ref`."""
        data: dict[str, Any] = json.loads(
            self._raw(ref, "hub-config/tasks.json")
        )
        return data


class FixtureHubSource:
    """Read the hub's state from local files, for tests and dry runs."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def data_version(self) -> DataVersion:
        """Return the version in `version.json`."""
        return DataVersion(**read_json(self.directory / "version.json"))

    def tasks(self, ref: str) -> dict[str, Any]:  # noqa: ARG002
        """Return `tasks.json`."""
        data: dict[str, Any] = read_json(self.directory / "tasks.json")
        return data


def open_reference_dates(
    tasks: Mapping[str, Any], target: str = TARGET
) -> set[str]:
    """Reference dates accepted by the round that takes `target`."""
    out: set[str] = set()
    for rnd in tasks.get("rounds", []):
        for mt in rnd.get("model_tasks", []):
            ids = mt.get("task_ids", {})
            t = ids.get("target", {})
            targets = (t.get("required") or []) + (t.get("optional") or [])
            if target not in targets:
                continue
            r = ids.get("reference_date", {})
            out.update((r.get("required") or []) + (r.get("optional") or []))
    return out


@dataclass(frozen=True)
class Decision:
    """Whether to run, and the facts behind it."""

    should_run: bool
    reason: str
    reference_date: str
    window_open: bool
    hours_left: float
    round_open: bool
    new_for_round: bool
    fresh: bool
    data_commit: str
    hub_ref: str
    as_of: str


def empty_state() -> dict[str, Any]:
    """A ledger with nothing recorded."""
    return {"schema_version": 1, "rounds": {}}


def decide(
    state: dict[str, Any],
    version: DataVersion,
    tasks: dict[str, Any],
    now: dt.datetime,
    *,
    force: bool = False,
    ignore_window: bool = False,
    reference_date: str | None = None,
    stale_fallback_hours: float = 12.0,
) -> Decision:
    """Apply the run rules in the module docstring."""
    win = (
        window_for(dt.date.fromisoformat(reference_date))
        if reference_date
        else current_window(now)
    )
    ref = win.reference_date.isoformat()
    rounds: dict[str, Any] = state.get("rounds", {})
    entry = rounds.get(ref)
    new_for_round = entry is None or entry.get("data_commit") != (
        version.data_commit
    )
    earlier_as_of = {
        r.get("as_of") for k, r in rounds.items() if k < ref and r.get("as_of")
    }
    earlier_commits = {
        r.get("data_commit") for k, r in rounds.items() if k < ref
    }
    fresh = (
        version.as_of not in earlier_as_of
        if version.as_of
        else version.data_commit not in earlier_commits
    )
    window_open = win.is_open(now)
    hours_left = round(win.hours_left(now), 2)
    round_open = ref in open_reference_dates(tasks)

    def out(run: bool, reason: str) -> Decision:
        return Decision(
            should_run=run,
            reason=reason,
            reference_date=ref,
            window_open=window_open,
            hours_left=hours_left,
            round_open=round_open,
            new_for_round=new_for_round,
            fresh=fresh,
            data_commit=version.data_commit,
            hub_ref=version.hub_ref,
            as_of=version.as_of,
        )

    if not window_open and not ignore_window:
        return out(False, f"the window for {ref} is not open")
    if not round_open and not ignore_window:
        return out(False, f"{ref} is not an open round in the hub's tasks.json")
    if force:
        return out(True, "forced by manual dispatch")
    if not new_for_round:
        return out(False, f"no new hub data since {ref} last ran")
    if entry is not None:
        return out(True, "hub data changed during the window; updating")
    if fresh:
        return out(True, f"new data vintage {version.as_of or '?'}")
    if window_open and hours_left <= stale_fallback_hours:
        return out(
            True,
            f"no new vintage and {hours_left:.1f} h left; "
            "submitting on the latest data",
        )
    return out(False, "waiting for a new data vintage")


def record(
    state: dict[str, Any],
    reference_date: str,
    *,
    data_commit: str,
    as_of: str,
    status: str,
    produced_by: str = "",
    pr_url: str = "",
    run: str = "",
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    """Return the ledger with this round's outcome recorded."""
    new: dict[str, Any] = json.loads(json.dumps(state or empty_state()))
    new.setdefault("rounds", {})
    history = new["rounds"].get(reference_date, {}).get("history", [])
    entry = {
        "data_commit": data_commit,
        "as_of": as_of,
        "status": status,
        "produced_by": produced_by,
        "pr_url": pr_url,
        "run": run,
        "updated": (now or dt.datetime.now(dt.UTC)).isoformat(),
    }
    history.append({k: v for k, v in entry.items() if k != "history"})
    entry["history"] = history
    new["rounds"][reference_date] = entry
    return new


def _summary(d: Decision) -> str:
    verdict = "run" if d.should_run else "skip"
    return (
        f"### Data check: {verdict}\n\n{d.reason}.\n\n"
        f"| reference date | window open | hours left | round in tasks.json"
        f" | data vintage | hub data commit |\n"
        f"| --- | --- | --- | --- | --- | --- |\n"
        f"| {d.reference_date} | {d.window_open} | {d.hours_left} | "
        f"{d.round_open} | {d.as_of} | `{d.data_commit[:10]}` |"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    det = sub.add_parser("detect")
    det.add_argument("--state", type=Path, required=True)
    det.add_argument("--hub", default=HUB)
    det.add_argument("--fixture", type=Path, default=None)
    det.add_argument("--now", default=None)
    det.add_argument("--force", action="store_true")
    det.add_argument("--ignore-window", action="store_true")
    det.add_argument("--reference-date", default=None)
    det.add_argument("--stale-fallback-hours", type=float, default=12.0)
    det.add_argument("--out", type=Path, default=None)
    rec = sub.add_parser("record")
    rec.add_argument("--state", type=Path, required=True)
    rec.add_argument("--reference-date", required=True)
    rec.add_argument("--data-commit", required=True)
    rec.add_argument("--as-of", default="")
    rec.add_argument("--status", required=True)
    rec.add_argument("--produced-by", default="")
    rec.add_argument("--pr-url", default="")
    rec.add_argument("--run", default="")
    args = p.parse_args(argv)

    state = read_json(args.state) if args.state.exists() else empty_state()
    if args.cmd == "record":
        write_json(
            args.state,
            record(
                state,
                args.reference_date,
                data_commit=args.data_commit,
                as_of=args.as_of,
                status=args.status,
                produced_by=args.produced_by,
                pr_url=args.pr_url,
                run=args.run,
            ),
        )
        return 0

    if args.ignore_window and not args.force:
        p.error("--ignore-window needs --force")
    source: HubSource = (
        FixtureHubSource(args.fixture)
        if args.fixture
        else GhHubSource(args.hub)
    )
    version = source.data_version()
    decision = decide(
        state,
        version,
        source.tasks(version.hub_ref),
        parse_now(args.now),
        force=args.force,
        ignore_window=args.ignore_window,
        reference_date=args.reference_date,
        stale_fallback_hours=args.stale_fallback_hours,
    )
    set_outputs(asdict(decision))
    append_summary(_summary(decision))
    if args.out:
        write_json(args.out, asdict(decision))
    print(json.dumps(asdict(decision), indent=2), file=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
