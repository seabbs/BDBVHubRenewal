"""Push a submission to the bot's fork and open or update the hub PR.

This follows the approach proven in `epiforecasts/cityforecasts`
(`bin/submit_to_hub.sh`):

1. Clone the bot's fork and fetch the hub's `main` as `upstream`.
2. Branch `<model_id>-<reference_date>` off `upstream/main`, not the
   fork's `main`, so the fork never needs syncing and one round's PR
   never waits on another's.
3. Copy the file into `model-output/<model_id>/` (and the metadata into
   `model-metadata/` when it is new or changed), commit and force-push.
4. Find an open PR from `<fork owner>:<branch>`. If there is one, edit
   its title and body; otherwise open one with `gh pr create`.

It is idempotent for a round. A re-run with the same file pushes nothing
and only refreshes the body. A run with changed data force-pushes the
branch, which updates the open PR. If the round's PR was merged or
closed, nothing is pushed unless `--allow-resubmit` is given, so a
closed PR is never reopened behind the organisers' backs.

`publish-plot` pushes the nowcast plot to a separate `submission-assets`
branch of the fork and prints a raw URL pinned to that commit. The image
then renders in the PR body without being part of the PR's diff, which
the hub's validation would reject. GitHub has no API for uploading PR
attachments, so this is the dependable way to embed an image.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .build_pr_body import HUB
from .common import (
    CommandError,
    Runner,
    append_summary,
    set_outputs,
    subprocess_runner,
)
from .window import window_for

ASSETS_BRANCH = "submission-assets"


@dataclass
class Outcome:
    """What `submit` did."""

    action: str
    branch: str
    pr_url: str = ""
    pr_number: int | None = None
    pushed: bool = False
    message: str = ""


def branch_name(model_id: str, reference_date: str) -> str:
    """The fork branch for one round."""
    return f"{model_id}-{reference_date}"


def github_url(repo: str) -> str:
    """HTTPS clone URL (credentials come from `gh auth setup-git`)."""
    return f"https://github.com/{repo}.git"


def find_pr(
    runner: Runner, hub: str, fork_owner: str, branch: str
) -> dict[str, object] | None:
    """The newest PR from `fork_owner:branch` into the hub, any state."""
    out = runner(
        [
            "gh",
            "pr",
            "list",
            "-R",
            hub,
            "--head",
            branch,
            "--state",
            "all",
            "--json",
            "number,url,state,headRepositoryOwner,updatedAt",
        ]
    ).stdout
    prs = [
        p
        for p in json.loads(out or "[]")
        if p.get("headRepositoryOwner", {}).get("login") == fork_owner
    ]
    if not prs:
        return None
    prs.sort(key=lambda p: str(p.get("updatedAt", "")), reverse=True)
    open_prs = [p for p in prs if p.get("state") == "OPEN"]
    found: dict[str, object] = (open_prs or prs)[0]
    return found


def _git(runner: Runner, repo: Path, *args: str, check: bool = True) -> str:
    return runner(["git", "-C", str(repo), *args], check=check).stdout.strip()


def _identity(name: str, email: str) -> list[str]:
    return ["-c", f"user.name={name}", "-c", f"user.email={email}"]


def publish_plot(
    runner: Runner,
    plot: Path,
    *,
    repo: str,
    dest: str,
    url: str | None = None,
    branch: str = ASSETS_BRANCH,
    author: tuple[str, str] = ("seabbs-bot", "signin@samabbott.co.uk"),
    workdir: Path | None = None,
) -> str:
    """Commit `plot` to `branch` of `repo` and return a pinned raw URL."""
    work = Path(workdir or tempfile.mkdtemp()) / "assets"
    work.mkdir(parents=True, exist_ok=True)
    remote = url or github_url(repo)
    _git(runner, work, "init", "-q")
    _git(runner, work, "remote", "add", "origin", remote)
    exists = _git(runner, work, "ls-remote", "--heads", "origin", branch)
    if exists:
        _git(runner, work, "fetch", "-q", "--depth", "1", "origin", branch)
        _git(runner, work, "checkout", "-q", "-B", branch, "FETCH_HEAD")
    else:
        _git(runner, work, "checkout", "-q", "--orphan", branch)
    target = work / dest
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(plot, target)
    _git(runner, work, "add", dest)
    runner(
        [
            "git",
            *_identity(*author),
            "-C",
            str(work),
            "commit",
            "-q",
            "-m",
            f"Add {dest}",
        ]
    )
    _git(runner, work, "push", "-q", "origin", branch)
    sha = _git(runner, work, "rev-parse", "HEAD")
    return f"https://raw.githubusercontent.com/{repo}/{sha}/{dest}"


def submit(
    runner: Runner,
    *,
    submission: Path,
    model_id: str,
    reference_date: str,
    title: str,
    body_file: Path,
    fork: str,
    hub: str = HUB,
    metadata: Path | None = None,
    fork_url: str | None = None,
    hub_url: str | None = None,
    dry_run: bool = True,
    allow_resubmit: bool = False,
    ignore_window: bool = False,
    now: dt.datetime | None = None,
    author: tuple[str, str] = ("seabbs-bot", "signin@samabbott.co.uk"),
    workdir: Path | None = None,
) -> Outcome:
    """Push the submission and open or update the round's PR."""
    branch = branch_name(model_id, reference_date)
    fork_owner = fork.split("/")[0]
    win = window_for(dt.date.fromisoformat(reference_date))
    moment = now or dt.datetime.now(dt.UTC)
    if not win.is_open(moment) and not ignore_window and not dry_run:
        return Outcome(
            "skipped-window",
            branch,
            message=f"the window for {reference_date} closed at "
            f"{win.closes.isoformat()}",
        )
    try:
        existing = find_pr(runner, hub, fork_owner, branch)
    except CommandError as err:
        if not dry_run:
            raise
        existing = None
        print(
            f"(dry run) could not look up existing PRs: {err}", file=sys.stderr
        )
    state = str(existing.get("state")) if existing else ""
    if state in ("MERGED", "CLOSED") and not allow_resubmit:
        return Outcome(
            f"skipped-{state.lower()}",
            branch,
            pr_url=str(existing["url"]),  # type: ignore[index]
            message=f"the round's PR is {state.lower()}; not resubmitting",
        )
    if state in ("MERGED", "CLOSED"):
        branch = f"{branch}-{moment.strftime('%Y%m%d%H%M')}"
        existing = None
    if dry_run:
        verb = (
            f"update PR #{existing['number']}" if existing else "open a new PR"
        )
        return Outcome(
            "dry-run",
            branch,
            pr_url=str(existing["url"]) if existing else "",
            message=f"would push {branch} to {fork} and {verb} on {hub}",
        )

    work = Path(workdir or tempfile.mkdtemp()) / "hub"
    work.parent.mkdir(parents=True, exist_ok=True)
    _git(
        runner,
        work.parent,
        "clone",
        "-q",
        "--filter=blob:none",
        fork_url or github_url(fork),
        str(work),
    )
    _git(runner, work, "remote", "add", "upstream", hub_url or github_url(hub))
    _git(runner, work, "fetch", "-q", "upstream", "main")
    _git(runner, work, "checkout", "-q", "-B", branch, "upstream/main")
    # The hub file name always carries the submitting model id, even when
    # a fallback (such as persistence) produced it.
    rel = f"model-output/{model_id}/{reference_date}-{model_id}.csv"
    (work / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(submission, work / rel)
    paths = [rel]
    if metadata is not None:
        meta_rel = f"model-metadata/{model_id}.yml"
        (work / meta_rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(metadata, work / meta_rel)
        paths.append(meta_rel)
    _git(runner, work, "add", *paths)
    staged = runner(
        ["git", "-C", str(work), "diff", "--cached", "--quiet"], check=False
    )
    if staged.returncode == 0:
        return Outcome(
            "unchanged-on-hub",
            branch,
            message=f"{rel} is already on {hub} main; nothing to submit",
        )
    runner(
        [
            "git",
            *_identity(*author),
            "-C",
            str(work),
            "commit",
            "-q",
            "-m",
            f"{model_id} submission for {reference_date}",
        ]
    )
    remote_head = _git(runner, work, "ls-remote", "--heads", "origin", branch)
    same = False
    if remote_head:
        _git(runner, work, "fetch", "-q", "origin", branch)
        same = _git(runner, work, "rev-parse", "FETCH_HEAD^{tree}") == _git(
            runner, work, "rev-parse", "HEAD^{tree}"
        )
    if not same:
        _git(runner, work, "push", "-q", "--force", "origin", branch)

    if existing:
        number = int(str(existing["number"]))
        runner(
            [
                "gh",
                "pr",
                "edit",
                str(number),
                "-R",
                hub,
                "--title",
                title,
                "--body-file",
                str(body_file),
            ]
        )
        return Outcome(
            "updated",
            branch,
            pr_url=str(existing["url"]),
            pr_number=number,
            pushed=not same,
            message="file unchanged; body refreshed"
            if same
            else "pushed the new file to the open PR",
        )
    url = (
        runner(
            [
                "gh",
                "pr",
                "create",
                "-R",
                hub,
                "--base",
                "main",
                "--head",
                f"{fork_owner}:{branch}",
                "--title",
                title,
                "--body-file",
                str(body_file),
            ]
        )
        .stdout.strip()
        .splitlines()[-1]
    )
    created = int(url.rsplit("/", 1)[-1]) if url[-1:].isdigit() else None
    return Outcome(
        "created", branch, pr_url=url, pr_number=created, pushed=True
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("publish-plot")
    pp.add_argument("--plot", type=Path, required=True)
    pp.add_argument("--repo", required=True)
    pp.add_argument("--dest", required=True)
    pp.add_argument("--url", default=None)
    s = sub.add_parser("submit")
    s.add_argument("--submission", type=Path, required=True)
    s.add_argument("--metadata", type=Path, default=None)
    s.add_argument("--model-id", required=True)
    s.add_argument("--reference-date", required=True)
    s.add_argument("--title", required=True)
    s.add_argument("--body-file", type=Path, required=True)
    s.add_argument("--fork", required=True)
    s.add_argument("--hub", default=HUB)
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--allow-resubmit", action="store_true")
    s.add_argument("--ignore-window", action="store_true")
    args = p.parse_args(argv)
    if args.cmd == "publish-plot":
        url = publish_plot(
            subprocess_runner,
            args.plot,
            repo=args.repo,
            dest=args.dest,
            url=args.url,
        )
        set_outputs({"url": url})
        print(url)
        return 0
    outcome = submit(
        subprocess_runner,
        submission=args.submission,
        metadata=args.metadata,
        model_id=args.model_id,
        reference_date=args.reference_date,
        title=args.title,
        body_file=args.body_file,
        fork=args.fork,
        hub=args.hub,
        dry_run=args.dry_run,
        allow_resubmit=args.allow_resubmit,
        ignore_window=args.ignore_window,
    )
    set_outputs(asdict(outcome))
    line = f"Hub PR: {outcome.action}. {outcome.message} {outcome.pr_url}"
    append_summary(f"### {line.strip()}")
    if args.dry_run:
        print("----- PR title -----")
        print(args.title)
        print("----- PR body -----")
        print(args.body_file.read_text())
    print(json.dumps(asdict(outcome), indent=2))
    return 1 if outcome.action == "skipped-window" else 0


if __name__ == "__main__":
    raise SystemExit(main())
