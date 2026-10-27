"""`gh` is mocked; `git` runs for real against local bare repositories."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

import pytest
from conftest import REF, FakeRunner, make_samples, write_samples

from submission_ci.open_or_update_pr import (
    branch_name,
    find_pr,
    publish_plot,
    submit,
)

MID = "epiforecasts-hubrenewal"
MONDAY = dt.datetime(2026, 11, 2, 10, tzinfo=dt.UTC)
FRIDAY = dt.datetime(2026, 11, 6, 10, tzinfo=dt.UTC)


def git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def repos(tmp_path: Path) -> dict[str, Path]:
    """An upstream hub and the bot's fork, as bare repos."""
    seed = tmp_path / "seed"
    seed.mkdir()
    git("init", "-q", "-b", "main", cwd=seed)
    (seed / "hub-config").mkdir()
    (seed / "hub-config" / "tasks.json").write_text("{}")
    git("add", ".", cwd=seed)
    git("commit", "-q", "-m", "init", cwd=seed)
    up = tmp_path / "upstream.git"
    fork = tmp_path / "fork.git"
    git("clone", "-q", "--bare", str(seed), str(up))
    git("clone", "-q", "--bare", str(up), str(fork))
    # Upstream moves on after the fork was made; the fork is never synced.
    (seed / "README.md").write_text("new")
    git("add", ".", cwd=seed)
    git("commit", "-q", "-m", "later", cwd=seed)
    git("push", "-q", str(up), "main", cwd=seed)
    return {"upstream": up, "fork": fork, "tmp": tmp_path}


def files(tmp: Path) -> tuple[Path, Path]:
    sub = write_samples(tmp / "in" / "f.csv", make_samples())
    body = tmp / "in" / "body.md"
    body.write_text("body\n\n@seabbs")
    return sub, body


def go(runner: FakeRunner, repos: dict[str, Path], **kw: object) -> object:
    sub, body = files(repos["tmp"])
    args: dict[str, object] = {
        "submission": sub,
        "model_id": MID,
        "reference_date": REF,
        "title": "t",
        "body_file": body,
        "fork": "seabbs-bot/hub",
        "hub": "up/hub",
        "fork_url": str(repos["fork"]),
        "hub_url": str(repos["upstream"]),
        "dry_run": False,
        "now": MONDAY,
        "workdir": repos["tmp"] / f"w{len(runner.calls)}",
    }
    args.update(kw)
    return submit(runner, **args)  # type: ignore[arg-type]


def prs(*items: dict[str, object]) -> str:
    return json.dumps(list(items))


def pr(
    state: str = "OPEN", owner: str = "seabbs-bot", n: int = 7
) -> dict[str, object]:
    return {
        "number": n,
        "url": f"https://github.com/up/hub/pull/{n}",
        "state": state,
        "headRepositoryOwner": {"login": owner},
        "updatedAt": "2026-11-02T00:00:00Z",
    }


def test_branch_name() -> None:
    assert branch_name(MID, REF) == f"{MID}-{REF}"


def test_find_pr_filters_owner_and_prefers_open(runner: FakeRunner) -> None:
    runner.on(
        ["gh", "pr", "list"],
        prs(pr(owner="someone"), pr("CLOSED", n=3), pr(n=5)),
    )
    found = find_pr(runner, "up/hub", "seabbs-bot", "b")
    assert found is not None and found["number"] == 5


def test_creates_pr_from_branch_off_upstream_main(
    runner: FakeRunner, repos: dict[str, Path]
) -> None:
    runner.on(["gh", "pr", "list"], prs())
    runner.on(["gh", "pr", "create"], "https://github.com/up/hub/pull/12\n")
    out = go(runner, repos, metadata=repos["tmp"] / "in" / "f.csv")
    assert out.action == "created" and out.pr_number == 12 and out.pushed  # type: ignore[attr-defined]
    branch = f"{MID}-{REF}"
    tree = git("ls-tree", "-r", "--name-only", branch, cwd=repos["fork"])
    assert f"model-output/{MID}/{REF}-{MID}.csv" in tree
    assert f"model-metadata/{MID}.yml" in tree
    assert "README.md" in tree  # based on upstream main, not the stale fork
    create = next(
        c for c in runner.gh_calls() if c[:3] == ["gh", "pr", "create"]
    )
    assert create[create.index("--head") + 1] == f"seabbs-bot:{branch}"
    assert create[create.index("-R") + 1] == "up/hub"


def test_rerun_same_file_only_refreshes_body(
    runner: FakeRunner, repos: dict[str, Path]
) -> None:
    runner.on(["gh", "pr", "list"], prs())
    runner.on(["gh", "pr", "create"], "https://github.com/up/hub/pull/12\n")
    go(runner, repos)
    first = git("rev-parse", f"{MID}-{REF}", cwd=repos["fork"])
    runner.on(["gh", "pr", "list"], prs(pr(n=12)))
    runner.on(["gh", "pr", "edit"], "")
    out = go(runner, repos)
    assert out.action == "updated" and not out.pushed  # type: ignore[attr-defined]
    assert git("rev-parse", f"{MID}-{REF}", cwd=repos["fork"]) == first
    assert sum(c[:3] == ["gh", "pr", "create"] for c in runner.calls) == 1
    edit = next(c for c in runner.gh_calls() if c[:3] == ["gh", "pr", "edit"])
    assert edit[3] == "12" and "--body-file" in edit


def test_changed_data_force_pushes_to_open_pr(
    runner: FakeRunner, repos: dict[str, Path]
) -> None:
    runner.on(["gh", "pr", "list"], prs())
    runner.on(["gh", "pr", "create"], "https://github.com/up/hub/pull/12\n")
    go(runner, repos)
    first = git("rev-parse", f"{MID}-{REF}", cwd=repos["fork"])
    runner.on(["gh", "pr", "list"], prs(pr(n=12)))
    runner.on(["gh", "pr", "edit"], "")
    write_samples(repos["tmp"] / "in2" / "f.csv", make_samples(seed=9))
    out = go(runner, repos, submission=repos["tmp"] / "in2" / "f.csv")
    assert out.action == "updated" and out.pushed  # type: ignore[attr-defined]
    assert git("rev-parse", f"{MID}-{REF}", cwd=repos["fork"]) != first


@pytest.mark.parametrize("state", ["MERGED", "CLOSED"])
def test_merged_or_closed_round_is_not_resubmitted(
    runner: FakeRunner, repos: dict[str, Path], state: str
) -> None:
    runner.on(["gh", "pr", "list"], prs(pr(state)))
    out = go(runner, repos)
    assert out.action == f"skipped-{state.lower()}"  # type: ignore[attr-defined]
    assert not [c for c in runner.calls if c[0] == "git"]


def test_dry_run_never_writes(
    runner: FakeRunner, repos: dict[str, Path]
) -> None:
    runner.on(["gh", "pr", "list"], prs(pr(n=4)))
    out = go(runner, repos, dry_run=True)
    assert out.action == "dry-run" and "update PR #4" in out.message  # type: ignore[attr-defined]
    assert not [c for c in runner.calls if c[0] == "git"]
    assert [c[:3] for c in runner.gh_calls()] == [["gh", "pr", "list"]]


def test_dry_run_tolerates_gh_failure(
    runner: FakeRunner, repos: dict[str, Path]
) -> None:
    runner.on(["gh", "pr", "list"], "", code=1)
    out = go(runner, repos, dry_run=True)
    assert out.action == "dry-run" and "open a new PR" in out.message  # type: ignore[attr-defined]


def test_closed_window_blocks_live_submission(
    runner: FakeRunner, repos: dict[str, Path]
) -> None:
    out = go(runner, repos, now=FRIDAY)
    assert out.action == "skipped-window"  # type: ignore[attr-defined]
    assert not runner.calls


def test_publish_plot_pins_commit(
    runner: FakeRunner, repos: dict[str, Path]
) -> None:
    png = repos["tmp"] / "p.png"
    png.write_bytes(b"\x89PNG")
    url1 = publish_plot(
        runner,
        png,
        repo="seabbs-bot/hub",
        dest="plots/a.png",
        url=str(repos["fork"]),
        workdir=repos["tmp"] / "a1",
    )
    url2 = publish_plot(
        runner,
        png,
        repo="seabbs-bot/hub",
        dest="plots/b.png",
        url=str(repos["fork"]),
        workdir=repos["tmp"] / "a2",
    )
    assert url1.startswith("https://raw.githubusercontent.com/seabbs-bot/hub/")
    assert url1.endswith("/plots/a.png") and url1 != url2
    tree = git(
        "ls-tree", "-r", "--name-only", "submission-assets", cwd=repos["fork"]
    )
    assert tree.split() == ["plots/a.png", "plots/b.png"]
    # The assets branch shares no history with the hub.
    assert "hub-config" not in tree
