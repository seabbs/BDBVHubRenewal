"""Round releases: when a round is released and what the release says."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest
from conftest import FakeRunner

from submission_ci.release_rounds import FOOTER, Plan, main, plan

REF = "2026-11-07"
PR_A = "https://github.com/InsightNet-US/BDBV-Modeling-Hub/pull/95"
PR_B = "https://github.com/InsightNet-US/BDBV-Modeling-Hub/pull/94"
RUN = "https://github.com/seabbs/BDBVHubRenewal/actions/runs/41691221605"
SHA = "16340ee3556f578b7482d79f98657e7611b60b7a"
AFTER = dt.datetime(2026, 11, 6, tzinfo=dt.UTC)
DURING = dt.datetime(2026, 11, 2, tzinfo=dt.UTC)


def model(pr: str, produced_by: str) -> dict[str, str]:
    return {
        "status": "submitted",
        "pr_url": pr,
        "run": RUN,
        "produced_by": produced_by,
        "code_commit": SHA,
    }


STATE = {
    "rounds": {
        REF: {
            "status": "submitted",
            "models": {
                "seabbs-rwnowcast": model(PR_B, "random walk on a nowcast"),
                "seabbs-renewal": model(PR_A, "renewal model"),
            },
        }
    }
}


def states(**by_pr: str) -> dict[str, str]:
    return {PR_A: "MERGED", PR_B: "MERGED"} | by_pr


def plan_with(
    now: dt.datetime = AFTER,
    pr: dict[str, str] | None = None,
    tags: tuple[str, ...] | set[str] = (),
) -> Plan:
    pr = pr or states()
    return plan(STATE, now, pr.__getitem__, lambda t: t in tags)[0]


def test_a_closed_round_with_every_pr_done_is_released() -> None:
    p = plan_with(pr=states(**{PR_B: "CLOSED"}))
    assert p.release is not None
    assert p.release.tag == f"round-{REF}"
    assert p.release.title == f"BDBV Hub round {REF}"
    assert p.release.target == SHA
    notes = p.release.notes
    assert notes.index("seabbs-renewal") < notes.index("seabbs-rwnowcast")
    assert "| seabbs-renewal | renewal model |" in notes
    assert f"[#95]({PR_A}) | merged |" in notes
    assert f"[#94]({PR_B}) | closed |" in notes
    assert f"[41691221605]({RUN})" in notes
    assert notes.rstrip().endswith(FOOTER)


def test_an_open_window_is_not_released() -> None:
    assert plan_with(now=DURING).release is None


def test_an_open_pr_holds_the_release() -> None:
    p = plan_with(pr=states(**{PR_B: "OPEN"}))
    assert p.release is None
    assert PR_B in p.reason


def test_an_existing_tag_is_skipped() -> None:
    p = plan_with(tags={f"round-{REF}"})
    assert p.release is None
    assert "exists" in p.reason


def test_the_round_commit_is_preferred_to_the_models() -> None:
    state = json.loads(json.dumps(STATE))
    state["rounds"][REF]["code_commit"] = "abc"
    p = plan(state, AFTER, states().__getitem__, lambda _t: False)[0]
    assert p.release is not None
    assert p.release.target == "abc"


def ledger(tmp_path: Path) -> Path:
    path = tmp_path / "state.json"
    path.write_text(json.dumps(STATE))
    return path


def runner(pr_state: str = "MERGED") -> FakeRunner:
    r = FakeRunner(real_git=False)
    r.on(["gh", "pr", "view"], f"{pr_state}\n")
    r.on(["gh", "api"], "", code=1)  # no tag yet
    r.on(["gh", "release", "create"], "")
    return r


def args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--state", str(ledger(tmp_path)),
        "--repo", "seabbs/BDBVHubRenewal",
        "--now", AFTER.isoformat(),
        *extra,
    ]  # fmt: skip


def creates(r: FakeRunner) -> list[list[str]]:
    return [c for c in r.gh_calls() if c[:3] == ["gh", "release", "create"]]


def test_a_dry_run_only_prints(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    r = runner()
    assert main(args(tmp_path, "--dry-run"), runner=r) == 0
    assert creates(r) == []
    assert f"would release round-{REF} at {SHA}" in capsys.readouterr().out


def test_a_live_run_creates_the_release_once(tmp_path: Path) -> None:
    r = runner()
    main(args(tmp_path), runner=r)
    (call,) = creates(r)
    assert call[3] == f"round-{REF}"
    assert call[call.index("--target") + 1] == SHA
    assert "--latest=false" in call


def test_nothing_is_created_while_a_pr_is_open(tmp_path: Path) -> None:
    r = runner("OPEN")
    main(args(tmp_path), runner=r)
    assert creates(r) == []
