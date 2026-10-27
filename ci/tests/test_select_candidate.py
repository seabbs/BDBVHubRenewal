"""Each path through the fallback chain."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import make_samples, write_json, write_samples

from submission_ci.common import read_json
from submission_ci.policy import load_policy
from submission_ci.select_candidate import main, run_select, triage

POLICY = load_policy(Path(__file__).parents[1] / "policy.toml")


def candidate(
    root: Path,
    cid: str,
    fit: str | None = "pass",
    checks: str = "pass",
    submission: bool = True,
    ppc: str | None = "pass",
) -> None:
    d = root / cid
    d.mkdir(parents=True)
    if fit is not None and ppc is not None:
        write_json(
            d / "ppc-summary.json",
            {
                "status": ppc,
                "failures": ["90% interval coverage of total overall is 0.3"]
                if ppc in ("fail", "error")
                else [],
                "warnings": ["90% interval coverage of total overall is 0.6"]
                if ppc == "warn"
                else [],
            },
        )
    if fit is not None:
        write_json(
            d / "fit-summary.json",
            {
                "status": fit,
                "failures": ["max R-hat is 1.4, past the failure threshold 1.1"]
                if fit == "fail"
                else [],
                "warnings": [
                    "min bulk ESS is 80, past the warning threshold 100"
                ]
                if fit == "warn"
                else [],
                "headline": {},
            },
        )
    write_json(
        d / "checks.json",
        {
            "status": checks,
            "checks": [
                {
                    "name": "growth",
                    "status": checks,
                    "summary": "CD-IT: median grows 9 times",
                }
            ],
        },
    )
    write_json(d / "run.json", {"model": "renewal", "config": cid})
    if submission:
        write_samples(
            d / "submission" / "model-output" / "m" / "f.csv", make_samples()
        )


def chain(root: Path, **spec: tuple[str | None, str]) -> None:
    for cid, (fit, checks) in spec.items():
        candidate(root, cid.replace("_", "-"), fit, checks)


def always(status: str):  # type: ignore[no-untyped-def]
    def run(_sub: Path, log: Path) -> str:
        log.write_text(status)
        return status

    return run


def test_chain_order_and_budget() -> None:
    assert [c.id for c in POLICY.chain()] == [
        "main",
        "retry-1",
        "retry-2",
        "fallback",
        "persistence",
    ]
    assert [c.id for c in POLICY.stage("primary")] == ["main", "fallback"]
    assert [c.id for c in POLICY.stage("retry")] == ["retry-1", "retry-2"]


def test_main_passes(tmp_path: Path) -> None:
    c = tmp_path / "c"
    chain(
        c,
        main=("pass", "pass"),
        fallback=("pass", "pass"),
        persistence=(None, "pass"),
    )
    assert not triage(POLICY, c)
    prov = run_select(POLICY, c, tmp_path / "g", always("pass"), retried=False)
    assert prov["selected"] == "main" and prov["status"] == "pass"
    assert prov["reason"] == "The main model cleared every check."
    assert (tmp_path / "g" / "selected" / "submission").is_dir()
    states = {a["id"]: a["status"] for a in prov["attempts"]}
    assert (
        states["fallback"] == "not needed" and states["retry-1"] == "not needed"
    )


def test_main_warns_and_is_submitted_with_warning(tmp_path: Path) -> None:
    c = tmp_path / "c"
    chain(c, main=("warn", "warn"), persistence=(None, "pass"))
    prov = run_select(POLICY, c, tmp_path / "g", always("pass"), retried=False)
    assert prov["selected"] == "main" and prov["status"] == "warn"
    assert any("convergence" in w for w in prov["warnings"])
    assert any("growth" in w for w in prov["warnings"])


def test_retry_passes(tmp_path: Path) -> None:
    c = tmp_path / "c"
    chain(
        c,
        main=("fail", "pass"),
        retry_1=("pass", "pass"),
        retry_2=("pass", "pass"),
        fallback=("pass", "pass"),
        persistence=(None, "pass"),
    )
    assert triage(POLICY, c)
    prov = run_select(POLICY, c, tmp_path / "g", always("pass"), retried=True)
    assert prov["selected"] == "retry-1"
    assert prov["produced_by"].startswith("main model, retry 1")
    assert (
        "Main model was ruled out (convergence: max R-hat is 1.4"
        in prov["reason"]
    )


def test_main_crash_triggers_retry(tmp_path: Path) -> None:
    c = tmp_path / "c"
    candidate(c, "main", fit="error", submission=False)
    assert triage(POLICY, c)
    (tmp_path / "empty").mkdir()
    assert triage(POLICY, tmp_path / "empty")  # main never produced output


def test_fallback_is_used(tmp_path: Path) -> None:
    c = tmp_path / "c"
    chain(
        c,
        main=("fail", "pass"),
        retry_1=("pass", "fail"),
        retry_2=("error", "pass"),
        fallback=("pass", "pass"),
        persistence=(None, "pass"),
    )
    prov = run_select(POLICY, c, tmp_path / "g", always("pass"), retried=True)
    assert prov["selected"] == "fallback" and prov["kind"] == "fallback"
    assert "growth" in prov["reason"]


def test_persistence_is_the_last_resort(tmp_path: Path) -> None:
    c = tmp_path / "c"
    chain(
        c,
        main=("fail", "pass"),
        retry_1=("fail", "pass"),
        retry_2=("fail", "pass"),
        fallback=("pass", "fail"),
        persistence=(None, "pass"),
    )
    prov = run_select(POLICY, c, tmp_path / "g", always("pass"), retried=True)
    assert prov["selected"] == "persistence"
    assert prov["produced_by"] == "persistence baseline"


def test_everything_fails_round_skipped(tmp_path: Path) -> None:
    c = tmp_path / "c"
    chain(
        c,
        main=("fail", "pass"),
        retry_1=("fail", "pass"),
        retry_2=("fail", "pass"),
        fallback=("pass", "fail"),
        persistence=(None, "fail"),
    )
    prov = run_select(POLICY, c, tmp_path / "g", always("pass"), retried=True)
    assert prov["selected"] is None and prov["status"] == "skip"
    assert prov["reason"].startswith("Every candidate was ruled out")
    md = (tmp_path / "g" / "provenance.md").read_text()
    assert md.startswith("## Round skipped")


def test_hub_validation_failure_moves_down_the_chain(tmp_path: Path) -> None:
    c = tmp_path / "c"
    chain(
        c,
        main=("pass", "pass"),
        fallback=("pass", "pass"),
        persistence=(None, "pass"),
    )
    seen: list[str] = []

    def validate(sub: Path, log: Path) -> str:
        seen.append(sub.parts[-5])
        log.write_text("x")
        return "fail" if len(seen) == 1 else "pass"

    prov = run_select(POLICY, c, tmp_path / "g", validate, retried=False)
    assert seen == ["main", "fallback"]
    assert prov["selected"] == "fallback"
    assert "hub validation failed" in prov["reason"]


def test_no_validator_marks_skip(tmp_path: Path) -> None:
    c = tmp_path / "c"
    chain(c, main=("pass", "pass"))
    prov = run_select(POLICY, c, tmp_path / "g", None, retried=False)
    assert prov["attempts"][0]["validation"] == "skip"


def test_cli_exit_code_on_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    c = tmp_path / "c"
    chain(c, main=("fail", "fail"), persistence=(None, "fail"))
    policy = str(Path(__file__).parents[1] / "policy.toml")
    rc = main(
        [
            "select",
            "--policy",
            policy,
            "--candidates",
            str(c),
            "--out",
            str(tmp_path / "g"),
            "--validate-cmd",
            "true",
        ]
    )
    assert rc == 3
    assert "skipped=true" in out.read_text()
    assert read_json(tmp_path / "g" / "provenance.json")["selected"] is None
    rc = main(["triage", "--policy", policy, "--candidates", str(c)])
    assert rc == 0 and "need_retry=true" in out.read_text()


STAND_IN = (
    "the `main` config runs the persistence baseline until the PyRenew "
    "renewal model is integrated"
)


def test_stand_in_is_named(tmp_path: Path) -> None:
    root = tmp_path / "candidates"
    chain(root, main=("skip", "pass"))
    write_json(
        root / "main" / "run.json", {"model": "renewal", "stand_in": STAND_IN}
    )
    prov = run_select(POLICY, root, tmp_path / "gate", None, retried=False)
    assert prov["stand_in"] == STAND_IN
    assert prov["produced_by"] == "main model (stand-in)"
    assert prov["reason"] == "The main model (stand-in) cleared every check."
    md = (tmp_path / "gate" / "provenance.md").read_text()
    assert f"**Stand-in:** {STAND_IN}." in md
    assert "| `main` | main model (stand-in) |" in md


def test_stand_in_from_diagnostics_in_chain_table(tmp_path: Path) -> None:
    root = tmp_path / "candidates"
    chain(root, main=("fail", "pass"), retry_1=("pass", "pass"))
    write_json(
        root / "main" / "diagnostics.json",
        {"sampler": "none", "stand_in": STAND_IN},
    )
    prov = run_select(POLICY, root, tmp_path / "gate", None, retried=True)
    assert prov["selected"] == "retry-1"
    assert prov["stand_in"] == ""
    assert prov["reason"].startswith("Main model (stand-in) was ruled out")
