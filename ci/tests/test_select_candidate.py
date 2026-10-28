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
        loose = fit == "loose"
        fit = "warn" if loose else fit
        write_json(
            d / "fit-summary.json",
            {
                "status": fit,
                "retry": loose,
                "failures": ["max R-hat is 1.4, past the failure threshold 1.1"]
                if fit == "fail"
                else [],
                "warnings": [
                    "max R-hat is 1.2, past the retry threshold 1.1"
                    if loose
                    else "min bulk ESS is 80, past the warning threshold 100"
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
        "renewalsp",
        "persistencenow",
        "persistence",
    ]
    assert [c.id for c in POLICY.stage("primary")] == [
        "main",
        "renewalsp",
        "bvd",
    ]
    assert [c.id for c in POLICY.stage("retry")] == ["retry-1"]


def test_main_passes(tmp_path: Path) -> None:
    c = tmp_path / "c"
    chain(
        c,
        main=("pass", "pass"),
        renewalsp=("pass", "pass"),
        persistence=(None, "pass"),
    )
    assert not triage(POLICY, c)
    prov = run_select(POLICY, c, tmp_path / "g", always("pass"), retried=False)
    assert prov["selected"] == "main" and prov["status"] == "pass"
    assert prov["reason"] == "The main model cleared every check."
    assert (tmp_path / "g" / "selected" / "submission").is_dir()
    states = {a["id"]: a["status"] for a in prov["attempts"]}
    assert (
        states["renewalsp"] == "not needed"
        and states["retry-1"] == "not needed"
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
        renewalsp=("pass", "pass"),
        persistence=(None, "pass"),
    )
    assert triage(POLICY, c)
    t = POLICY.timing
    assert triage(POLICY, c, minutes_left=t.retry_minutes + t.reserve_minutes)
    # Too close to the window's end for a refit: go down the chain.
    assert not triage(POLICY, c, minutes_left=t.retry_minutes)
    prov = run_select(POLICY, c, tmp_path / "g", always("pass"), retried=True)
    assert prov["selected"] == "retry-1"
    assert prov["produced_by"].startswith("main model, refitted")
    assert (
        "Main model was ruled out (convergence: max R-hat is 1.4"
        in prov["reason"]
    )


def test_loose_main_is_refitted_once(tmp_path: Path) -> None:
    c = tmp_path / "c"
    chain(
        c,
        main=("loose", "pass"),
        retry_1=("pass", "pass"),
        renewalsp=("pass", "pass"),
        persistence=(None, "pass"),
    )
    assert triage(POLICY, c)
    prov = run_select(POLICY, c, tmp_path / "g", always("pass"), retried=True)
    assert prov["selected"] == "retry-1"
    assert prov["handled"][0] == (
        "main model refitted once (max R-hat is 1.2, past the retry "
        "threshold 1.1)"
    )


def test_loose_main_is_kept_when_the_refit_is_broken(tmp_path: Path) -> None:
    c = tmp_path / "c"
    chain(
        c,
        main=("loose", "pass"),
        retry_1=("fail", "pass"),
        renewalsp=("pass", "pass"),
        persistence=(None, "pass"),
    )
    prov = run_select(POLICY, c, tmp_path / "g", always("pass"), retried=True)
    assert prov["selected"] == "main"
    assert prov["handled"][0].startswith(
        "main model kept after one refit did not help"
    )


def test_loose_main_without_a_refit_is_accepted(tmp_path: Path) -> None:
    c = tmp_path / "c"
    chain(c, main=("loose", "pass"), persistence=(None, "pass"))
    prov = run_select(POLICY, c, tmp_path / "g", always("pass"), retried=False)
    assert prov["selected"] == "main"
    assert prov["handled"][0].startswith("accepted without a refit")


def test_predictive_and_plausibility_never_rule_out(tmp_path: Path) -> None:
    c = tmp_path / "c"
    candidate(c, "main", checks="warn", ppc="fail")
    candidate(c, "persistence", fit=None)
    prov = run_select(POLICY, c, tmp_path / "g", always("pass"), retried=False)
    assert prov["selected"] == "main"
    assert any(w.startswith("ppc:") for w in prov["warnings"])


def test_handled_notes_are_collected(tmp_path: Path) -> None:
    c = tmp_path / "c"
    candidate(c, "main")
    write_json(
        c / "main" / "submission" / "handled.json",
        ["delays: used the 2026-10-17 vintage, the latest available"],
    )
    checks = read_json(c / "main" / "checks.json")
    write_json(
        c / "main" / "checks.json",
        {**checks, "handled": ["national (CD) rebuilt as the sum"]},
    )
    prov = run_select(POLICY, c, tmp_path / "g", always("pass"), retried=False)
    assert prov["handled"] == [
        "delays: used the 2026-10-17 vintage, the latest available",
        "national (CD) rebuilt as the sum",
    ]


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
        renewalsp=("pass", "pass"),
        persistence=(None, "pass"),
    )
    prov = run_select(POLICY, c, tmp_path / "g", always("pass"), retried=True)
    assert prov["selected"] == "renewalsp" and prov["kind"] == "fallback"
    assert "growth" in prov["reason"]


def test_persistencenow_before_persistence(tmp_path: Path) -> None:
    c = tmp_path / "c"
    chain(
        c,
        main=("fail", "pass"),
        retry_1=("fail", "pass"),
        renewalsp=("pass", "fail"),
        persistencenow=(None, "pass"),
        persistence=(None, "pass"),
    )
    prov = run_select(POLICY, c, tmp_path / "g", always("pass"), retried=True)
    assert prov["selected"] == "persistencenow"
    assert prov["kind"] == "persistence"


def test_persistence_is_the_last_resort(tmp_path: Path) -> None:
    c = tmp_path / "c"
    chain(
        c,
        main=("fail", "pass"),
        retry_1=("fail", "pass"),
        renewalsp=("pass", "fail"),
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
        renewalsp=("pass", "fail"),
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
        renewalsp=("pass", "pass"),
        persistence=(None, "pass"),
    )
    seen: list[str] = []

    def validate(sub: Path, log: Path) -> str:
        seen.append(sub.parts[-5])
        log.write_text("x")
        return "fail" if len(seen) == 1 else "pass"

    prov = run_select(POLICY, c, tmp_path / "g", validate, retried=False)
    assert seen == ["main", "renewalsp"]
    assert prov["selected"] == "renewalsp"
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
    "the `main` config runs the persistence baseline in place of the model"
)


def test_stand_in_is_named(tmp_path: Path) -> None:
    root = tmp_path / "candidates"
    chain(root, main=("skip", "pass"))
    write_json(
        root / "main" / "run.json",
        {"model": "renewal", "stand_in": STAND_IN},
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


def test_persistence_chosen_never_selects_a_shadow(tmp_path: Path) -> None:
    pol = POLICY.for_model("persistence")
    c = tmp_path / "c"
    chain(
        c,
        main=("pass", "pass"),
        renewalsp=("pass", "pass"),
        persistence=(None, "pass"),
    )
    assert not triage(pol, c)
    prov = run_select(pol, c, tmp_path / "g", always("pass"), retried=False)
    assert prov["selected"] == "persistence"


def test_provenance_lists_the_shadow_candidates(tmp_path: Path) -> None:
    pol = POLICY.for_model("persistence")
    c = tmp_path / "c"
    chain(c, main=("pass", "pass"), persistence=(None, "pass"))
    prov = run_select(pol, c, tmp_path / "g", always("pass"), retried=False)
    shadows = {s["id"]: s for s in prov["shadows"]}
    assert set(shadows) == {"main", "renewalsp", "bvd", "persistencenow"}
    assert shadows["main"]["fit"] is not None
    assert shadows["bvd"]["fit"] is None


def test_a_track_submits_one_candidate_alone(tmp_path: Path) -> None:
    # A second hub model (its own id) is the one candidate, with no chain.
    pol = POLICY.only("persistencenow")
    assert [c.id for c in pol.chain()] == ["persistencenow"]
    c = tmp_path / "c"
    chain(c, persistencenow=(None, "pass"), persistence=(None, "pass"))
    prov = run_select(pol, c, tmp_path / "g", always("pass"), retried=False)
    assert prov["selected"] == "persistencenow"
    bad = tmp_path / "bad"
    chain(bad, persistencenow=(None, "fail"), persistence=(None, "pass"))
    prov = run_select(pol, bad, tmp_path / "g2", always("pass"), retried=False)
    assert prov["selected"] is None
