"""Each path through the fallback chain."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import (
    POLICY,
    POLICY_PATH,
    make_samples,
    write_json,
    write_samples,
)

from submission_ci.common import read_json
from submission_ci.select_candidate import main, run_select

BVD = "BVDOutbreakSize-informed renewal model"


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


def test_first_candidate_passes(tmp_path: Path) -> None:
    c = tmp_path / "c"
    chain(
        c,
        bvd=("pass", "pass"),
        main=("pass", "pass"),
        randomwalk=(None, "pass"),
    )
    prov = run_select(POLICY, c, tmp_path / "g", always("pass"))
    assert prov["selected"] == "bvd" and prov["status"] == "pass"
    assert prov["reason"] == f"The {BVD} cleared every check."
    assert (tmp_path / "g" / "selected" / "submission").is_dir()
    states = {a["id"]: a["status"] for a in prov["attempts"]}
    assert states["main"] == "not needed"


def test_warnings_are_submitted_with_the_candidate(tmp_path: Path) -> None:
    c = tmp_path / "c"
    chain(c, bvd=("warn", "warn"), randomwalk=(None, "pass"))
    prov = run_select(POLICY, c, tmp_path / "g", always("pass"))
    assert prov["selected"] == "bvd" and prov["status"] == "warn"
    assert any("convergence" in w for w in prov["warnings"])
    assert any("growth" in w for w in prov["warnings"])


def test_a_failed_fit_moves_down_the_chain(tmp_path: Path) -> None:
    # No refit: past the convergence failure thresholds the chain moves
    # straight on.
    c = tmp_path / "c"
    chain(
        c,
        bvd=("fail", "pass"),
        main=("pass", "pass"),
        randomwalk=(None, "pass"),
    )
    prov = run_select(POLICY, c, tmp_path / "g", always("pass"))
    assert prov["selected"] == "main" and prov["kind"] == "fallback"
    assert (
        f"{BVD} was ruled out (convergence: max R-hat is 1.4" in prov["reason"]
    )


def test_predictive_and_plausibility_never_rule_out(tmp_path: Path) -> None:
    c = tmp_path / "c"
    candidate(c, "bvd", checks="warn", ppc="fail")
    candidate(c, "randomwalk", fit=None)
    prov = run_select(POLICY, c, tmp_path / "g", always("pass"))
    assert prov["selected"] == "bvd"
    assert any(w.startswith("ppc:") for w in prov["warnings"])


def test_handled_notes_are_collected(tmp_path: Path) -> None:
    c = tmp_path / "c"
    candidate(c, "bvd")
    write_json(
        c / "bvd" / "submission" / "handled.json",
        ["delays: used the 2026-10-17 vintage, the latest available"],
    )
    prov = run_select(POLICY, c, tmp_path / "g", always("pass"))
    assert prov["handled"] == [
        "delays: used the 2026-10-17 vintage, the latest available",
    ]


def test_a_crashed_fit_moves_down_the_chain(tmp_path: Path) -> None:
    c = tmp_path / "c"
    candidate(c, "bvd", fit="error", submission=False)
    chain(c, main=("pass", "pass"))
    prov = run_select(POLICY, c, tmp_path / "g", always("pass"))
    assert prov["selected"] == "main"


def test_each_renewal_model_in_turn(tmp_path: Path) -> None:
    c = tmp_path / "c"
    chain(
        c,
        bvd=("pass", "fail"),
        main=("pass", "pass"),
        randomwalk=(None, "pass"),
    )
    prov = run_select(POLICY, c, tmp_path / "g", always("pass"))
    assert prov["selected"] == "main" and prov["kind"] == "fallback"
    assert "growth" in prov["reason"]


def test_randomwalknowcast_before_randomwalk(tmp_path: Path) -> None:
    c = tmp_path / "c"
    chain(
        c,
        bvd=("fail", "pass"),
        main=("fail", "pass"),
        randomwalknowcast=(None, "pass"),
        randomwalk=(None, "pass"),
    )
    prov = run_select(POLICY, c, tmp_path / "g", always("pass"))
    assert prov["selected"] == "randomwalknowcast"
    assert prov["kind"] == "randomwalk"


def test_randomwalk_is_the_last_resort(tmp_path: Path) -> None:
    c = tmp_path / "c"
    chain(
        c,
        bvd=("fail", "pass"),
        main=("fail", "pass"),
        randomwalk=(None, "pass"),
    )
    prov = run_select(POLICY, c, tmp_path / "g", always("pass"))
    assert prov["selected"] == "randomwalk"
    assert prov["produced_by"] == "random walk baseline"


def test_everything_fails_round_skipped(tmp_path: Path) -> None:
    c = tmp_path / "c"
    chain(
        c,
        bvd=("fail", "pass"),
        main=("fail", "pass"),
        randomwalk=(None, "fail"),
    )
    prov = run_select(POLICY, c, tmp_path / "g", always("pass"))
    assert prov["selected"] is None and prov["status"] == "skip"
    assert prov["reason"].startswith("Every candidate was ruled out")
    md = (tmp_path / "g" / "provenance.md").read_text()
    assert md.startswith("## Round skipped")


def test_hub_validation_failure_moves_down_the_chain(tmp_path: Path) -> None:
    c = tmp_path / "c"
    chain(
        c,
        bvd=("pass", "pass"),
        main=("pass", "pass"),
        randomwalk=(None, "pass"),
    )
    seen: list[str] = []

    def validate(sub: Path, log: Path) -> str:
        seen.append(sub.parts[-5])
        log.write_text("x")
        return "fail" if len(seen) == 1 else "pass"

    prov = run_select(POLICY, c, tmp_path / "g", validate)
    assert seen == ["bvd", "main"]
    assert prov["selected"] == "main"
    assert "hub validation failed" in prov["reason"]


def test_no_validator_marks_skip(tmp_path: Path) -> None:
    c = tmp_path / "c"
    chain(c, bvd=("pass", "pass"))
    prov = run_select(POLICY, c, tmp_path / "g", None)
    assert prov["attempts"][0]["validation"] == "skip"


def test_cli_exit_code_on_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    c = tmp_path / "c"
    chain(c, bvd=("fail", "fail"), randomwalk=(None, "fail"))
    rc = main(
        [
            "--policy",
            str(POLICY_PATH),
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


def test_randomwalk_chosen_never_selects_a_shadow(tmp_path: Path) -> None:
    pol = POLICY.for_model("randomwalk")
    c = tmp_path / "c"
    chain(
        c,
        bvd=("pass", "pass"),
        main=("pass", "pass"),
        randomwalk=(None, "pass"),
    )
    prov = run_select(pol, c, tmp_path / "g", always("pass"))
    assert prov["selected"] == "randomwalk"


def test_provenance_lists_the_shadow_candidates(tmp_path: Path) -> None:
    pol = POLICY.for_model("randomwalk")
    c = tmp_path / "c"
    chain(c, bvd=("pass", "pass"), randomwalk=(None, "pass"))
    prov = run_select(pol, c, tmp_path / "g", always("pass"))
    shadows = {s["id"]: s for s in prov["shadows"]}
    assert set(shadows) == {"bvd", "main", "randomwalknowcast"}
    assert shadows["bvd"]["fit"] is not None
    assert shadows["main"]["fit"] is None


def test_baseline_track_never_falls_back_to_randomwalk(
    tmp_path: Path,
) -> None:
    # seabbs-rwnowcast submits random walk on a nowcast or nothing.
    pol = POLICY.for_model("randomwalknowcast", only=True)
    assert [c.id for c in pol.chain()] == ["randomwalknowcast"]
    c = tmp_path / "c"
    chain(c, randomwalknowcast=(None, "fail"), randomwalk=(None, "pass"))
    prov = run_select(
        pol, c, tmp_path / "g", always("pass"), shadows=False, only=True
    )
    assert prov["selected"] is None
    assert "no other model is submitted in its place" in prov["reason"]
    assert not (tmp_path / "g" / "selected").exists()
    assert "Round skipped" in (tmp_path / "g" / "provenance.md").read_text()


def test_baseline_track_skips_when_the_model_did_not_run(
    tmp_path: Path,
) -> None:
    pol = POLICY.for_model("randomwalknowcast", only=True)
    c = tmp_path / "c"
    chain(c, randomwalk=(None, "pass"))
    prov = run_select(
        pol, c, tmp_path / "g", always("pass"), shadows=False, only=True
    )
    assert prov["selected"] is None
    assert "could not run" in prov["reason"]


def test_a_chosen_model_starts_the_chain() -> None:
    # The chain runs on from the chosen model; those before it are
    # shadows.
    pol = POLICY.for_model("renewal")
    assert [c.id for c in pol.chain()] == [
        "main",
        "randomwalknowcast",
        "randomwalk",
    ]
    assert pol.get("bvd").kind == "shadow"
    assert {c.id for c in pol.stage("primary")} == {"bvd", "main"}


def test_a_chosen_variant_falls_back_to_the_random_walk(
    tmp_path: Path,
) -> None:
    pol = POLICY.for_model("renewal")
    c = tmp_path / "c"
    chain(
        c,
        bvd=("pass", "pass"),
        main=("fail", "pass"),
        randomwalknowcast=(None, "pass"),
        randomwalk=(None, "pass"),
    )
    prov = run_select(pol, c, tmp_path / "g", always("pass"), shadows=False)
    assert prov["selected"] == "randomwalknowcast"
