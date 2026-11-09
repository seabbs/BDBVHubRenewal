from __future__ import annotations

from pathlib import Path

import pytest
from conftest import POLICY, POLICY_PATH, diagnostics, write_json

from submission_ci.common import read_json
from submission_ci.summarise_fit import check_fit, headline, main, markdown

LIMITS = POLICY.convergence


def test_policy_matches_bvdoutbreaksize() -> None:
    # docs/fits/convergence.jl in BVDOutbreakSize: past its failure
    # thresholds a fit is ruled out.
    assert set(LIMITS) == {"fail", "warn"}
    assert LIMITS["fail"] == {
        "rhat": 1.1,
        "ess_bulk": 25,
        "ess_tail": 25,
        "divergent_fraction": 0.05,
    }
    assert LIMITS["warn"] == {
        "rhat": 1.05,
        "ess_bulk": 100,
        "ess_tail": 100,
        "divergent_fraction": 0.01,
    }


def test_headline_skips_undefined_entries() -> None:
    h = headline(diagnostics())
    assert h["max_rhat"] == pytest.approx(1.004)
    assert h["min_ess_bulk"] == 900
    assert h["n_draws"] == 4000 and h["n_divergent"] == 0


@pytest.mark.parametrize(
    ("kwargs", "status"),
    [
        ({}, "pass"),
        ({"rhat": 1.07}, "warn"),
        ({"ess": 60}, "warn"),
        ({"div": 80}, "warn"),  # 2% of draws
        ({"rhat": 1.3}, "fail"),
        ({"ess": 20}, "fail"),
        ({"div": 400}, "fail"),  # 10% of draws
    ],
)
def test_verdict_levels(kwargs: dict[str, float], status: str) -> None:
    c = check_fit("main", diagnostics(**kwargs), LIMITS)  # type: ignore[arg-type]
    assert c.status == status


def test_missing_diagnostics_is_an_error() -> None:
    c = check_fit("main", None, LIMITS)
    assert c.status == "error" and c.failures


def test_no_sampler_is_not_applicable() -> None:
    c = check_fit("randomwalk", diagnostics(sampler="none"), LIMITS)
    assert c.status == "skip" and not c.applicable


def test_failure_lists_culprits() -> None:
    c = check_fit("main", diagnostics(rhat=1.8), LIMITS)
    assert c.worst and c.families
    md = markdown([c], LIMITS)
    assert "Worst-mixing parameters" in md and "**fail**" in md
    assert "1.800" in md  # R-hat to three decimals
    assert "Ruled out (fall back)" in md


def test_policy_thresholds_apply() -> None:
    lim = {**LIMITS, "fail": {**LIMITS["fail"], "rhat": 1.02}}
    assert check_fit("m", diagnostics(rhat=1.03), lim).status == "fail"


def test_cli_exit_codes_and_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    good = write_json(tmp_path / "good.json", diagnostics())
    bad = write_json(tmp_path / "bad.json", diagnostics(rhat=2.0))
    assert (
        main(
            [
                "--id",
                "m",
                "--policy",
                str(POLICY_PATH),
                "--diagnostics",
                str(good),
                "--out",
                str(tmp_path / "a"),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "--id",
                "m",
                "--policy",
                str(POLICY_PATH),
                "--diagnostics",
                str(bad),
                "--out",
                str(tmp_path / "b"),
            ]
        )
        == 1
    )
    assert (
        main(
            [
                "--id",
                "m",
                "--policy",
                str(POLICY_PATH),
                "--diagnostics",
                str(tmp_path / "nope.json"),
                "--out",
                str(tmp_path / "c"),
            ]
        )
        == 2
    )
    assert (
        main(
            [
                "--id",
                "m",
                "--policy",
                str(POLICY_PATH),
                "--diagnostics",
                str(bad),
                "--out",
                str(tmp_path / "d"),
                "--no-fail",
            ]
        )
        == 0
    )
    data = read_json(tmp_path / "b" / "fit-summary.json")
    assert (
        data["status"] == "fail" and data["thresholds"]["fail"]["rhat"] == 1.1
    )
    assert "Fit convergence" in summary.read_text()


def test_bvd_release_is_carried_into_the_summary(tmp_path: Path) -> None:
    release = {
        "tag": "results-3180",
        "as_of": "2026-10-23",
        "joint_max_rhat": 1.041,
        "max_rhat": 1.1,
        "skipped": [
            {"tag": "results-3190", "reason": "no readable joint R-hat"}
        ],
    }
    out = check_fit("bvd", diagnostics() | {"bvd": release}, LIMITS)
    assert out.bvd == release
    assert check_fit("main", diagnostics(), LIMITS).bvd is None
    path = tmp_path / "diagnostics.json"
    write_json(path, diagnostics() | {"bvd": release})
    assert (
        main(
            [
                "--id",
                "bvd",
                "--diagnostics",
                str(path),
                "--out",
                str(tmp_path),
                "--policy",
                str(POLICY_PATH),
            ]
        )
        == 0
    )
    assert read_json(tmp_path / "fit-summary.json")["bvd"] == release
