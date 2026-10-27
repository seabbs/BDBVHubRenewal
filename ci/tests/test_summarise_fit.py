from __future__ import annotations

from pathlib import Path

import pytest
from conftest import diagnostics, write_json

from submission_ci.common import read_json
from submission_ci.summarise_fit import (
    DEFAULT_THRESHOLDS,
    check_fit,
    headline,
    main,
    markdown,
    thresholds,
)

LIMITS = thresholds()


def test_defaults_match_bvdoutbreaksize() -> None:
    # docs/fits/convergence.jl in BVDOutbreakSize
    assert DEFAULT_THRESHOLDS["fail"] == {
        "rhat": 1.1,
        "ess_bulk": 25.0,
        "ess_tail": 25.0,
        "divergent_fraction": 0.05,
    }
    assert DEFAULT_THRESHOLDS["warn"] == {
        "rhat": 1.05,
        "ess_bulk": 100.0,
        "ess_tail": 100.0,
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
        ({"ess": 10}, "fail"),
        ({"div": 400}, "fail"),  # 10% of draws
    ],
)
def test_verdict_levels(kwargs: dict[str, float], status: str) -> None:
    assert check_fit("main", diagnostics(**kwargs), LIMITS).status == status  # type: ignore[arg-type]


def test_missing_diagnostics_is_an_error() -> None:
    c = check_fit("main", None, LIMITS)
    assert c.status == "error" and c.failures


def test_no_sampler_is_not_applicable() -> None:
    c = check_fit("persistence", diagnostics(sampler="none"), LIMITS)
    assert c.status == "skip" and not c.applicable


def test_failure_lists_culprits() -> None:
    c = check_fit("main", diagnostics(rhat=1.3), LIMITS)
    assert c.worst and c.families
    md = markdown([c], LIMITS)
    assert "Worst-mixing parameters" in md and "**fail**" in md
    assert "1.300" in md  # R-hat to three decimals


def test_env_overrides_and_rejects_nonfinite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BDBV_CONVERGENCE_FAIL_RHAT", "1.05")
    assert thresholds()["fail"]["rhat"] == 1.05
    monkeypatch.setenv("BDBV_CONVERGENCE_FAIL_RHAT", "inf")
    with pytest.raises(ValueError, match="finite"):
        thresholds()
    monkeypatch.setenv("BDBV_CONVERGENCE_FAIL_RHAT", "abc")
    with pytest.raises(ValueError, match="not a number"):
        thresholds()


def test_policy_thresholds_apply() -> None:
    lim = thresholds({"fail": {"rhat": 1.02}})
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
