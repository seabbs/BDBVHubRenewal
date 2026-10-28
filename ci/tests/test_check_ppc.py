"""The posterior predictive check gate on `ppc.json`."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest
from conftest import AS_OF, FIXTURES, diagnostics, write_json

from submission_ci.check_ppc import (
    check_ppc,
    is_sampled,
    limits,
    main,
    pit_summary,
)
from submission_ci.common import read_json
from submission_ci.policy import load_policy
from submission_ci.select_candidate import assess

_spec = importlib.util.spec_from_file_location(
    "fake_cli", FIXTURES / "fake_bdbvhubrenewal.py"
)
assert _spec is not None and _spec.loader is not None
fake = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fake)

POLICY = load_policy(Path(__file__).parents[1] / "policy.toml")
LIMITS = limits(POLICY.ppc)


def ppc(scenario: str = "pass") -> dict[str, Any]:
    payload: dict[str, Any] = fake.fake_ppc(AS_OF, scenario, "main")
    return payload


def run(tmp_path: Path, payload: Any, *extra: str) -> dict[str, Any]:
    args = ["--id", "main", "--out", str(tmp_path / "o")]
    path = tmp_path / "ppc.json"
    if payload is not None:
        if isinstance(payload, str):
            path.write_text(payload)
        else:
            write_json(path, payload)
    main([*args, "--ppc", str(path), "--no-fail", *extra])
    return dict(read_json(tmp_path / "o" / "ppc-summary.json"))


def test_policy_thresholds() -> None:
    assert LIMITS["coverage_warn"] == 0.70
    assert LIMITS["coverage_fail"] == 0.50
    assert LIMITS["recent_weeks"] == 4
    assert LIMITS["pit_ks_warn"] == 0.2


def test_calibrated_passes() -> None:
    c = check_ppc("main", ppc("pass"), LIMITS)
    assert c.status == "pass", c.warnings
    total = c.streams["total"]
    assert total["cov90"] >= 0.85
    assert total["recent_n"] == 12  # 4 complete weeks x 3 provinces
    assert c.recent_weeks == [
        "2026-09-12",
        "2026-09-19",
        "2026-09-26",
        "2026-10-03",
    ]
    assert {r["location"] for r in c.locations} == {"CD-HU", "CD-IT", "CD-NK"}


def test_low_coverage_warns() -> None:
    c = check_ppc("main", ppc("ppcwarn"), LIMITS)
    assert c.status == "warn"
    assert not c.failures
    assert any("overall" in w for w in c.warnings)
    assert "too narrow" in c.pit["shape"]


def test_very_low_coverage_only_warns() -> None:
    c = check_ppc("main", ppc("ppcfail"), LIMITS)
    assert c.status == "warn"
    assert not c.failures
    assert any("poor (below 0.5)" in m for m in c.warnings)


def test_env_overrides_thresholds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BDBV_PPC_COVERAGE_WARN", "0.5")
    monkeypatch.setenv("BDBV_PPC_COVERAGE_FAIL", "0.2")
    monkeypatch.setenv("BDBV_PPC_PIT_KS_WARN", "0.9")
    lim = limits(POLICY.ppc)
    assert check_ppc("main", ppc("ppcwarn"), lim).status == "pass"
    monkeypatch.setenv("BDBV_PPC_COVERAGE_FAIL", "inf")
    with pytest.raises(ValueError, match="finite"):
        limits(POLICY.ppc)


def test_unavailable_is_na(tmp_path: Path) -> None:
    na = {"schema_version": 1, "available": False, "reason": "no posterior"}
    data = run(tmp_path, na)
    assert data["status"] == "skip"
    assert data["reason"] == "no posterior"


def test_persistence_without_file_is_na(tmp_path: Path) -> None:
    assert run(tmp_path, None, "--kind", "persistence")["status"] == "skip"


def test_no_sampler_without_file_is_na(tmp_path: Path) -> None:
    diag = write_json(tmp_path / "d.json", diagnostics(sampler="none"))
    data = run(tmp_path, None, "--diagnostics", str(diag))
    assert data["status"] == "skip"


def test_sampled_fit_without_file_warns(tmp_path: Path) -> None:
    diag = write_json(tmp_path / "d.json", diagnostics())
    data = run(tmp_path, None, "--diagnostics", str(diag))
    assert data["status"] == "warn"
    assert "no ppc.json" in data["warnings"][0]
    assert is_sampled("main", None)  # a fit that wrote nothing


def test_unreadable_file_warns(tmp_path: Path) -> None:
    assert run(tmp_path, "{not json")["status"] == "warn"
    assert run(tmp_path, {"schema_version": 1})["status"] == "warn"


def test_cli_report_and_exit_codes(tmp_path: Path) -> None:
    path = write_json(tmp_path / "ppc.json", ppc("ppcfail"))
    args = ["--id", "main", "--ppc", str(path), "--out", str(tmp_path / "o")]
    assert main(args) == 0
    assert main([*args, "--no-fail"]) == 0
    md = (tmp_path / "o" / "ppc-summary.md").read_text()
    assert "## Posterior predictive checks for `main`: WARN" in md
    assert "Reported only, never gated" in md
    assert "<summary>Coverage of total by province</summary>" in md
    data = read_json(tmp_path / "o" / "ppc-summary.json")
    assert data["headline"].startswith("90% coverage")
    missing = ["--id", "m", "--ppc", str(tmp_path / "none.json")]
    assert main([*missing, "--out", str(tmp_path / "e")]) == 0


def test_pit_summary_shapes() -> None:
    assert pit_summary([])["n"] == 0
    wide = pit_summary([0.4 + 0.2 * i / 99 for i in range(100)])
    assert "too wide" in wide["shape"]
    assert wide["ks"] > 0.35
    high = pit_summary([0.8 + 0.2 * i / 99 for i in range(100)])
    assert "too low" in high["shape"]


@pytest.mark.parametrize(
    ("status", "blocks"),
    [("pass", False), ("warn", False), ("fail", True), ("error", True)],
)
def test_assess_folds_ppc(tmp_path: Path, status: str, blocks: bool) -> None:
    d = tmp_path / "main"
    write_json(d / "fit-summary.json", {"status": "pass"})
    write_json(d / "checks.json", {"status": "pass", "checks": []})
    write_json(d / "submission" / "x.csv", {})
    write_json(
        d / "ppc-summary.json",
        {
            "status": status,
            "failures": ["coverage 0.3"] if blocks else [],
            "warnings": ["coverage 0.6"] if status == "warn" else [],
        },
    )
    a = assess(POLICY.get("main"), d)
    assert a.ppc == status
    # Reported only: an old fail or error summary still only warns.
    assert a.status == ("warn" if blocks or status == "warn" else status)
    assert not a.reasons
    if status == "warn":
        assert a.warnings == ["ppc: coverage 0.6"]
    if blocks:
        assert a.warnings == ["ppc: coverage 0.3"]


def test_assess_without_ppc_summary(tmp_path: Path) -> None:
    d = tmp_path / "main"
    write_json(d / "fit-summary.json", {"status": "pass"})
    write_json(d / "checks.json", {"status": "pass", "checks": []})
    write_json(d / "submission" / "x.csv", {})
    a = assess(POLICY.get("main"), d)
    assert a.ppc == "warn"
    assert a.status == "warn"
    p = tmp_path / "persistence"
    write_json(p / "checks.json", {"status": "pass", "checks": []})
    write_json(p / "submission" / "x.csv", {})
    assert assess(POLICY.get("persistence"), p).ppc == "skip"
