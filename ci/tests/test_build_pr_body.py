from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from conftest import REF, make_samples, write_json, write_samples

from submission_ci.build_pr_body import FOOTER, MAX_BODY, build, main, marker


def gate(
    tmp_path: Path,
    status: str = "pass",
    warnings: list[str] | None = None,
    produced_by: str = "main model",
) -> Path:
    g = tmp_path / "gate"
    sel = g / "selected"
    write_samples(
        sel / "submission" / "model-output" / "m" / f"{REF}-m.csv",
        make_samples(),
    )
    write_json(
        sel / "fit-summary.json",
        {
            "status": "pass",
            "headline": {
                "max_rhat": 1.004,
                "min_ess_bulk": 812.0,
                "n_divergent": 0.0,
                "n_draws": 4000.0,
            },
        },
    )
    write_json(
        sel / "checks.json",
        {
            "status": status,
            "checks": [{"name": "growth", "status": status, "summary": "x"}],
        },
    )
    (sel / "fit-summary.md").write_text(
        "<!-- bdbv-fit-convergence -->\n## Fit convergence\n"
    )
    (sel / "checks.md").write_text("## Forecast checks\n")
    (g / "provenance.md").write_text("## Produced by: main model\n")
    write_json(
        g / "provenance.json",
        {
            "selected": "main",
            "kind": "main",
            "produced_by": produced_by,
            "status": status,
            "reason": "The main model cleared every check.",
            "warnings": warnings or [],
            "attempts": [{"id": "main", "validation": "pass"}],
            "run": {
                "model": "renewal",
                "config": "main",
                "package_version": "0.2.0",
            },
        },
    )
    return g


def body(g: Path, **kw: Any) -> str:
    from submission_ci.common import read_json

    args: dict[str, Any] = {
        "provenance": read_json(g / "provenance.json"),
        "selected_dir": g / "selected",
        "reference_date": REF,
        "model_id": "epiforecasts-renewal",
        "hub_ref": "a" * 40,
        "data_commit": "b" * 40,
        "as_of": "2026-10-31",
        "plot_url": "https://raw.githubusercontent.com/o/r/sha/p.png",
        "validation": "pass",
        "source_sha": "c" * 40,
    }
    args.update(kw)
    return build(**args)


def test_body_has_required_parts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITHUB_REPOSITORY", "seabbs/x")
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    text = body(gate(tmp_path))
    assert text.startswith(marker("epiforecasts-renewal", REF))
    assert "Automated submission" in text and "No person reviewed it" in text
    assert "**Produced by: main model.**" in text
    assert "![Nowcast and forecast" in text and "p.png" in text
    assert "| Convergence | PASS (max R-hat 1.004" in text
    assert "| Hub validation | PASS |" in text
    assert "renewal (`main`)" in text and "0.2.0" in text
    assert "2026-10-31" in text and "bbbbbbbbbb" in text
    assert "National (`CD`)" in text
    assert "actions/runs/42" in text
    assert text.rstrip().endswith(FOOTER) and "@seabbs" in text.splitlines()[-1]


def test_warnings_are_stated(tmp_path: Path) -> None:
    text = body(
        gate(
            tmp_path, status="warn", warnings=["growth: CD-IT grows 2.5 times"]
        )
    )
    assert "| Overall | WARN |" in text
    assert "[!WARNING]" in text and "CD-IT grows 2.5 times" in text
    assert "(growth flagged)" in text


def test_fallback_provenance_is_named(tmp_path: Path) -> None:
    text = body(gate(tmp_path, produced_by="persistence baseline"))
    assert "**Produced by: persistence baseline.**" in text


def test_dry_run_note(tmp_path: Path) -> None:
    assert "**dry run**" in body(gate(tmp_path), dry_run=True)


def test_long_body_is_truncated(tmp_path: Path) -> None:
    g = gate(tmp_path)
    (g / "selected" / "checks.md").write_text("x" * (MAX_BODY + 10))
    text = body(g)
    assert len(text) <= MAX_BODY and text.rstrip().endswith(FOOTER)


def test_cli(tmp_path: Path) -> None:
    g = gate(tmp_path)
    assert (
        main(
            [
                "--gate",
                str(g),
                "--reference-date",
                REF,
                "--model-id",
                "m",
                "--hub-ref",
                "a",
                "--data-commit",
                "b",
                "--plot-url",
                "p.png",
                "--out",
                str(tmp_path / "o"),
            ]
        )
        == 0
    )
    assert (
        tmp_path / "o" / "pr-title.txt"
    ).read_text() == f"m submission for {REF}"
    assert (
        "| Hub validation | PASS |"
        in (tmp_path / "o" / "pr-body.md").read_text()
    )
