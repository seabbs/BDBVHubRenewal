from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
from conftest import REF, make_samples, write_json, write_samples
from test_check_coherence import GAP, coherence
from test_check_ppc import ppc
from test_select_candidate import STAND_IN

from submission_ci.build_pr_body import (
    FOOTER,
    MAX_BODY,
    build,
    main,
    marker,
    round_url,
)
from submission_ci.check_coherence import assess
from submission_ci.check_ppc import summarise
from submission_ci.common import read_json


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
        "model_id": "seabbs-renewal",
        "hub_ref": "a" * 40,
        "data_commit": "b" * 40,
        "as_of": "2026-10-31",
        "plot_url": "https://raw.githubusercontent.com/o/r/sha/p.png",
        "validation": "pass",
        "source_sha": "c" * 40,
    }
    args.update(kw)
    return build(**args)


def visible(text: str) -> list[str]:
    """Non-blank lines outside the collapsed sections, bar the marker."""
    lines, depth = [], 0
    for line in text.splitlines():
        depth += line.count("<details>") - line.count("</details>")
        if line.count("<details>") or line.count("</details>"):
            continue
        if depth == 0 and line.strip() and not line.startswith("<!--"):
            lines.append(line)
    return lines


def test_body_has_required_parts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITHUB_REPOSITORY", "seabbs/x")
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    text = body(gate(tmp_path))
    assert text.startswith(marker("seabbs-renewal", REF))
    shown = visible(text)
    assert len(shown) <= 6, shown
    assert shown[0] == (
        f"Automated submission of `seabbs-renewal` for {REF}, from "
        "data as of 2026-10-31, for 4 locations."
    )
    assert shown[1] == "All checks passed."
    assert shown[2].startswith("![Nowcast and forecast") and "p.png" in text
    assert shown[3] == f"This round in our docs: {round_url(REF)}"
    assert shown[-1] == FOOTER
    assert "| Convergence | PASS (max R-hat 1.004" in text
    assert "| Hub validation | PASS |" in text
    assert "renewal (`main`), version 0.2.0" in text
    assert "bbbbbbbbbb" in text and "actions/runs/42" in text
    assert "<summary>National summary</summary>" in text
    assert "<summary>Provenance</summary>" in text
    # One sampled fit converged: no chain section, but convergence.
    assert "<summary>Fallback chain</summary>" not in text
    assert "<summary>Convergence</summary>" in text


def test_warnings_are_stated(tmp_path: Path) -> None:
    text = body(
        gate(
            tmp_path, status="warn", warnings=["growth: CD-IT grows 2.5 times"]
        )
    )
    assert visible(text)[1] == "All checks passed, with warnings on growth."
    assert "(growth flagged)" in text
    assert "- WARN growth: x" in text


def test_fallback_is_named_with_why(tmp_path: Path) -> None:
    g = gate(tmp_path, produced_by="persistence baseline")
    prov = read_json(g / "provenance.json")
    prov.update(
        kind="persistence",
        selected="persistence",
        attempts=[
            {
                "id": "main",
                "label": "main model",
                "status": "fail",
                "reasons": ["convergence: max R-hat is 1.4"],
            },
            {
                "id": "persistence",
                "label": "persistence baseline",
                "status": "pass",
                "validation": "pass",
                "reasons": [],
            },
        ],
    )
    write_json(g / "provenance.json", prov)
    text = body(g)
    assert visible(text)[1] == (
        "Fallback model used (persistence baseline): main model was ruled "
        "out (convergence: max R-hat is 1.4)."
    )
    assert "<summary>Fallback chain</summary>" in text
    assert "| persistence baseline | submitted |  |" in text


def test_retry_is_named(tmp_path: Path) -> None:
    g = gate(tmp_path, produced_by="main model, retry 1")
    prov = read_json(g / "provenance.json")
    prov.update(kind="retry", selected="retry-1", attempts=[
        {"id": "main", "label": "main model", "status": "fail",
         "reasons": ["convergence: max R-hat is 1.4"]},
    ])  # fmt: skip
    write_json(g / "provenance.json", prov)
    assert visible(body(g))[1].startswith(
        "Main model refitted once: main model was ruled out"
    )


def test_handled_line_leads(tmp_path: Path) -> None:
    g = gate(tmp_path)
    prov = read_json(g / "provenance.json")
    prov["handled"] = [
        "main model refitted once (max R-hat is 1.2, past the retry "
        "threshold 1.1)",
        "delays: used the 2026-10-17 vintage, the latest available",
    ]
    write_json(g / "provenance.json", prov)
    shown = visible(body(g))
    assert shown[1] == "All checks passed."
    assert shown[2] == (
        "Handled: main model refitted once (max R-hat is 1.2, past the "
        "retry threshold 1.1); delays: used the 2026-10-17 vintage, the "
        "latest available."
    )
    assert shown[3].startswith("![Nowcast")
    assert len(shown) == 6
    assert "Handled:" not in body(gate(tmp_path / "plain"))


def test_dry_run_note(tmp_path: Path) -> None:
    line = visible(body(gate(tmp_path), dry_run=True))[1]
    assert line == "All checks passed. Dry run: nothing was pushed."


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
                "--docs",
                "https://example.org/docs/",
                "--out",
                str(tmp_path / "o"),
            ]
        )
        == 0
    )
    assert (
        tmp_path / "o" / "pr-title.txt"
    ).read_text() == f"m submission for {REF}"
    out = (tmp_path / "o" / "pr-body.md").read_text()
    assert "| Hub validation | PASS |" in out
    assert f"https://example.org/docs/submission/rounds/{REF}/" in out


def test_data_coherence_reported_once(tmp_path: Path) -> None:
    g = gate(tmp_path)
    write_json(
        g / "selected" / "coherence-summary.json",
        asdict(assess(coherence([GAP]))),
    )
    text = body(g)
    shown = visible(text)
    assert shown[1] == "All checks passed."
    assert shown[2] == (
        "Handled: national (CD) differs from the sum of the provinces in 1 "
        "week (total stream, largest gap 2); modelled as an unassigned "
        "national component."
    )
    assert len(shown) <= 6
    assert text.count("| Data coherence | Handled (national (CD) differs") == 1
    assert text.count("<summary>Data coherence</summary>") == 1
    assert "| 2026-10-24 | total | 50 | 48 | 2 |" in text
    assert "unassigned national component" in text


def test_data_coherence_na_without_file(tmp_path: Path) -> None:
    text = body(gate(tmp_path))
    assert "| Data coherence | N/A (not checked) |" in text
    assert "<summary>Data coherence</summary>" not in text


def test_ppc_row_and_province_details(tmp_path: Path) -> None:
    g = gate(tmp_path)
    sel = g / "selected"
    write_json(tmp_path / "ppc.json", ppc("ppcwarn"))
    summarise("main", tmp_path / "ppc.json", sel, sampled=True)
    text = body(g)
    assert (
        "| Posterior predictive checks | WARN (90% coverage 0.57 overall, "
        "0.67 last 4 weeks; 50% coverage" in text
    )
    assert "<summary>Posterior predictive checks</summary>" in text
    assert "| all provinces | " in text
    assert "| CD-IT |" in text
    assert "\n\nPIT: KS distance" in text


def test_ppc_row_without_posterior(tmp_path: Path) -> None:
    g = gate(tmp_path)
    (g / "selected" / "fit-summary.json").unlink()
    assert "| Posterior predictive checks | N/A (no posterior) |" in body(g)
    write_json(
        g / "selected" / "ppc-summary.json",
        {"status": "skip", "reason": "the model has no sampler"},
    )
    assert (
        "| Posterior predictive checks | N/A (the model has no sampler) |"
        in body(g)
    )


def test_stand_in_is_stated(tmp_path: Path) -> None:
    g = gate(tmp_path, produced_by="main model (stand-in)")
    prov = read_json(g / "provenance.json")
    prov["run"]["model"] = "persistence"
    write_json(g / "provenance.json", {**prov, "stand_in": STAND_IN})
    assert visible(body(g))[1] == (
        "All checks passed. Stand-in: persistence runs as the `main` model."
    )
    assert "Stand-in" not in body(gate(tmp_path / "real"))


def test_body_reports_shadow_candidates(tmp_path: Path) -> None:
    from submission_ci.common import read_json

    g = gate(tmp_path)
    prov = read_json(g / "provenance.json")
    prov["shadows"] = [
        {
            "id": "bvd",
            "label": "BVDOutbreakSize-informed variant",
            "fit": {"status": "pass", "headline": {"max_rhat": 1.01}},
            "ppc": None,
        }
    ]
    text = body(g, provenance=prov)
    assert "Shadow candidates (fitted and checked, not submitted)" in text
    assert "BVDOutbreakSize-informed variant" in text
    # Collapsed, so not on the visible lines.
    assert not any("BVDOutbreakSize" in line for line in visible(text))


def test_body_reports_the_bvd_release(tmp_path: Path) -> None:
    from submission_ci.common import read_json

    g = gate(tmp_path)
    prov = read_json(g / "provenance.json")
    release = {
        "tag": "results-3180",
        "as_of": "2026-10-23",
        "joint_max_rhat": 1.041,
        "max_rhat": 1.1,
        "skipped": [
            {"tag": "results-3190", "reason": "joint R-hat 1.230 is above 1.1"}
        ],
    }
    prov["shadows"] = [
        {
            "id": "bvd",
            "label": "BVDOutbreakSize-informed variant",
            "fit": {
                "status": "pass",
                "headline": {"max_rhat": 1.01},
                "bvd": release,
            },
            "ppc": None,
        }
    ]
    text = body(g, provenance=prov)
    assert (
        "BVDOutbreakSize-informed variant used BVDOutbreakSize release "
        "`results-3180` (data to 2026-10-23, joint R-hat 1.041, limit 1.1)."
    ) in text
    assert "Skipped `results-3190`: joint R-hat 1.230 is above 1.1." in text
    assert not any("results-3180" in line for line in visible(text))
    prov["shadows"][0]["fit"]["bvd"] = {"tag": None}
    text = body(g, provenance=prov)
    assert (
        "BVDOutbreakSize-informed variant used no BVDOutbreakSize release: "
        "none qualified, so it ran as the main model."
    ) in text


def test_convergence_section_reports_the_bvd_release(tmp_path: Path) -> None:
    from submission_ci.common import read_json

    g = gate(tmp_path)
    fit = read_json(g / "selected" / "fit-summary.json")
    fit["bvd"] = {
        "tag": "results-3180",
        "as_of": "2026-10-23",
        "joint_max_rhat": 1.041,
        "max_rhat": 1.1,
        "skipped": [],
    }
    write_json(g / "selected" / "fit-summary.json", fit)
    text = body(g)
    assert (
        "The model used BVDOutbreakSize release `results-3180` (data to "
        "2026-10-23, joint R-hat 1.041, limit 1.1)."
    ) in text
    assert "Skipped" not in text
