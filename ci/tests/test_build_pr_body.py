from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from conftest import (
    HUB,
    POLICY,
    POLICY_PATH,
    REF,
    make_samples,
    write_json,
    write_samples,
)
from test_check_coherence import GAP, coherence
from test_check_ppc import ppc

from submission_ci.build_pr_body import (
    FOOTER,
    MAX_BODY,
    SIGN_OFF,
    build,
    main,
    marker,
    round_url,
    summary_table,
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
    args: dict[str, Any] = {
        "provenance": read_json(g / "provenance.json"),
        "selected_dir": g / "selected",
        "reference_date": REF,
        "model_id": "seabbs-renewal",
        "as_of": "2026-10-31",
        "plot_url": "https://raw.githubusercontent.com/o/r/sha/p.png",
        "validation": "pass",
    }
    args.update(kw)
    return build(**args)


def visible(text: str) -> list[str]:
    """Non-blank lines outside the collapsed block, bar the marker."""
    lines, depth = [], 0
    for line in text.splitlines():
        depth += line.count("<details>") - line.count("</details>")
        if line.count("<details>") or line.count("</details>"):
            continue
        if depth == 0 and line.strip() and not line.startswith("<!--"):
            lines.append(line)
    return lines


def test_body_is_short_and_signed_off(tmp_path: Path) -> None:
    text = body(gate(tmp_path))
    assert text.startswith(marker("seabbs-renewal", REF))
    shown = visible(text)
    assert shown[0] == (
        f"Submission of `seabbs-renewal` for {REF}, from data as of "
        "2026-10-31, for 4 locations."
    )
    assert shown[1] == SIGN_OFF == "Signed off by Sam Abbott (@seabbs)."
    assert shown[2] == "Model: main model (`renewal`)."
    assert shown[3].startswith("![Nowcast and forecast") and "p.png" in text
    assert "| week ending | median | 90% interval |" in text
    for row in (
        "| Convergence | PASS |",
        "| Forecast checks | PASS |",
        "| Hub validation | PASS |",
        "| Data coherence | N/A |",
    ):
        assert row in shown
    assert f"This round in our docs: {round_url(REF)}" in shown
    assert shown[-1] == FOOTER
    assert "Dry run" not in text
    assert text.count("<details>") == 1
    assert "<summary>Details</summary>" in text
    assert "**Convergence**" in text and "**Forecast checks**" in text


def test_dry_run_note_and_no_relative_plot(tmp_path: Path) -> None:
    text = body(gate(tmp_path), dry_run=True, plot_url="nowcast.png")
    assert "Dry run: nothing was pushed." in visible(text)
    assert "nowcast.png" not in text


def test_cascade_model_named_with_why(tmp_path: Path) -> None:
    g = gate(tmp_path, produced_by="single-patch province renewal models")
    prov = read_json(g / "provenance.json")
    prov.update(
        kind="fallback",
        selected="renewalsp",
        attempts=[
            {
                "id": "bvd",
                "label": "BVDOutbreakSize-informed renewal model",
                "status": "fail",
                "reasons": ["convergence: max R-hat is 1.4"],
            },
            {
                "id": "renewalsp",
                "label": "single-patch province renewal models",
                "status": "pass",
                "reasons": [],
            },
        ],
    )
    prov["run"]["model"] = "renewalsp"
    text = body(g, provenance=prov)
    assert visible(text)[2] == (
        "Model: single-patch province renewal models (`renewalsp`), "
        "because BVDOutbreakSize-informed renewal model was ruled out "
        "(convergence: max R-hat is 1.4)."
    )


def test_first_model_in_the_chain(tmp_path: Path) -> None:
    g = gate(tmp_path, produced_by="BVDOutbreakSize-informed renewal model")
    prov = read_json(g / "provenance.json")
    prov.update(
        selected="bvd",
        attempts=[
            {"id": "bvd", "label": "x", "status": "pass", "reasons": []},
            {"id": "renewalsp", "label": "y", "status": "not needed"},
        ],
    )
    prov["run"]["model"] = "renewalbvd"
    assert visible(body(g, provenance=prov))[2] == (
        "Model: BVDOutbreakSize-informed renewal model (`renewalbvd`), the "
        "first model in our chain and the first to clear its checks."
    )


def test_warnings_show_as_status_only(tmp_path: Path) -> None:
    text = body(gate(tmp_path, status="warn"))
    assert "| Forecast checks | WARN |" in visible(text)
    assert "WARN growth: x" in text


def test_long_body_is_truncated(tmp_path: Path) -> None:
    g = gate(tmp_path)
    checks = read_json(g / "selected" / "checks.json")
    checks["checks"][0].update(status="warn", summary="x" * (MAX_BODY + 10))
    write_json(g / "selected" / "checks.json", checks)
    text = body(g)
    assert len(text) <= MAX_BODY and text.rstrip().endswith(FOOTER)


def test_cli(tmp_path: Path) -> None:
    g = gate(tmp_path)
    args = [
        "--gate", str(g), "--reference-date", REF, "--model-id", "m",
        "--plot-url", "p.png", "--docs", "https://example.org/docs/",
        "--observed", str(HUB / "time-series.csv"),
        "--policy", str(POLICY_PATH), "--as-of", "2026-10-31",
        "--out", str(tmp_path / "o"),
    ]  # fmt: skip
    assert main(args) == 0
    assert (
        tmp_path / "o" / "pr-title.txt"
    ).read_text() == f"m submission for {REF}"
    out = (tmp_path / "o" / "pr-body.md").read_text()
    assert "| Hub validation | PASS |" in out
    assert f"https://example.org/docs/submission/rounds/{REF}/" in out
    assert "forecast / reported" in out


def test_data_coherence_status(tmp_path: Path) -> None:
    g = gate(tmp_path)
    write_json(
        g / "selected" / "coherence-summary.json",
        asdict(assess(coherence([GAP]))),
    )
    text = body(g)
    assert any(line.startswith("| Data coherence | ") for line in visible(text))


def test_ppc_status_and_details(tmp_path: Path) -> None:
    g = gate(tmp_path)
    sel = g / "selected"
    write_json(tmp_path / "ppc.json", ppc("ppcwarn"))
    summarise("main", tmp_path / "ppc.json", sel, sampled=True, lim=POLICY.ppc)
    text = body(g)
    assert "| Posterior predictive checks | WARN |" in visible(text)
    assert "**Posterior predictive checks**" in text
    assert "| CD-IT |" in text


def test_convergence_details_report_the_bvd_release(tmp_path: Path) -> None:
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
    assert not any("results-3180" in line for line in visible(text))


def test_summary_table_compares_to_recent_reports(tmp_path: Path) -> None:
    g = gate(tmp_path)
    sub = next((g / "selected" / "submission").glob("**/*.csv"))
    obs = HUB / "time-series.csv"
    table = summary_table(sub, obs, "2026-10-31", REF, lag=4, n=2)
    lines = table.splitlines()
    assert "reported, mean of last 2 complete weeks" in lines[0]
    assert f"forecast, week ending {REF}" in lines[0]
    assert "forecast / reported" in lines[0]
    names = [line.split("|")[1].strip() for line in lines[2:]]
    assert names[0] == "DRC" and set(names[1:]) == {
        "Haut Uele",
        "Ituri",
        "Nord Kivu",
    }
    ituri = next(line for line in lines if "| Ituri |" in line)
    # Weeks ending on or before 2026-10-03 are complete at 2026-10-31.
    cells = [c.strip() for c in ituri.split("|")[1:-1]]
    recent = float(cells[1])
    assert recent > 0 and " to " in cells[2] and " to " in cells[3]
    median = float(cells[3].split(" ")[0])
    assert float(cells[4]) == round(median / recent, 2)


def test_summary_table_sits_above_the_fold(tmp_path: Path) -> None:
    text = body(
        gate(tmp_path),
        observed=HUB / "time-series.csv",
        as_of="2026-10-31",
    )
    shown = visible(text)
    assert any(line.startswith("|  | reported, mean of last") for line in shown)
    assert "**National (`CD`) by week**" in text
    assert not any("| week ending | median |" in line for line in shown)


def test_says_which_stored_fit_it_read(tmp_path: Path) -> None:
    g = gate(tmp_path)
    prov = read_json(g / "provenance.json")
    prov["run"].update(
        kind="main",
        store={
            "model": "renewal",
            "origin": "2026-10-31",
            "key": "0123456789abcdef",
            "commit": "c0ffee0" * 5 + "c",
            "fitted_at": "2026-11-01T08:00:00+00:00",
            "store_commit": "5707e00" * 5 + "5",
        },
    )
    assert visible(body(g, provenance=prov))[3] == (
        "Fit: read from the fit store, the `renewal` fit at 2026-10-31 "
        "(key `0123456789abcdef`, fitted at `c0ffee0`, fit-store commit "
        "`5707e00`)."
    )


def test_says_when_the_fit_was_made_in_the_run(tmp_path: Path) -> None:
    g = gate(tmp_path)
    prov = read_json(g / "provenance.json")
    prov["run"].update(kind="main", store=None)
    assert visible(body(g, provenance=prov))[3] == (
        "Fit: made in this run, as the fit store holds no fit for this round."
    )
    prov["run"].update(kind="randomwalk")
    assert not any(
        line.startswith("Fit:") for line in visible(body(g, provenance=prov))
    )
