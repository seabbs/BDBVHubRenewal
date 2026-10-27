"""Write the body and title of the hub pull request.

The body leads with what a reviewer needs: that this is an automated
submission, which step of the fallback chain produced it and why, a
short status table and the nowcast plot. Details (diagnostics, checks,
the chain) follow in collapsed sections, and the body ends by
mentioning @seabbs.
"""

from __future__ import annotations

import argparse
import datetime as dt
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .check_coherence import Coherence
from .check_coherence import details as coherence_details
from .common import (
    STATUS_LABEL,
    fmt,
    fmt_rhat,
    markdown_table,
    read_json,
    run_url,
    worst,
    write_text,
)
from .hubdata import quantile, read_samples

MAX_BODY = 60000  # GitHub's limit is 65536 characters
SOURCE_REPO = "seabbs/bdbv-hub-baseline-forecasts"
HUB = "InsightNet-US/BDBV-Modeling-Hub"
FOOTER = "This was opened by a bot. Please ping @seabbs for any questions."


def marker(model_id: str, reference_date: str) -> str:
    """Hidden marker that identifies the PR for a round."""
    return f"<!-- bdbv-submission:{model_id}:{reference_date} -->"


def title(model_id: str, reference_date: str) -> str:
    """PR title, in the form other teams use on this hub."""
    return f"{model_id} submission for {reference_date}"


def _strip_marker(md: str) -> str:
    return "\n".join(
        line for line in md.splitlines() if not line.startswith("<!--")
    ).strip()


def _demote(md: str) -> str:
    """Push headings down two levels so they sit inside a section."""
    return "\n".join(
        "##" + line if line.startswith("#") else line
        for line in md.splitlines()
    )


def _fit_cell(fit: Mapping[str, Any] | None) -> str:
    if fit is None:
        return "N/A (no sampler)"
    status = STATUS_LABEL[fit["status"]]
    if fit["status"] == "skip":
        return f"{status} (no sampler)"
    h = fit.get("headline") or {}
    return (
        f"{status} (max R-hat {fmt_rhat(h.get('max_rhat'))}, min bulk ESS "
        f"{fmt(h.get('min_ess_bulk'))}, {fmt(h.get('n_divergent'))} "
        f"divergences in {fmt(h.get('n_draws'))} draws)"
    )


def _ppc_cell(
    ppc: Mapping[str, Any] | None, fit: Mapping[str, Any] | None
) -> str:
    if ppc is None:
        return "N/A (no posterior)" if fit is None else "not run"
    status = STATUS_LABEL[ppc["status"]]
    if ppc["status"] == "skip":
        return f"{status} ({ppc.get('reason') or 'no posterior'})"
    detail = ppc.get("headline") or "; ".join(ppc.get("failures") or [])
    return f"{status} ({detail})" if detail else status


def _checks_cell(checks: Mapping[str, Any]) -> str:
    flagged = [
        c["name"] for c in checks["checks"] if c["status"] in ("warn", "fail")
    ]
    extra = f" ({', '.join(flagged)} flagged)" if flagged else ""
    return f"{STATUS_LABEL[checks['status']]}{extra}"


def _read_coherence(selected_dir: Path) -> Coherence | None:
    path = selected_dir / "coherence-summary.json"
    return Coherence(**read_json(path)) if path.exists() else None


def _coherence_cell(c: Coherence | None) -> str:
    if c is None:
        return "N/A (not checked)"
    return f"{STATUS_LABEL[c.status]} ({c.summary})"


def national_table(
    submission: Path, location: str = "CD", last: int = 7
) -> str:
    """Median and 90% interval for the last `last` weeks of a location."""
    s = read_samples(submission)
    if location not in s.values:
        return ""
    rows = []
    for w in s.weeks(location)[-last:]:
        v = s.week_values(location, w)
        rows.append(
            [
                w,
                fmt(quantile(v, 0.5)),
                f"{fmt(quantile(v, 0.05))} to {fmt(quantile(v, 0.95))}",
            ]
        )
    return markdown_table(["week ending", "median", "90% interval"], rows)


def build(
    *,
    provenance: Mapping[str, Any],
    selected_dir: Path,
    reference_date: str,
    model_id: str,
    hub_ref: str,
    data_commit: str,
    as_of: str,
    plot_url: str,
    validation: str,
    dry_run: bool = False,
    source_repo: str = SOURCE_REPO,
    source_sha: str = "",
    hub: str = HUB,
    now: dt.datetime | None = None,
) -> str:
    """Return the PR body markdown."""
    run = dict(provenance.get("run") or {})
    fit_path = selected_dir / "fit-summary.json"
    fit = read_json(fit_path) if fit_path.exists() else None
    ppc_path = selected_dir / "ppc-summary.json"
    ppc = read_json(ppc_path) if ppc_path.exists() else None
    checks = read_json(selected_dir / "checks.json")
    subs = sorted((selected_dir / "submission").glob("**/*.csv"))
    sub = subs[0]
    samples = read_samples(sub)
    coherence = _read_coherence(selected_dir)
    # The data check only warns, and is reported once here rather than
    # with each candidate's warnings, but it still colours the overall.
    status = worst(
        [provenance["status"]]
        + (
            [coherence.status]
            if coherence and coherence.status == "warn"
            else []
        )
    )
    link = run_url()
    stamp = (now or dt.datetime.now(dt.UTC)).strftime("%Y-%m-%d %H:%M UTC")

    out = [marker(model_id, reference_date)]
    out.append(f"## {title(model_id, reference_date)}\n")
    note = (
        "> [!NOTE]\n"
        "> **Automated submission.** A scheduled GitHub Actions workflow in "
        f"[{source_repo}](https://github.com/{source_repo}) fitted the model,"
        " ran the checks below and opened this pull request. No person "
        "reviewed it before it was opened. If the hub data change before "
        "the window closes, the workflow updates this pull request in place."
    )
    if dry_run:
        note += "\n>\n> This is a **dry run**: nothing was pushed or opened."
    out.append(note + "\n")
    out.append(f"**Produced by: {provenance['produced_by']}.** ")
    out.append(provenance["reason"] + "\n")
    out.append(
        markdown_table(
            ["check", "status"],
            [
                ["Overall", STATUS_LABEL[status]],
                ["Convergence", _fit_cell(fit)],
                ["Posterior predictive checks", _ppc_cell(ppc, fit)],
                ["Forecast checks", _checks_cell(checks)],
                ["Hub validation", STATUS_LABEL.get(validation, validation)],
                ["Data coherence", _coherence_cell(coherence)],
            ],
        )
        + "\n"
    )
    warnings = provenance.get("warnings") or []
    if warnings:
        out.append("> [!WARNING]\n> Submitted with warnings:")
        out.extend(f"> - {w}" for w in warnings)
        out.append("")

    out.append("### Nowcast and forecast\n")
    out.append(
        f"![Nowcast and forecast for {reference_date}]({plot_url})\n\n"
        "Observed cases by onset week in the data vintage used, with the "
        "submitted median and 50% and 90% intervals. Weeks up to the "
        "reference date are a nowcast; later weeks are a forecast.\n"
    )
    if (selected_dir / "ppc.png").exists():
        out.append(
            "The posterior predictive check plot (`ppc.png`, predictive "
            "intervals against the observed counts by province) is in the "
            "run's candidate artefact.\n"
        )
    nat = national_table(sub)
    if nat:
        out.append("National (`CD`) summary of the submitted samples:\n")
        out.append(nat + "\n")

    out.append("### Submission\n")
    hub_link = f"https://github.com/{hub}/tree/{hub_ref}"
    data_link = f"https://github.com/{hub}/commit/{data_commit}"
    code = (
        f"[`{source_sha[:10]}`](https://github.com/{source_repo}/commit/"
        f"{source_sha})"
        if source_sha
        else "n/a"
    )
    out.append(
        markdown_table(
            ["", ""],
            [
                ["Model", f"`{model_id}`"],
                [
                    "Model and config",
                    f"{run.get('model', 'n/a')} (`{run.get('config', 'n/a')}`)",
                ],
                [
                    "Version",
                    f"{run.get('package_version', 'n/a')}, code {code}",
                ],
                ["Reference date", reference_date],
                ["Data vintage (`as_of`)", as_of or "n/a"],
                ["Hub data commit", f"[`{data_commit[:10]}`]({data_link})"],
                ["Hub checkout", f"[`{hub_ref[:10]}`]({hub_link})"],
                ["File", f"`model-output/{model_id}/{sub.name}`"],
                [
                    "Contents",
                    f"{len(samples.locations)} locations, {samples.rows} rows",
                ],
                ["Built", stamp],
            ],
        )
        + "\n"
    )
    details = [
        ("Fallback chain", (selected_dir.parent / "provenance.md")),
        ("Convergence diagnostics", selected_dir / "fit-summary.md"),
        ("Posterior predictive checks", selected_dir / "ppc-summary.md"),
        ("Forecast checks", selected_dir / "checks.md"),
    ]
    for label, path in details:
        if path.exists():
            body = _demote(_strip_marker(path.read_text()))
            out.append(
                f"<details><summary>{label}</summary>\n\n{body}\n\n</details>\n"
            )
    if coherence is not None and coherence.status != "skip":
        out.append(
            "<details><summary>Data coherence</summary>\n\n"
            f"National against the sum of the provinces in the data vintage "
            f"used.\n\n{coherence_details(coherence)}\n</details>\n"
        )
    if link:
        out.append(f"Workflow run and artefacts: {link}\n")
    out.append(FOOTER)
    text = "\n".join(out) + "\n"
    if len(text) > MAX_BODY:
        cut = text[: MAX_BODY - 400]
        text = (
            cut
            + "\n\n</details>\n\n(Truncated; the full report is in the run's "
            f"artefacts: {link or 'see the workflow run'}.)\n\n{FOOTER}\n"
        )
    return text


def selected_validation(provenance: Mapping[str, Any]) -> str:
    """Hub validation result of the selected attempt."""
    for a in provenance.get("attempts", []):
        if a["id"] == provenance.get("selected"):
            return str(a["validation"])
    return "not run"


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--gate",
        type=Path,
        required=True,
        help="directory with provenance.json and selected/",
    )
    p.add_argument("--reference-date", required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--hub-ref", required=True)
    p.add_argument("--data-commit", required=True)
    p.add_argument("--as-of", default="")
    p.add_argument("--plot-url", required=True)
    p.add_argument(
        "--validation",
        default="",
        help="defaults to the selected attempt's result",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--source-sha", default="")
    p.add_argument("--source-repo", default=SOURCE_REPO)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    prov = read_json(args.gate / "provenance.json")
    validation = args.validation or selected_validation(prov)
    body = build(
        provenance=prov,
        selected_dir=args.gate / "selected",
        reference_date=args.reference_date,
        model_id=args.model_id,
        hub_ref=args.hub_ref,
        data_commit=args.data_commit,
        as_of=args.as_of,
        plot_url=args.plot_url,
        validation=validation,
        dry_run=args.dry_run,
        source_repo=args.source_repo,
        source_sha=args.source_sha,
    )
    write_text(args.out / "pr-body.md", body)
    write_text(
        args.out / "pr-title.txt", title(args.model_id, args.reference_date)
    )
    print(f"wrote {args.out / 'pr-body.md'} ({len(body)} characters)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
