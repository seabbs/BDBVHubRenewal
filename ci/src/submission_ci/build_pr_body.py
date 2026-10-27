"""Write the body and title of the hub pull request.

The hub's organisers read the body, so it is short enough to scan: one
line on what was submitted, one plain status line, one `Handled:` line
on any problem the workflow dealt with and how, the nowcast plot, a link
to the round's page in our docs and the bot footer. Everything else
(the checks, the fallback chain, diagnostics, provenance and a national
summary) sits in short collapsed sections. The full reports stay in the
job summary and the run's artefacts.
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
    write_text,
)
from .hubdata import quantile, read_samples

MAX_BODY = 60000  # GitHub's limit is 65536 characters
SOURCE_REPO = "seabbs/BDBVHubRenewal"
HUB = "InsightNet-US/BDBV-Modeling-Hub"
FOOTER = "This was opened by a bot. Please ping @seabbs for any questions."
DOCS = "https://seabbs.github.io/BDBVHubRenewal"


def round_url(reference_date: str, docs: str = DOCS) -> str:
    """Stable docs page of a round's submitted nowcast."""
    return f"{docs.rstrip('/')}/submission/rounds/{reference_date}/"


def marker(model_id: str, reference_date: str) -> str:
    """Hidden marker that identifies the PR for a round."""
    return f"<!-- bdbv-submission:{model_id}:{reference_date} -->"


def title(model_id: str, reference_date: str) -> str:
    """PR title, in the form other teams use on this hub."""
    return f"{model_id} submission for {reference_date}"


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
    if c.gap:
        return f"Handled ({c.summary})"
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


def _warning_names(warnings: Sequence[str]) -> list[str]:
    """Check names from `name: message` warnings, in order, once each."""
    names: list[str] = []
    for w in warnings:
        name = w.split(":", 1)[0].strip()
        if name and name not in names:
            names.append(name)
    return names


def _first_reason(provenance: Mapping[str, Any]) -> str:
    for a in provenance.get("attempts", []):
        if a.get("id") != provenance.get("selected") and a.get("reasons"):
            return f"{a['label']} was ruled out ({a['reasons'][0]})"
    return ""


def status_line(
    provenance: Mapping[str, Any],
    coherence: Coherence | None = None,
    dry_run: bool = False,
) -> str:
    """One plain sentence (or three) on how the submission came about."""
    kind = provenance.get("kind")
    names = _warning_names(provenance.get("warnings") or [])
    if coherence is not None and coherence.status == "warn":
        names.append("data coherence")
    why = _first_reason(provenance)
    if kind == "retry" and why:
        head = f"Main model refitted once: {why[0].lower()}{why[1:]}."
    elif kind in ("fallback", "persistence"):
        head = (
            f"Fallback model used ({provenance['produced_by']})"
            + (f": {why[0].lower()}{why[1:]}" if why else "")
            + "."
        )
    else:
        head = "All checks passed."
    if names:
        head = head[:-1] + f", with warnings on {', '.join(names)}."
    run = provenance.get("run") or {}
    if provenance.get("stand_in"):
        head += (
            f" Stand-in: {run.get('model', 'persistence')} runs as the "
            f"`{run.get('config', 'main')}` model."
        )
    if dry_run:
        head += " Dry run: nothing was pushed."
    return head


def handled_line(
    provenance: Mapping[str, Any], coherence: Coherence | None = None
) -> str:
    """One `Handled:` line on the problems dealt with, or empty."""
    notes = [str(h) for h in provenance.get("handled") or []]
    if coherence is not None and coherence.handled():
        notes.append(coherence.handled())
    if not notes:
        return ""
    text = "; ".join(notes)
    return f"Handled: {text}{'' if text.endswith('.') else '.'}"


def _section(label: str, body: str) -> str:
    return (
        f"<details><summary>{label}</summary>\n\n{body.strip()}\n\n</details>\n"
    )


def _checks_section(
    fit: Mapping[str, Any] | None,
    ppc: Mapping[str, Any] | None,
    checks: Mapping[str, Any],
    validation: str,
    coherence: Coherence | None,
) -> str:
    table = markdown_table(
        ["check", "status"],
        [
            ["Convergence", _fit_cell(fit)],
            ["Posterior predictive checks", _ppc_cell(ppc, fit)],
            ["Forecast checks", _checks_cell(checks)],
            ["Hub validation", STATUS_LABEL.get(validation, validation)],
            ["Data coherence", _coherence_cell(coherence)],
        ],
    )
    return _section("Checks", table)


def _why(a: Mapping[str, Any]) -> str:
    """Why an attempt was passed over: its first reason, or its refit."""
    if a.get("reasons"):
        return str(a["reasons"][0])
    if a.get("retry"):
        return next(
            (str(w) for w in a.get("warnings", []) if "retry threshold" in w),
            "refitted once",
        )
    return ""


def _chain_section(provenance: Mapping[str, Any]) -> str:
    ran = [
        a
        for a in provenance.get("attempts", [])
        if a.get("status") not in ("not needed", "not run")
    ]
    if len(ran) < 2:
        return ""
    rows = [
        [
            a["label"],
            "submitted"
            if a["id"] == provenance.get("selected")
            else STATUS_LABEL.get(a["status"], a["status"]),
            _why(a),
        ]
        for a in ran
    ]
    return _section(
        "Fallback chain", markdown_table(["step", "outcome", "why"], rows)
    )


def _convergence_section(fit: Mapping[str, Any] | None) -> str:
    if fit is None or fit.get("status") == "skip":
        return ""
    lines = [f"{_fit_cell(fit)}."]
    lines += [f"- {m}" for m in (fit.get("failures") or [])]
    lines += [f"- {m}" for m in (fit.get("warnings") or [])]
    return _section("Convergence", "\n".join(lines))


def _ppc_section(ppc: Mapping[str, Any] | None) -> str:
    if ppc is None or not ppc.get("available"):
        return ""
    levels = list(ppc.get("levels") or [])
    weeks = len(ppc.get("recent_weeks") or [])
    stream = ppc.get("stream", "total")
    overall = ppc.get("streams", {}).get(stream, {})
    rows = [
        [
            "all provinces",
            *(fmt(overall.get(f"cov{p}"), 2) for p in levels),
            *(fmt(overall.get(f"recent_cov{p}"), 2) for p in levels),
        ]
    ] + [
        [
            str(r["location"]),
            *(fmt(r.get(f"cov{p}"), 2) for p in levels),
            *(fmt(r.get(f"recent_cov{p}"), 2) for p in levels),
        ]
        for r in ppc.get("locations", [])
    ]
    table = markdown_table(
        [
            "",
            *(f"{p}%" for p in levels),
            *(f"{p}%, last {weeks} weeks" for p in levels),
        ],
        rows,
    )
    pit = ppc.get("pit") or {}
    note = (
        f"\n\nPIT: KS distance {fmt(pit.get('ks'), 2)} from uniform"
        + (f"; {pit['shape']}" if pit.get("shape") else "")
        + "."
        if pit.get("n")
        else ""
    )
    return _section(
        "Posterior predictive checks",
        f"Share of observed weekly {stream} counts inside the in-sample "
        f"predictive intervals.\n\n{table}{note}",
    )


def _forecast_section(checks: Mapping[str, Any]) -> str:
    flagged = [
        c for c in checks["checks"] if c["status"] not in ("pass", "skip")
    ]
    passed = [c["name"] for c in checks["checks"] if c["status"] == "pass"]
    lines = [
        f"- {STATUS_LABEL[c['status']]} {c['name']}: {c['summary']}"
        for c in flagged
    ]
    if passed:
        lines.append(f"- Passed: {', '.join(passed)}.")
    return _section("Forecast checks", "\n".join(lines))


def _provenance_section(
    *,
    run: Mapping[str, Any],
    model_id: str,
    sub: Path,
    samples: Any,
    as_of: str,
    hub: str,
    hub_ref: str,
    data_commit: str,
    source_repo: str,
    source_sha: str,
    stamp: str,
    link: str,
) -> str:
    code = (
        f"[`{source_sha[:10]}`](https://github.com/{source_repo}/commit/"
        f"{source_sha})"
        if source_sha
        else "n/a"
    )
    rows = [
        ["File", f"`model-output/{model_id}/{sub.name}`"],
        [
            "Contents",
            f"{len(samples.locations)} locations, {samples.rows} rows",
        ],
        [
            "Model",
            f"{run.get('model', 'n/a')} (`{run.get('config', 'n/a')}`), "
            f"version {run.get('package_version', 'n/a')}, code {code}",
        ],
        ["Data vintage (`as_of`)", as_of or "n/a"],
        [
            "Hub data commit",
            f"[`{data_commit[:10]}`](https://github.com/{hub}/commit/"
            f"{data_commit})",
        ],
        [
            "Hub config",
            f"[`{hub_ref[:10]}`](https://github.com/{hub}/tree/{hub_ref})",
        ],
        ["Built", stamp + (f", [workflow run]({link})" if link else "")],
    ]
    return _section("Provenance", markdown_table(["", ""], rows))


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
    docs: str = DOCS,
    now: dt.datetime | None = None,
) -> str:
    """Return the PR body markdown."""
    run = dict(provenance.get("run") or {})
    fit_path = selected_dir / "fit-summary.json"
    fit = read_json(fit_path) if fit_path.exists() else None
    ppc_path = selected_dir / "ppc-summary.json"
    ppc = read_json(ppc_path) if ppc_path.exists() else None
    checks = read_json(selected_dir / "checks.json")
    sub = sorted((selected_dir / "submission").glob("**/*.csv"))[0]
    samples = read_samples(sub)
    coherence = _read_coherence(selected_dir)
    link = run_url()
    stamp = (now or dt.datetime.now(dt.UTC)).strftime("%Y-%m-%d %H:%M UTC")
    out = [marker(model_id, reference_date)]
    out.append(
        f"Automated submission of `{model_id}` for {reference_date}, from "
        f"data as of {as_of or 'n/a'}, for {len(samples.locations)} "
        f"locations.\n"
    )
    out.append(status_line(provenance, coherence, dry_run) + "\n")
    handled = handled_line(provenance, coherence)
    if handled:
        out.append(handled + "\n")
    out.append(f"![Nowcast and forecast for {reference_date}]({plot_url})\n")
    out.append(f"This round in our docs: {round_url(reference_date, docs)}\n")
    sections = [
        _checks_section(fit, ppc, checks, validation, coherence),
        _chain_section(provenance),
        _convergence_section(fit),
        _ppc_section(ppc),
        _forecast_section(checks),
        _section(
            "Data coherence",
            f"National against the sum of the provinces in the data.\n\n"
            f"{coherence_details(coherence)}",
        )
        if coherence is not None
        and (coherence.gap or coherence.status == "warn")
        else "",
    ]
    nat = national_table(sub)
    if nat:
        sections.append(
            _section("National summary", f"Submitted `CD` samples.\n\n{nat}")
        )
    sections.append(
        _provenance_section(
            run=run,
            model_id=model_id,
            sub=sub,
            samples=samples,
            as_of=as_of,
            hub=hub,
            hub_ref=hub_ref,
            data_commit=data_commit,
            source_repo=source_repo,
            source_sha=source_sha,
            stamp=stamp,
            link=link,
        )
    )
    out.extend(x for x in sections if x)
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
    p.add_argument("--docs", default=DOCS, help="base URL of the docs site")
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
        docs=args.docs,
    )
    write_text(args.out / "pr-body.md", body)
    write_text(
        args.out / "pr-title.txt", title(args.model_id, args.reference_date)
    )
    print(f"wrote {args.out / 'pr-body.md'} ({len(body)} characters)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
