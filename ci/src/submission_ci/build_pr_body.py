"""Write the body and title of the hub pull request.

The hub's organisers read the body, so it is short: what was submitted,
Sam's sign-off, the model in one sentence, a summary of national and
each province with cases against their recent reported level, a checks
table with one status per check, the round's page in our docs and the
bot footer. The national weekly table, convergence, posterior predictive
checks and the forecast checks sit in one collapsed Details block. The full
reports stay in the job summary and the run's artefacts.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .check_coherence import Coherence
from .common import (
    STATUS_LABEL,
    fmt,
    fmt_rhat,
    markdown_table,
    read_json,
    run_url,
    write_text,
)
from .hubdata import quantile, read_observed, read_samples

MAX_BODY = 60000  # GitHub's limit is 65536 characters
HUB = "InsightNet-US/BDBV-Modeling-Hub"
FOOTER = "This was opened by a bot. Please ping @seabbs for any questions."
SIGN_OFF = "Signed off by Sam Abbott (@seabbs)."
DOCS = "https://samabbott.co.uk/BDBVHubRenewal"


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


def _read_coherence(selected_dir: Path) -> Coherence | None:
    path = selected_dir / "coherence-summary.json"
    return Coherence(**read_json(path)) if path.exists() else None


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


def _first_reason(provenance: Mapping[str, Any]) -> str:
    for a in provenance.get("attempts", []):
        if a.get("id") != provenance.get("selected") and a.get("reasons"):
            return f"{a['label']} was ruled out ({a['reasons'][0]})"
    return ""


def _first_choice(provenance: Mapping[str, Any]) -> bool:
    """Whether the submitted candidate was the first in its chain, as for
    a hub model of its own, so it is no fallback."""
    attempts = provenance.get("attempts") or []
    return bool(attempts) and attempts[0].get("id") == provenance.get(
        "selected"
    )


def _section(label: str, body: str) -> str:
    return (
        f"<details><summary>{label}</summary>\n\n{body.strip()}\n\n</details>\n"
    )


def _why(a: Mapping[str, Any]) -> str:
    """Why an attempt was passed over: its first reason."""
    return str(a["reasons"][0]) if a.get("reasons") else ""


def bvd_lines(who: str, fit: Mapping[str, Any] | None) -> list[str]:
    """The BVDOutbreakSize release a fit used, and the newer ones skipped."""
    release = (fit or {}).get("bvd")
    if not release:
        return []
    if not release.get("tag"):
        return [
            f"{who} used no BVDOutbreakSize release: none qualified, so it "
            "ran as the main model."
        ]
    lines = [
        f"{who} used BVDOutbreakSize release `{release['tag']}` (data to "
        f"{release.get('as_of', 'n/a')}, joint R-hat "
        f"{fmt_rhat(release.get('joint_max_rhat'))}, limit "
        f"{release.get('max_rhat', 'n/a')})."
    ]
    lines += [
        f"- Skipped `{s['tag']}`: {s['reason']}."
        for s in release.get("skipped") or []
    ]
    return lines


def _convergence_section(fit: Mapping[str, Any] | None) -> str:
    if fit is None or fit.get("status") == "skip":
        return ""
    lines = [f"{_fit_cell(fit)}."]
    lines += [f"- {m}" for m in (fit.get("failures") or [])]
    lines += [f"- {m}" for m in (fit.get("warnings") or [])]
    bvd = bvd_lines("The model", fit)
    if bvd:
        lines += ["", *bvd]
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


def _names(observed: Path) -> dict[str, str]:
    with observed.open(newline="", encoding="utf-8") as fh:
        return {r["location"]: r["location_name"] for r in csv.DictReader(fh)}


def summary_table(
    submission: Path,
    observed: Path,
    as_of: str,
    reference_date: str,
    lag: int,
    n: int,
) -> str:
    """National and each province with cases: recent level, nowcast, forecast.

    The recent level is the mean of the last `n` weeks reported in full
    (ending at least `lag` weeks before the data vintage), as the gate's
    `observed` check uses. The nowcast is the newest reported week and the
    forecast the reference date's week, as median and 90% interval.
    """
    s = read_samples(submission)
    obs = read_observed(observed, as_of or None)
    names = _names(observed)
    cutoff = (
        dt.date.fromisoformat(obs.as_of) - dt.timedelta(days=7 * lag)
    ).isoformat()
    places = [
        "CD",
        *sorted(
            loc
            for loc in s.locations
            if loc.count("-") == 1 and sum(obs.counts.get(loc, {}).values()) > 0
        ),
    ]

    def interval(v: list[float]) -> str:
        return (
            f"{fmt(quantile(v, 0.5))} ({fmt(quantile(v, 0.05))} to "
            f"{fmt(quantile(v, 0.95))})"
        )

    rows = []
    for loc in places:
        counts = obs.counts.get(loc, {})
        done = sorted(w for w in counts if w <= cutoff)[-n:]
        recent = sum(counts[w] for w in done) / len(done) if done else 0.0
        newest = max(counts)
        fc = s.week_values(loc, reference_date)
        rows.append(
            [
                names.get(loc, loc),
                fmt(recent),
                interval(s.week_values(loc, newest)),
                interval(fc),
                f"{quantile(fc, 0.5) / recent:.2f}" if recent else "n/a",
            ]
        )
    newest = max(obs.counts["CD"])
    return markdown_table(
        [
            "",
            f"reported, mean of last {n} complete weeks",
            f"nowcast, week ending {newest}",
            f"forecast, week ending {reference_date}",
            "forecast / reported",
        ],
        rows,
    )


def model_line(provenance: Mapping[str, Any]) -> str:
    """The submitted model in one sentence, and why it was the one."""
    run = provenance.get("run") or {}
    label = provenance.get("produced_by") or "the model"
    model = run.get("model")
    name = f"{label} (`{model}`)" if model else label
    attempts = provenance.get("attempts") or []
    if len(attempts) < 2:
        return f"Model: {name}."
    if _first_choice(provenance):
        return (
            f"Model: {name}, the first model in our chain and the first to "
            "clear its checks."
        )
    return f"Model: {name}, because {_first_reason(provenance)}."


def checks_table(
    fit: Mapping[str, Any] | None,
    ppc: Mapping[str, Any] | None,
    checks: Mapping[str, Any],
    validation: str,
    coherence: Coherence | None,
) -> str:
    """One row per check, with its status only."""

    def label(status: str | None) -> str:
        return STATUS_LABEL.get(status, status) if status else "N/A"

    rows: list[tuple[str, str | None]] = [
        ("Convergence", fit["status"] if fit else None),
        ("Posterior predictive checks", ppc["status"] if ppc else None),
        ("Forecast checks", checks["status"]),
        ("Hub validation", validation),
        ("Data coherence", coherence.status if coherence else None),
    ]
    return markdown_table(
        ["check", "status"], [[name, label(s)] for name, s in rows]
    )


def _details(
    fit: Mapping[str, Any] | None,
    ppc: Mapping[str, Any] | None,
    checks: Mapping[str, Any],
    national: str = "",
) -> str:
    """The national weekly table and the checks' detail, collapsed."""
    parts = [f"**National (`CD`) by week**\n\n{national}"] if national else []
    for title_, block in (
        ("Convergence", _convergence_section(fit)),
        ("Posterior predictive checks", _ppc_section(ppc)),
        ("Forecast checks", _forecast_section(checks)),
    ):
        if block:
            inner = block.split("</summary>\n\n", 1)[1].rsplit(
                "\n\n</details>", 1
            )[0]
            parts.append(f"**{title_}**\n\n{inner}")
    return _section("Details", "\n\n".join(parts)) if parts else ""


def build(
    *,
    provenance: Mapping[str, Any],
    selected_dir: Path,
    reference_date: str,
    model_id: str,
    as_of: str,
    plot_url: str,
    validation: str,
    observed: Path | None = None,
    complete_lag_weeks: int = 4,
    complete_weeks: int = 2,
    dry_run: bool = False,
    docs: str = DOCS,
) -> str:
    """Return the PR body markdown."""
    fit_path = selected_dir / "fit-summary.json"
    fit = read_json(fit_path) if fit_path.exists() else None
    ppc_path = selected_dir / "ppc-summary.json"
    ppc = read_json(ppc_path) if ppc_path.exists() else None
    checks = read_json(selected_dir / "checks.json")
    sub = sorted((selected_dir / "submission").glob("**/*.csv"))[0]
    samples = read_samples(sub)
    coherence = _read_coherence(selected_dir)
    link = run_url()
    out = [marker(model_id, reference_date)]
    out.append(
        f"Submission of `{model_id}` for {reference_date}, from data as of "
        f"{as_of or 'n/a'}, for {len(samples.locations)} locations.\n"
    )
    out.append(SIGN_OFF + "\n")
    out.append(model_line(provenance) + "\n")
    if dry_run:
        out.append("Dry run: nothing was pushed.\n")
    if plot_url.startswith("http"):
        out.append(
            f"![Nowcast and forecast for {reference_date}]({plot_url})\n"
        )
    if observed is not None:
        out.append(
            summary_table(
                sub,
                observed,
                as_of,
                reference_date,
                complete_lag_weeks,
                complete_weeks,
            )
            + "\n"
        )
    out.append(checks_table(fit, ppc, checks, validation, coherence) + "\n")
    out.append(f"This round in our docs: {round_url(reference_date, docs)}\n")
    details = _details(fit, ppc, checks, national_table(sub))
    if details:
        out.append(details)
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
    p.add_argument("--as-of", default="")
    p.add_argument("--plot-url", required=True)
    p.add_argument(
        "--validation",
        default="",
        help="defaults to the selected attempt's result",
    )
    p.add_argument(
        "--observed", type=Path, default=None, help="time-series.csv"
    )
    p.add_argument("--policy", type=Path, default=None, help="policy.toml")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--docs", default=DOCS, help="base URL of the docs site")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    prov = read_json(args.gate / "provenance.json")
    validation = args.validation or selected_validation(prov)
    checks_cfg = (
        tomllib.loads(args.policy.read_text())["checks"] if args.policy else {}
    )
    body = build(
        provenance=prov,
        selected_dir=args.gate / "selected",
        reference_date=args.reference_date,
        model_id=args.model_id,
        as_of=args.as_of,
        plot_url=args.plot_url,
        validation=validation,
        observed=args.observed,
        complete_lag_weeks=int(checks_cfg.get("complete_lag_weeks", 4)),
        complete_weeks=int(checks_cfg.get("complete_weeks", 2)),
        dry_run=args.dry_run,
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
