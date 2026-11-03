"""Check a candidate submission file before it can be submitted.

Each check returns pass, warn, fail or skip. Only a file the hub would
reject fails, which blocks the candidate; everything else is handled or
warns, and is stated in the PR body. Thresholds are in `policy.toml`
under `[checks]`.

Format (fail):
  `schema`       columns, output type, target, reference date, Saturdays,
                 and, given `tasks.json`, valid locations and weeks.
  `values`       no negative, NaN, infinite or unparsable values.
  `samples`      exactly 100 samples per location, each a full trajectory.

Handled (warn or pass):
  `coverage`     every expected location, with every nowcast and
                 forecast week (warns).
  `zones`        health zone rows (`CD-XX-NN`) for every province in the
                 file, which seabbs-rwnowcast requires (`--require-zones`),
                 and each province's zones summing to it (warns only).
  `national_sum` national equals the sum of the provinces plus the
                 unassigned cases in `national.json`, per sample and
                 week (fails otherwise). A modelled province missing
                 from the file warns.

Plausibility (`plausible` fails; the others warn):
  `plausible`    each location's forecast near its recent data: median
                 and 97.5% quantile under set multiples of the last 4
                 weeks' maximum, and the median not collapsed. Fails.
  `bounds`       the 95th centile of each week is not far above the
                 largest weekly count the location has had.
  `growth`       week-on-week growth of the median over the forecast.
  `observed`     the median of the last complete weeks is near what was
                 observed.

Consistency (warn):
  `baseline`     forecast median against the random walk baseline.
  `previous`     shared weeks against last week's submission.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .common import (
    STATUS_LABEL,
    append_summary,
    fmt,
    markdown_table,
    read_json,
    set_outputs,
    worst,
    write_json,
    write_text,
)
from .forecast_date import parse_date
from .hubdata import (
    SUBMISSION_COLUMNS,
    TARGET,
    Observed,
    Samples,
    is_province,
    is_saturday,
    quantile,
    read_observed,
    read_samples,
)
from .policy import load_policy

MAX_DETAILS = 8


@dataclass
class Check:
    """The result of one check."""

    name: str
    status: str
    summary: str
    details: list[str] = field(default_factory=list)


def _level(ratio: float, warn: float) -> str:
    """Status of a ratio against a symmetric warn level."""
    return "warn" if ratio > warn or ratio < 1 / warn else "pass"


def _week(ref: dt.date, k: int) -> str:
    return (ref + dt.timedelta(days=7 * k)).isoformat()


def _medians(s: Samples, loc: str, weeks: Sequence[str]) -> list[float]:
    return [quantile(s.week_values(loc, w), 0.5) for w in weeks]


def _result(name: str, found: list[tuple[str, str]], ok: str) -> Check:
    """Fold per-item findings into one check."""
    status = worst(st for st, _ in found)
    shown = [f"{STATUS_LABEL[st]}: {msg}" for st, msg in found if st != "pass"]
    if status == "pass":
        return Check(name, "pass", ok)
    n = len(shown)
    first = next(msg for st, msg in found if st == status)
    summary = first if n == 1 else f"{first} (and {n - 1} more)"
    extra = [f"... and {n - MAX_DETAILS} more"] if n > MAX_DETAILS else []
    return Check(name, status, summary, shown[:MAX_DETAILS] + extra)


def _forecast_date_problems(s: Samples, ref: str) -> list[tuple[str, str]]:
    """A new file has one valid `forecast_date`, on or before the round's."""
    if "forecast_date" not in s.columns:
        return []  # the missing column is reported with the others
    dates = sorted(s.forecast_dates)
    bad = [d for d in dates if parse_date(d) is None]
    if bad:
        return [("fail", f"forecast_date is not YYYY-MM-DD: {bad[:3]}")]
    if len(dates) != 1:
        return [("fail", f"a new forecast has one forecast_date, not {dates}")]
    if dates[0] > ref:
        return [("fail", f"forecast_date {dates[0]} is after {ref}")]
    return []


def check_schema(
    s: Samples,
    ref: str,
    file_name: str | None,
    tasks: Mapping[str, Any] | None,
) -> Check:
    """Columns, types, dates and hub task options."""
    found: list[tuple[str, str]] = []
    cols = set(s.columns)
    if cols != SUBMISSION_COLUMNS:
        missing = sorted(SUBMISSION_COLUMNS - cols)
        extra = sorted(cols - SUBMISSION_COLUMNS)
        found.append(("fail", f"columns: missing {missing}, extra {extra}"))
    if s.output_types != {"sample"}:
        found.append(("fail", f"output types {sorted(s.output_types)}"))
    if s.targets != {TARGET}:
        found.append(("fail", f"targets {sorted(s.targets)}"))
    if s.reference_dates != {ref}:
        found.append(
            ("fail", f"reference dates {sorted(s.reference_dates)}, not {ref}")
        )
    found.extend(_forecast_date_problems(s, ref))
    if file_name is not None and not file_name.startswith(f"{ref}-"):
        found.append(("fail", f"file name {file_name} does not start {ref}-"))
    if s.duplicates:
        found.append(("fail", f"{s.duplicates} duplicated rows"))
    weeks = {w for loc in s.locations for w in s.weeks(loc)}
    bad = sorted(w for w in weeks if not is_saturday(w))
    if bad:
        found.append(("fail", f"target end dates not Saturdays: {bad[:5]}"))
    if tasks:
        from .detect_new_vintage import open_reference_dates

        opts = _task_options(tasks)
        if ref not in open_reference_dates(tasks):
            found.append(("fail", f"{ref} is not an open reference date"))
        bad_loc = sorted(set(s.locations) - opts["location"])
        if opts["location"] and bad_loc:
            found.append(
                ("fail", f"locations not in tasks.json: {bad_loc[:5]}")
            )
        bad_w = sorted(weeks - opts["target_end_date"])
        if opts["target_end_date"] and bad_w:
            found.append(("fail", f"weeks not in tasks.json: {bad_w[:5]}"))
    return _result("schema", found, "format matches the hub's round")


def _task_options(tasks: Mapping[str, Any]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {"location": set(), "target_end_date": set()}
    for rnd in tasks.get("rounds", []):
        for mt in rnd.get("model_tasks", []):
            ids = mt.get("task_ids", {})
            t = ids.get("target", {})
            if TARGET not in (t.get("required") or []) + (
                t.get("optional") or []
            ):
                continue
            for key, values in out.items():
                v = ids.get(key, {})
                values.update(
                    (v.get("required") or []) + (v.get("optional") or [])
                )
    return out


def check_values(s: Samples) -> Check:
    """No negative, missing or infinite values."""
    found: list[tuple[str, str]] = [
        ("fail", f"unparsable value at {u}") for u in s.unparsable
    ]
    for loc in s.locations:
        vals = [v for traj in s.values[loc].values() for v in traj.values()]
        nan = sum(math.isnan(v) for v in vals)
        inf = sum(math.isinf(v) for v in vals)
        neg = sum(v < 0 for v in vals if math.isfinite(v))
        for n, what in ((nan, "NaN"), (inf, "infinite"), (neg, "negative")):
            if n:
                found.append(("fail", f"{loc}: {n} {what} values"))
    return _result("values", found, "all values finite and non-negative")


def check_samples(s: Samples, n: int) -> Check:
    """Exactly `n` samples per location, each covering the same weeks."""
    found: list[tuple[str, str]] = []
    for loc in s.locations:
        trajs = s.values[loc]
        if len(trajs) != n:
            found.append(("fail", f"{loc}: {len(trajs)} samples, not {n}"))
        weeks = set(s.weeks(loc))
        short = [sid for sid, t in trajs.items() if set(t) != weeks]
        if short:
            found.append(
                (
                    "fail",
                    f"{loc}: {len(short)} samples miss some weeks "
                    f"(e.g. sample {short[0]})",
                )
            )
    return _result("samples", found, f"{n} full trajectories per location")


def check_coverage(
    s: Samples,
    ref: dt.date,
    expected: Sequence[str],
    horizon: int,
    nowcast: int,
) -> Check:
    """Every expected location, nowcast week and forecast week."""
    found: list[tuple[str, str]] = []
    if not s.locations:
        found.append(("fail", "the file has no rows"))
    missing_loc = sorted(set(expected) - set(s.locations))
    if missing_loc:
        found.append(("warn", f"missing locations {missing_loc}"))
    need = [_week(ref, k) for k in range(-(nowcast - 1), horizon + 1)]
    for loc in s.locations:
        gap = [w for w in need if w not in s.weeks(loc)]
        if gap:
            found.append(("warn", f"{loc}: missing weeks {gap}"))
    return _result(
        "coverage",
        found,
        f"{len(s.locations)} locations, weeks {need[0]} to {need[-1]}",
    )


def _unassigned(sidecar: Mapping[str, Any]) -> dict[tuple[str, str], float]:
    """Unassigned national cases by (sample id, week) from `national.json`."""
    out: dict[tuple[str, str], float] = {}
    for row in sidecar["unassigned"]:
        key = (str(int(row["sample"])), str(row["target_end_date"]))
        out[key] = out.get(key, 0.0) + float(row["value"])
    return out


def check_national_sum(
    s: Samples, tol: float, sidecar: Mapping[str, Any]
) -> Check:
    """National equals the sum of provinces exactly, per sample and week.

    National sample k must equal the sum of province sample k plus the
    sample's modelled unassigned component (`national.json`), for every
    week, up to float noise (`tol` absolute, or 1e-12 relative). Every
    modelled province must be in the file.
    """
    national = str(sidecar["national"])
    provs = [loc for loc in s.locations if is_province(loc)]
    modelled = sorted(sidecar["modelled_provinces"])
    unassigned = _unassigned(sidecar)
    basis = f"national.json: {len(modelled)} modelled provinces" + (
        " plus the unassigned component" if unassigned else ""
    )
    found: list[tuple[str, str]] = []
    missing = sorted(set(modelled) - set(s.locations))
    if missing:
        found.append(
            (
                "warn",
                f"modelled provinces missing: {missing}; national is the "
                "sum of the provinces submitted",
            )
        )
    if national not in s.values:
        found.append(("fail", f"{national} is not in the file"))
    for sid, traj in s.values.get(national, {}).items():
        for week, value in traj.items():
            parts = [s.values[p].get(sid, {}).get(week) for p in provs]
            if any(v is None for v in parts):
                found.append(
                    ("fail", f"sample {sid}, {week}: a province lacks it")
                )
                continue
            extra = unassigned.get((sid, week), 0.0)
            total = math.fsum([*(v for v in parts if v is not None), extra])
            if not math.isclose(total, value, rel_tol=1e-12, abs_tol=tol):
                found.append(
                    (
                        "fail",
                        f"sample {sid}, {week}: national {fmt(value, 12)} "
                        f"but provinces{' and unassigned' if extra else ''} "
                        f"sum to {fmt(total, 12)}",
                    )
                )
    out = _result(
        "national_sum", found, f"{national} is the exact sum ({basis})"
    )
    if out.status != "pass":
        out.summary = f"{out.summary} ({basis})"
    return out


def check_bounds(s: Samples, obs: Observed, floor: float, warn: float) -> Check:
    """95th centile of each week against the largest observed week."""
    found: list[tuple[str, str]] = []
    for loc in s.locations:
        hist = max(obs.counts.get(loc, {}).values(), default=0.0)
        cap = max(hist, floor)
        top = max(
            (quantile(s.week_values(loc, w), 0.95) for w in s.weeks(loc)),
            default=0.0,
        )
        ratio = top / cap
        found.append(
            (
                "warn" if ratio > warn else "pass",
                f"{loc}: 95th centile {fmt(top)} is {fmt(ratio)} times the "
                f"bound base {fmt(cap)}",
            )
        )
    return _result("bounds", found, "trajectories within the bounds")


def check_plausible(
    s: Samples,
    obs: Observed,
    ref: dt.date,
    horizon: int,
    c: Mapping[str, Any],
) -> Check:
    """Rule out a forecast far from the recent data, at each location.

    The base is the largest of the last `plausible_recent_weeks` observed
    weeks, floored at `plausible_floor`. For each week from the nowcast
    to the last horizon, the median must lie within
    `plausible_median_ratio` times the base, the 97.5% quantile within
    `plausible_q975_ratio` times it, and, where the recent maximum is at
    least the floor, the median above the maximum over
    `plausible_low_ratio`. Locations without observed data (the health
    zones) are skipped. Any breach fails, so the chain moves on.
    """
    n_recent = int(c["plausible_recent_weeks"])
    floor = float(c["plausible_floor"])
    hi, tail, lo = (
        float(c["plausible_median_ratio"]),
        float(c["plausible_q975_ratio"]),
        float(c["plausible_low_ratio"]),
    )
    weeks = [_week(ref, k) for k in range(-n_recent + 1, horizon + 1)]
    found: list[tuple[str, str]] = []
    for loc in s.locations:
        series = obs.counts.get(loc)
        if not series:
            continue
        recent = [series[w] for w in sorted(series)[-n_recent:]]
        top = max(recent, default=0.0)
        base = max(top, floor)
        for w in weeks:
            values = s.week_values(loc, w)
            if not values:
                continue
            med, q975 = quantile(values, 0.5), quantile(values, 0.975)
            if med > hi * base:
                found.append(
                    (
                        "fail",
                        f"{loc} {w}: median {fmt(med)} is over {fmt(hi)} "
                        f"times the recent maximum {fmt(base)}",
                    )
                )
            elif q975 > tail * base:
                found.append(
                    (
                        "fail",
                        f"{loc} {w}: 97.5% quantile {fmt(q975)} is over "
                        f"{fmt(tail)} times the recent maximum {fmt(base)}",
                    )
                )
            elif top >= floor and med < top / lo:
                found.append(
                    (
                        "fail",
                        f"{loc} {w}: median {fmt(med)} is under 1/"
                        f"{fmt(lo)} of the recent maximum {fmt(top)}",
                    )
                )
    return _result("plausible", found, "within range of the recent data")


def check_growth(
    s: Samples,
    ref: dt.date,
    horizon: int,
    offset: float,
    warn: float,
) -> Check:
    """Week-on-week growth of the median across the forecast weeks."""
    weeks = [_week(ref, k) for k in range(0, horizon + 1)]
    found: list[tuple[str, str]] = []
    for loc in s.locations:
        med = _medians(s, loc, weeks)
        for (w0, m0), (w1, m1) in zip(
            zip(weeks, med, strict=True),
            zip(weeks[1:], med[1:], strict=True),
            strict=False,
        ):
            if math.isnan(m0) or math.isnan(m1):
                continue
            ratio = (m1 + offset) / (m0 + offset)
            found.append(
                (
                    _level(ratio, warn),
                    f"{loc}: median grows {fmt(ratio)} times from {w0} to {w1}",
                )
            )
    return _result("growth", found, "forecast growth within the cap")


def _ratio_check(
    name: str,
    s: Samples,
    ref_values: Mapping[str, Mapping[str, float]],
    weeks_for: Any,
    offset: float,
    warn: float,
    what: str,
) -> Check:
    found: list[tuple[str, str]] = []
    for loc in s.locations:
        other = ref_values.get(loc, {})
        weeks = [w for w in weeks_for(loc) if w in other and w in s.weeks(loc)]
        if not weeks:
            continue
        ours = math.fsum(_medians(s, loc, weeks))
        theirs = math.fsum(other[w] for w in weeks)
        ratio = (ours + offset) / (theirs + offset)
        found.append(
            (
                _level(ratio, warn),
                f"{loc}: median {fmt(ours)} against {what} {fmt(theirs)} "
                f"over {weeks[0]} to {weeks[-1]} (ratio {fmt(ratio)})",
            )
        )
    if not found:
        return Check(name, "skip", f"no weeks shared with {what}")
    return _result(name, found, f"close to {what}")


def check_observed(
    s: Samples,
    obs: Observed,
    lag: int,
    n: int,
    offset: float,
    warn: float,
) -> Check:
    """Median of the last complete weeks against the observed counts."""
    cutoff = (
        dt.date.fromisoformat(obs.as_of) - dt.timedelta(days=7 * lag)
    ).isoformat()

    def weeks_for(loc: str) -> list[str]:
        done = sorted(w for w in obs.counts.get(loc, {}) if w <= cutoff)
        return done[-n:]

    return _ratio_check(
        "observed",
        s,
        obs.counts,
        weeks_for,
        offset,
        warn,
        f"observed ({obs.as_of})",
    )


def _median_table(s: Samples) -> dict[str, dict[str, float]]:
    return {
        loc: {w: quantile(s.week_values(loc, w), 0.5) for w in s.weeks(loc)}
        for loc in s.locations
    }


def check_against(
    name: str,
    s: Samples,
    other: Samples | None,
    weeks_for: Any,
    offset: float,
    warn: float,
    what: str,
) -> Check:
    """Medians against another set of samples."""
    if other is None:
        return Check(name, "skip", f"no {what} to compare with")
    return _ratio_check(
        name, s, _median_table(other), weeks_for, offset, warn, what
    )


def check_zones(s: Samples, require: bool, tol: float) -> Check:
    """Every province has zone rows, and its zones add up to it."""
    provs = [loc for loc in s.locations if is_province(loc)]
    zones: dict[str, list[str]] = {p: [] for p in provs}
    for loc in s.locations:
        parent = loc.rsplit("-", 1)[0]
        if loc.count("-") == 2 and parent in zones:
            zones[parent].append(loc)
    missing = [p for p in provs if not zones[p]]
    if not any(zones.values()):
        if require:
            return Check("zones", "fail", "no health zone rows")
        return Check("zones", "skip", "no health zone rows")
    if missing:
        return Check(
            "zones",
            "fail" if require else "warn",
            f"provinces without zone rows: {', '.join(missing)}",
        )
    off = 0
    for p, zs in zones.items():
        for sid, weeks in s.values[p].items():
            for week, value in weeks.items():
                total = sum(s.values[z].get(sid, {}).get(week, 0.0) for z in zs)
                if abs(total - value) > tol:
                    off += 1
    n = sum(len(zs) for zs in zones.values())
    if off:
        return Check(
            "zones",
            "warn",
            f"{n} zones; {off} province sample-weeks differ from the sum "
            "of their zones",
        )
    return Check("zones", "pass", f"{n} zones in {len(provs)} provinces")


def run_checks(
    submission: Path,
    *,
    reference_date: str,
    observed: Observed,
    expected_locations: Sequence[str] = (),
    tasks: Mapping[str, Any] | None = None,
    baseline: Path | None = None,
    previous: Path | None = None,
    kind: str = "main",
    settings: Mapping[str, Any],
    national_sidecar: Mapping[str, Any],
    require_zones: bool = False,
) -> list[Check]:
    """Run every check on one candidate file."""
    s = read_samples(submission)
    ref = dt.date.fromisoformat(reference_date)
    off = float(settings["ratio_offset"])
    horizon = int(settings["horizon_weeks"])
    forecast_weeks = {_week(ref, k) for k in range(1, horizon + 1)}
    out = [
        check_schema(s, reference_date, submission.name, tasks),
        check_values(s),
        check_samples(s, int(settings["samples"])),
        check_coverage(
            s, ref, expected_locations, horizon, int(settings["nowcast_weeks"])
        ),
        check_zones(s, require_zones, float(settings["sum_tolerance"])),
        check_national_sum(
            s, float(settings["sum_tolerance"]), national_sidecar
        ),
        check_bounds(
            s,
            observed,
            float(settings["bound_floor"]),
            float(settings["bound_warn"]),
        ),
        check_plausible(s, observed, ref, horizon, settings),
        check_growth(
            s,
            ref,
            horizon,
            off,
            float(settings["growth_warn"]),
        ),
        check_observed(
            s,
            observed,
            int(settings["complete_lag_weeks"]),
            int(settings["complete_weeks"]),
            off,
            float(settings["observed_warn"]),
        ),
    ]
    if kind == "randomwalk":
        out.append(Check("baseline", "skip", "this is the baseline"))
    else:
        out.append(
            check_against(
                "baseline",
                s,
                read_samples(baseline)
                if baseline and baseline.exists()
                else None,
                lambda loc: sorted(
                    w for w in s.weeks(loc) if w in forecast_weeks
                ),
                off,
                float(settings["baseline_warn"]),
                "random walk baseline",
            )
        )
    out.append(
        check_against(
            "previous",
            s,
            read_samples(previous) if previous and previous.exists() else None,
            s.weeks,
            off,
            float(settings["previous_warn"]),
            "last week's submission",
        )
    )
    return out


def markdown(cid: str, checks: Sequence[Check]) -> str:
    """The forecast checks as a markdown section."""
    status = worst(c.status for c in checks)
    lines = [
        f"## Forecast checks for `{cid}`: {STATUS_LABEL[status]}\n",
        markdown_table(
            ["check", "status", "result"],
            [[c.name, STATUS_LABEL[c.status], c.summary] for c in checks],
        ),
        "",
    ]
    detailed = [c for c in checks if c.details]
    if detailed:
        lines.append("<details><summary>Findings</summary>\n")
        for c in detailed:
            lines.append(f"**{c.name}**\n")
            lines.extend(f"- {d}" for d in c.details)
            lines.append("")
        lines.append("</details>\n")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point. Exits 1 when a check fails."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--id", required=True)
    p.add_argument("--kind", default="main")
    p.add_argument("--submission", type=Path, required=True)
    p.add_argument("--reference-date", required=True)
    p.add_argument("--observed", type=Path, required=True)
    p.add_argument("--as-of", default=None)
    p.add_argument("--locations", default="")
    p.add_argument("--tasks", type=Path, default=None)
    p.add_argument("--baseline", type=Path, default=None)
    p.add_argument("--previous", type=Path, default=None)
    p.add_argument(
        "--national-sidecar",
        type=Path,
        required=True,
        help="national.json written beside the nowcast",
    )
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--no-fail", action="store_true")
    p.add_argument(
        "--require-zones",
        action="store_true",
        help="Rule the file out if a province has no health zone rows",
    )
    args = p.parse_args(argv)
    settings = load_policy(args.policy).checks
    if not args.submission.exists():
        checks = [
            Check("schema", "error", f"{args.submission} was not written")
        ]
    else:
        checks = run_checks(
            args.submission,
            reference_date=args.reference_date,
            observed=read_observed(args.observed, args.as_of),
            expected_locations=[x for x in args.locations.split(",") if x],
            tasks=read_json(args.tasks) if args.tasks else None,
            baseline=args.baseline,
            previous=args.previous,
            kind=args.kind,
            settings=settings,
            national_sidecar=read_json(args.national_sidecar),
            require_zones=args.require_zones,
        )
    status = worst(c.status for c in checks)
    md = markdown(args.id, checks)
    write_text(args.out / "checks.md", md)
    write_json(
        args.out / "checks.json",
        {
            "id": args.id,
            "status": status,
            "checks": [asdict(c) for c in checks],
        },
    )
    append_summary(md)
    set_outputs({"status": status})
    print(f"{args.id}: forecast checks {status}")
    return 0 if args.no_fail or status in ("pass", "warn", "skip") else 1


if __name__ == "__main__":
    raise SystemExit(main())
