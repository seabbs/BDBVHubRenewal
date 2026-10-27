"""Gate a fit on its posterior predictive checks (`ppc.json`).

The fit writes `ppc.json` beside `diagnostics.json` (see INTEGRATION.md):
one row per location, stream and in-sample onset week with the observed
count, the predictive median, central intervals and the randomised PIT.
This check reports, for the gated stream (`total` by default) and each
other stream:

- the share of observed counts inside the 50% and 90% intervals,
  overall, by province, and over the last complete weeks;
- a PIT summary: the Kolmogorov-Smirnov distance from uniform, the
  share in the outer deciles (0.2 when calibrated; more means the
  intervals are too narrow, less too wide) and the mean PIT (below 0.5
  means the model predicts too high, above too low).

Thresholds are in `policy.toml` under `[ppc]`, each overridable by an
environment variable `BDBV_PPC_<KEY>` (for example
`BDBV_PPC_COVERAGE_WARN`):

- `fail` when the coverage of the `level` interval (0.9), overall or
  over the last `recent_weeks` complete weeks, is below `coverage_fail`;
- `warn` when it is below `coverage_warn`, or when the PIT KS distance
  is above `pit_ks_warn` with at least `pit_min_rows` rows;
- `n/a` (`skip`) when the model has no posterior (`available: false`,
  or no sampler and no file);
- `error` when a sampled fit has no readable `ppc.json`, which gates
  like a failure, as a missing diagnostics file does.
"""

from __future__ import annotations

import argparse
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
    write_json,
    write_text,
)
from .summarise_fit import _env_number

DEFAULTS: dict[str, Any] = {
    "stream": "total",
    "level": 0.9,
    "coverage_warn": 0.70,
    "coverage_fail": 0.50,
    "recent_weeks": 4,
    "pit_ks_warn": 0.2,
    "pit_min_rows": 20,
}
STREAMS = ("total", "observed", "imputed")
# Descriptive cut-offs for the PIT shape. Reported, never gated.
OUTER_NARROW = 0.3  # share in the outer deciles above which: too narrow
OUTER_WIDE = 0.1  # and below which: too wide
MEAN_BIAS = 0.15  # mean PIT further than this from 0.5: biased


def limits(base: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Thresholds from `base` (or the defaults) with env overrides."""
    out = {**DEFAULTS, **(base or {})}
    for key, value in out.items():
        if isinstance(value, int | float):
            out[key] = _env_number(f"BDBV_PPC_{key.upper()}", float(value))
    out["recent_weeks"] = int(out["recent_weeks"])
    out["pit_min_rows"] = int(out["pit_min_rows"])
    return out


def _pct(level: float) -> int:
    return round(100 * float(level))


def _num(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


def coverage(rows: Sequence[Mapping[str, Any]], pct: int) -> float:
    """Share of observed counts inside the `pct`% interval."""
    hits = []
    for r in rows:
        y = _num(r.get("observed"))
        lo, hi = _num(r.get(f"lower_{pct}")), _num(r.get(f"upper_{pct}"))
        if not any(math.isnan(v) for v in (y, lo, hi)):
            hits.append(lo <= y <= hi)
    return sum(hits) / len(hits) if hits else math.nan


def pit_summary(pits: Sequence[Any]) -> dict[str, Any]:
    """KS distance from uniform, outer-decile share, mean and shape."""
    u = sorted(p for p in (_num(x) for x in pits) if not math.isnan(p))
    n = len(u)
    if not n:
        return {"n": 0, "ks": None, "outer": None, "mean": None, "shape": ""}
    ks = max(max((i + 1) / n - x, x - i / n) for i, x in enumerate(u))
    outer = sum(x < 0.1 or x > 0.9 for x in u) / n
    mean = sum(u) / n
    shape = []
    if outer > OUTER_NARROW:
        shape.append("intervals look too narrow (PIT piled at 0 and 1)")
    elif outer < OUTER_WIDE:
        shape.append("intervals look too wide (PIT bunched in the middle)")
    if mean < 0.5 - MEAN_BIAS:
        shape.append("predictions look too high (PIT mostly low)")
    elif mean > 0.5 + MEAN_BIAS:
        shape.append("predictions look too low (PIT mostly high)")
    return {
        "n": n,
        "ks": ks,
        "outer": outer,
        "mean": mean,
        "shape": "; ".join(shape),
    }


@dataclass
class PPCCheck:
    """The verdict on one fit's posterior predictive checks."""

    id: str
    status: str
    available: bool
    reason: str = ""
    model: str = ""
    config: str = ""
    stream: str = "total"
    levels: list[int] = field(default_factory=list)
    recent_weeks: list[str] = field(default_factory=list)
    # stream -> {n, cov<pct>, recent_n, recent_cov<pct>}
    streams: dict[str, dict[str, float]] = field(default_factory=dict)
    # gated stream, one row per location
    locations: list[dict[str, Any]] = field(default_factory=list)
    pit: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _metrics(
    rows: Sequence[Mapping[str, Any]], recent: set[str], levels: list[int]
) -> dict[str, float]:
    last = [r for r in rows if r["complete"] and r["target_end_date"] in recent]
    out: dict[str, float] = {"n": len(rows), "recent_n": len(last)}
    for pct in levels:
        out[f"cov{pct}"] = coverage(rows, pct)
        out[f"recent_cov{pct}"] = coverage(last, pct)
    return out


def check_ppc(
    fit_id: str,
    ppc: Mapping[str, Any] | None,
    lim: Mapping[str, Any],
    *,
    sampled: bool = True,
    error: str = "",
) -> PPCCheck:
    """Gate one fit's `ppc.json` payload."""
    if ppc is None or error:
        if not sampled and not error:
            return PPCCheck(fit_id, "skip", False, "the model has no sampler")
        why = error or "no ppc.json was written"
        return PPCCheck(
            fit_id,
            "error",
            False,
            why,
            failures=[f"{why}, so the fit's predictions cannot be checked"],
        )
    if not ppc.get("available"):
        return PPCCheck(
            fit_id,
            "skip",
            False,
            str(ppc.get("reason", "not available")),
            model=str(ppc.get("model", "")),
            config=str(ppc.get("config", "")),
        )
    rows = [
        {**r, "complete": bool(r.get("complete"))}
        for r in ppc.get("rows") or []
    ]
    levels = sorted({_pct(x) for x in ppc.get("intervals") or (0.5, 0.9)})
    present = [s for s in STREAMS if any(r["stream"] == s for r in rows)]
    present += sorted({str(r["stream"]) for r in rows} - set(present))
    stream = str(lim["stream"])
    gated = (
        [r for r in rows if r["stream"] == stream]
        if stream in present
        else rows
    )
    if stream not in present:
        stream = "all streams"
    done = sorted({r["target_end_date"] for r in gated if r["complete"]})
    recent = done[-int(lim["recent_weeks"]) :]
    out = PPCCheck(
        fit_id,
        "pass",
        True,
        model=str(ppc.get("model", "")),
        config=str(ppc.get("config", "")),
        stream=stream,
        levels=levels,
        recent_weeks=recent,
    )
    rset = set(recent)
    out.streams = {
        s: _metrics([r for r in rows if r["stream"] == s], rset, levels)
        for s in present
    }
    for loc in sorted({str(r["location"]) for r in gated}):
        lrows = [r for r in gated if r["location"] == loc]
        out.locations.append({"location": loc, **_metrics(lrows, rset, levels)})
    out.pit = pit_summary([r.get("pit") for r in gated])
    if not rows:
        out.status = "error"
        out.failures.append("ppc.json is available but has no rows")
        return out
    pct = _pct(lim["level"])
    head = _metrics(gated, rset, [*levels, pct])
    for key, where in (
        (f"cov{pct}", "overall"),
        (f"recent_cov{pct}", (f"over the last {len(recent)} complete weeks")),
    ):
        value = head[key]
        if math.isnan(value):
            continue
        msg = f"{pct}% interval coverage of {stream} {where} is {fmt(value)}"
        if value < lim["coverage_fail"]:
            out.failures.append(
                f"{msg}, below the failure threshold "
                f"{fmt(lim['coverage_fail'])}"
            )
        elif value < lim["coverage_warn"]:
            out.warnings.append(
                f"{msg}, below the warning threshold "
                f"{fmt(lim['coverage_warn'])}"
            )
    ks = out.pit.get("ks")
    if (
        ks is not None
        and out.pit["n"] >= lim["pit_min_rows"]
        and ks > lim["pit_ks_warn"]
    ):
        shape = f": {out.pit['shape']}" if out.pit["shape"] else ""
        out.warnings.append(
            f"PIT of {stream} is {fmt(ks)} from uniform (KS distance), "
            f"above {fmt(lim['pit_ks_warn'])}{shape}"
        )
    out.status = (
        "fail" if out.failures else ("warn" if out.warnings else "pass")
    )
    return out


def headline(c: PPCCheck, level: int = 90) -> str:
    """Coverage of the gated stream in one line, for tables."""
    s = c.streams.get(c.stream)
    if not c.available or not s:
        return ""
    parts = []
    for pct in sorted({50, level} & set(c.levels), reverse=True):
        parts.append(
            f"{pct}% coverage {fmt(s.get(f'cov{pct}'), 2)} overall, "
            f"{fmt(s.get(f'recent_cov{pct}'), 2)} last "
            f"{len(c.recent_weeks)} weeks"
        )
    return "; ".join(parts)


_PHRASE = {
    "pass": "predictive intervals cover the data as intended",
    "warn": "predictive intervals cover the data less well than intended",
    "fail": "predictive intervals miss too much of the data",
    "error": "could not be checked",
    "skip": "no posterior, so posterior predictive checks do not apply",
}


def markdown(c: PPCCheck, lim: Mapping[str, Any]) -> str:
    """The posterior predictive check report for one fit."""
    lines = [
        f"## Posterior predictive checks for `{c.id}`: "
        f"{STATUS_LABEL[c.status]}\n",
        f"`{c.id}`: {_PHRASE[c.status]}"
        + (f" ({c.reason})" if c.reason else "")
        + ".\n",
    ]
    if c.available and c.streams:
        cols = [f"{p}%" for p in c.levels]
        lines.append(
            markdown_table(
                ["stream", "rows"]
                + [f"{x} coverage" for x in cols]
                + [
                    f"{x}, last {len(c.recent_weeks)} complete weeks"
                    for x in cols
                ],
                [
                    [
                        f"**{s}**" if s == c.stream else s,
                        fmt(m["n"]),
                        *(fmt(m[f"cov{p}"], 2) for p in c.levels),
                        *(fmt(m[f"recent_cov{p}"], 2) for p in c.levels),
                    ]
                    for s, m in c.streams.items()
                ],
            )
        )
        lines.append("")
        p = c.pit
        if p.get("n"):
            lines.append(
                f"PIT of {c.stream} ({p['n']} rows, randomised): KS distance "
                f"{fmt(p['ks'], 2)}, {fmt(p['outer'], 2)} in the outer "
                f"deciles (0.2 if calibrated), mean {fmt(p['mean'], 2)}"
                + (f"; {p['shape']}" if p["shape"] else "")
                + ".\n"
            )
    lines.extend(f"- **fail**: {m}" for m in c.failures)
    lines.extend(f"- warn: {m}" for m in c.warnings)
    if c.failures or c.warnings:
        lines.append("")
    if c.locations:
        lines.append(
            f"<details><summary>Coverage of {c.stream} by province</summary>\n"
        )
        lines.append(
            markdown_table(
                ["province", "weeks"]
                + [f"{p}%" for p in c.levels]
                + [f"{p}%, last {len(c.recent_weeks)} weeks" for p in c.levels],
                [
                    [
                        str(r["location"]),
                        fmt(r["n"]),
                        *(fmt(r[f"cov{p}"], 2) for p in c.levels),
                        *(fmt(r[f"recent_cov{p}"], 2) for p in c.levels),
                    ]
                    for r in c.locations
                ],
            )
        )
        lines.append("\n</details>\n")
    if c.available:
        pct = _pct(lim["level"])
        lines.append(
            f"Thresholds. {pct}% coverage of {c.stream}, overall and over "
            f"the last {lim['recent_weeks']} complete weeks: fail below "
            f"{fmt(lim['coverage_fail'])}, warn below "
            f"{fmt(lim['coverage_warn'])}. PIT KS distance: warn above "
            f"{fmt(lim['pit_ks_warn'])} (with at least "
            f"{lim['pit_min_rows']} rows)."
        )
    return "\n".join(lines) + "\n"


def _clean(x: Any) -> Any:
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return x


def summarise(
    fit_id: str,
    ppc_path: Path,
    out_dir: Path,
    *,
    sampled: bool,
    base: Mapping[str, Any] | None = None,
) -> PPCCheck:
    """Gate one fit and write `ppc-summary.{md,json}` to `out_dir`."""
    lim = limits(base)
    ppc, error = None, ""
    if ppc_path.exists():
        try:
            ppc = read_json(ppc_path)
            if not isinstance(ppc, dict) or "available" not in ppc:
                ppc, error = None, "ppc.json has no 'available' field"
        except ValueError as err:
            error = f"ppc.json is unreadable ({err})"
    check = check_ppc(fit_id, ppc, lim, sampled=sampled, error=error)
    write_text(out_dir / "ppc-summary.md", markdown(check, lim))
    data = asdict(check)
    data["thresholds"] = lim
    data["headline"] = headline(check, _pct(lim["level"]))
    write_json(out_dir / "ppc-summary.json", _clean(data))
    return check


def is_sampled(kind: str, diagnostics: Path | None) -> bool:
    """Whether the candidate's fit used a sampler.

    Persistence and fits whose diagnostics say `sampler: none` did not.
    Any other fit is taken to be sampled, so a fit that wrote nothing
    cannot pass by omission.
    """
    if kind == "persistence":
        return False
    if diagnostics is None or not diagnostics.exists():
        return True
    try:
        diag = read_json(diagnostics)
    except ValueError:
        return True
    return not (isinstance(diag, dict) and diag.get("sampler") == "none")


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point. Exits 1 on fail and 2 on error."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--id", required=True)
    p.add_argument("--kind", default="main")
    p.add_argument("--ppc", type=Path, required=True)
    p.add_argument("--diagnostics", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--policy", type=Path, default=None)
    p.add_argument("--no-fail", action="store_true")
    args = p.parse_args(argv)
    base = None
    if args.policy:
        from .policy import load_policy

        base = load_policy(args.policy).ppc
    check = summarise(
        args.id,
        args.ppc,
        args.out,
        sampled=is_sampled(args.kind, args.diagnostics),
        base=base,
    )
    append_summary((args.out / "ppc-summary.md").read_text())
    set_outputs({"status": check.status, "available": check.available})
    print(f"{args.id}: posterior predictive checks {check.status}")
    if args.no_fail:
        return 0
    return {"fail": 1, "error": 2}.get(check.status, 0)


if __name__ == "__main__":
    raise SystemExit(main())
