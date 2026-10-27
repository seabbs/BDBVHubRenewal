"""Summarise a fit's convergence diagnostics and gate on them.

A port of BVDOutbreakSize's convergence gate (`docs/fits/convergence.jl`
and `check_convergence.jl`) to the `diagnostics.json` the main repo
writes (see INTEGRATION.md):

- two thresholds per diagnostic, *fail* (the sampler has not explored
  the posterior) and *warn* (converged but mixing poorly), with the same
  defaults and the same environment overrides
  (`BDBV_CONVERGENCE_FAIL_RHAT` and so on), which must be finite;
- a diagnostic that cannot be computed is reported as `n/a` and neither
  fails nor warns;
- a missing or malformed diagnostics file is an `error`, which gates
  like a failure: an unjudged fit is not a passing one;
- the report gives the headline table, the per-chain sampler behaviour,
  the reasons for each failure or warning, and the worst-mixing
  parameters, singly and grouped.
"""

from __future__ import annotations

import argparse
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .common import (
    STATUS_LABEL,
    append_summary,
    fmt,
    fmt_rhat,
    markdown_table,
    read_json,
    set_outputs,
    write_json,
    write_text,
)

MARKER = "<!-- bdbv-fit-convergence -->"
DIAGNOSTICS = ("rhat", "ess_bulk", "ess_tail", "divergent_fraction")
DEFAULT_THRESHOLDS: dict[str, dict[str, float]] = {
    "fail": {
        "rhat": 1.1,
        "ess_bulk": 25.0,
        "ess_tail": 25.0,
        "divergent_fraction": 0.05,
    },
    "warn": {
        "rhat": 1.05,
        "ess_bulk": 100.0,
        "ess_tail": 100.0,
        "divergent_fraction": 0.01,
    },
}
_ENV = {
    "rhat": "RHAT",
    "ess_bulk": "ESS_BULK",
    "ess_tail": "ESS_TAIL",
    "divergent_fraction": "DIVERGENT",
}


def _env_number(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        n = float(value)
    except ValueError:
        raise ValueError(
            f'{name} is "{value}", which is not a number'
        ) from None
    if not math.isfinite(n):
        raise ValueError(f'{name} is "{value}"; a threshold has to be finite')
    return n


def thresholds(
    base: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, dict[str, float]]:
    """Thresholds from `base` (or the defaults) with env overrides."""
    out: dict[str, dict[str, float]] = {}
    for level in ("fail", "warn"):
        src = dict(DEFAULT_THRESHOLDS[level])
        if base and level in base:
            src.update({k: float(v) for k, v in base[level].items()})
        out[level] = {
            k: _env_number(
                f"BDBV_CONVERGENCE_{level.upper()}_{_ENV[k]}", src[k]
            )
            for k in DIAGNOSTICS
        }
    return out


def _finite(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


@dataclass
class FitCheck:
    """The verdict on one fit."""

    id: str
    status: str
    applicable: bool
    headline: dict[str, float] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    chains: list[dict[str, Any]] = field(default_factory=list)
    worst: list[dict[str, Any]] = field(default_factory=list)
    families: list[dict[str, Any]] = field(default_factory=list)


def headline(diag: Mapping[str, Any]) -> dict[str, float]:
    """Worst R-hat, smallest ESS, divergences and draws."""
    params = diag.get("parameters", [])
    rhats = [r for p in params if not math.isnan(r := _finite(p.get("rhat")))]
    bulk = [
        e for p in params if not math.isnan(e := _finite(p.get("ess_bulk")))
    ]
    tail = [
        e for p in params if not math.isnan(e := _finite(p.get("ess_tail")))
    ]
    chains = diag.get("chains", [])
    div = sum(int(c.get("divergences", 0)) for c in chains)
    draws = sum(int(c.get("draws", 0)) for c in chains) or int(
        diag.get("num_chains", 0)
    ) * int(diag.get("num_samples", 0))
    return {
        "max_rhat": max(rhats) if rhats else math.nan,
        "min_ess_bulk": min(bulk) if bulk else math.nan,
        "min_ess_tail": min(tail) if tail else math.nan,
        "n_divergent": float(div),
        "n_draws": float(draws),
        "divergent_fraction": div / draws if draws else math.nan,
    }


def verdict(
    head: Mapping[str, float], limits: Mapping[str, Mapping[str, float]]
) -> tuple[str, list[str], list[str]]:
    """Status and messages for the headline diagnostics."""
    checks = (
        ("max R-hat", head["max_rhat"], "rhat", True),
        ("min bulk ESS", head["min_ess_bulk"], "ess_bulk", False),
        ("min tail ESS", head["min_ess_tail"], "ess_tail", False),
        (
            "divergent fraction",
            head["divergent_fraction"],
            "divergent_fraction",
            True,
        ),
    )
    failures: list[str] = []
    warnings: list[str] = []
    for name, value, key, higher_is_worse in checks:
        if math.isnan(value):
            continue

        def breaches(
            limit: float, v: float = value, hi: bool = higher_is_worse
        ) -> bool:
            return v > limit if hi else v < limit

        if breaches(limits["fail"][key]):
            failures.append(
                f"{name} is {fmt(value)}, past the failure threshold "
                f"{fmt(limits['fail'][key])}"
            )
        elif breaches(limits["warn"][key]):
            warnings.append(
                f"{name} is {fmt(value)}, past the warning threshold "
                f"{fmt(limits['warn'][key])}"
            )
    status = "fail" if failures else ("warn" if warnings else "pass")
    return status, failures, warnings


def _label(p: Mapping[str, Any]) -> str:
    idx = p.get("index")
    return f"{p['name']}[{idx}]" if idx not in (None, "", 0) else str(p["name"])


def worst_parameters(
    diag: Mapping[str, Any], n: int = 10
) -> list[dict[str, Any]]:
    """The `n` worst-mixing parameter elements, by bulk ESS."""
    params = [
        p
        for p in diag.get("parameters", [])
        if not math.isnan(_finite(p.get("rhat")))
    ]
    ranked = sorted(params, key=lambda p: _finite(p.get("ess_bulk")))
    return [
        {
            "parameter": _label(p),
            "rhat": _finite(p.get("rhat")),
            "ess_bulk": _finite(p.get("ess_bulk")),
            "ess_tail": _finite(p.get("ess_tail")),
        }
        for p in ranked[:n]
    ]


def families(
    diag: Mapping[str, Any], rhat_threshold: float, n: int = 10
) -> list[dict[str, Any]]:
    """Diagnostics grouped by parameter name, worst bulk ESS first."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for p in diag.get("parameters", []):
        if not math.isnan(_finite(p.get("rhat"))):
            groups.setdefault(str(p["name"]), []).append(p)
    rows: list[dict[str, Any]] = []
    for name, ps in groups.items():
        rh = [_finite(p.get("rhat")) for p in ps]
        eb = [e for p in ps if not math.isnan(e := _finite(p.get("ess_bulk")))]
        rows.append(
            {
                "parameter": name,
                "elements": len(ps),
                "max_rhat": max(rh),
                "min_ess_bulk": min(eb) if eb else math.nan,
                "above": sum(r > rhat_threshold for r in rh),
            }
        )
    rows.sort(key=lambda r: float(r["min_ess_bulk"]))
    return rows[:n]


def check_fit(
    fit_id: str,
    diag: Mapping[str, Any] | None,
    limits: Mapping[str, Mapping[str, float]],
) -> FitCheck:
    """Gate one fit's diagnostics."""
    if diag is None:
        return FitCheck(
            fit_id,
            "error",
            True,
            failures=["no diagnostics file, so the fit cannot be judged"],
        )
    if diag.get("sampler", "nuts") == "none":
        return FitCheck(fit_id, "skip", False)
    head = headline(diag)
    status, failures, warnings = verdict(head, limits)
    out = FitCheck(
        fit_id,
        status,
        True,
        headline=head,
        failures=failures,
        warnings=warnings,
        chains=list(diag.get("chains", [])),
    )
    if status != "pass":
        out.worst = worst_parameters(diag)
        out.families = families(diag, limits["warn"]["rhat"])
    return out


_PHRASE = {
    "pass": "cleared the convergence thresholds",
    "warn": "converged, with diagnostics past the warning thresholds",
    "fail": "has not converged",
    "error": "could not be judged",
    "skip": "has no sampler, so convergence does not apply",
}


def markdown(
    checks: Sequence[FitCheck],
    limits: Mapping[str, Mapping[str, float]],
    *,
    marker: str = MARKER,
    context: str = "",
) -> str:
    """The convergence report for one or more fits."""
    lines = [marker] if marker else []
    lines.append("## Fit convergence\n")
    if context:
        lines.append(context + "\n")
    rows = []
    for c in checks:
        h = c.headline
        rows.append(
            [
                f"`{c.id}`",
                STATUS_LABEL[c.status],
                fmt_rhat(h.get("max_rhat")),
                fmt(h.get("min_ess_bulk")),
                fmt(h.get("min_ess_tail")),
                fmt(h.get("n_divergent")),
                fmt(h.get("n_draws")),
            ]
        )
    lines.append(
        markdown_table(
            [
                "fit",
                "status",
                "max R-hat",
                "min ESS bulk",
                "min ESS tail",
                "divergences",
                "draws",
            ],
            rows,
        )
    )
    lines.append("")
    for c in checks:
        lines.append(f"- `{c.id}` {_PHRASE[c.status]}.")
    lines.append("")
    if any(c.chains for c in checks):
        lines.append("<details><summary>Sampler behaviour by chain</summary>\n")
        for c in checks:
            if not c.chains:
                continue
            lines.append(f"`{c.id}`:\n")
            lines.append(
                markdown_table(
                    [
                        "chain",
                        "divergences",
                        "step size",
                        "at max tree depth",
                        "mean acceptance",
                        "E-BFMI",
                    ],
                    [
                        [
                            str(ch.get("chain", "")),
                            str(ch.get("divergences", 0)),
                            fmt(_finite(ch.get("step_size"))),
                            str(ch.get("tree_depth_hits", "n/a")),
                            fmt(_finite(ch.get("mean_accept"))),
                            fmt(_finite(ch.get("ebfmi"))),
                        ]
                        for ch in c.chains
                    ],
                )
            )
            lines.append("")
        lines.append("</details>\n")
    for c in checks:
        if not (c.failures or c.warnings):
            continue
        lines.append(f"### `{c.id}`: {STATUS_LABEL[c.status]}\n")
        lines.extend(f"- **fail**: {m}" for m in c.failures)
        lines.extend(f"- warn: {m}" for m in c.warnings)
        lines.append("")
        if c.worst:
            lines.append(
                f"<details><summary>Worst-mixing parameters of `{c.id}`"
                "</summary>\n"
            )
            lines.append(
                markdown_table(
                    ["parameter", "R-hat", "ESS bulk", "ESS tail"],
                    [
                        [
                            str(w["parameter"]),
                            fmt_rhat(w["rhat"]),
                            fmt(w["ess_bulk"]),
                            fmt(w["ess_tail"]),
                        ]
                        for w in c.worst
                    ],
                )
            )
            lines.append("\nGrouped, one row per parameter:\n")
            lines.append(
                markdown_table(
                    [
                        "parameter",
                        "elements",
                        "max R-hat",
                        "min ESS bulk",
                        f"above {fmt(limits['warn']['rhat'])}",
                    ],
                    [
                        [
                            str(f["parameter"]),
                            str(f["elements"]),
                            fmt_rhat(f["max_rhat"]),
                            fmt(f["min_ess_bulk"]),
                            str(f["above"]),
                        ]
                        for f in c.families
                    ],
                )
            )
            lines.append("\n</details>\n")
    f, w = limits["fail"], limits["warn"]
    lines.append(
        f"Thresholds. Fail: R-hat above {fmt(f['rhat'])}, bulk ESS below "
        f"{fmt(f['ess_bulk'])}, tail ESS below {fmt(f['ess_tail'])}, "
        f"divergences above {fmt(f['divergent_fraction'])} of draws. "
        f"Warn: {fmt(w['rhat'])}, {fmt(w['ess_bulk'])}, "
        f"{fmt(w['ess_tail'])}, {fmt(w['divergent_fraction'])}."
    )
    return "\n".join(lines) + "\n"


def summarise(
    fit_id: str,
    diagnostics: Path,
    out_dir: Path,
    base: Mapping[str, Mapping[str, float]] | None = None,
) -> FitCheck:
    """Gate one fit and write `fit-summary.{md,json}` to `out_dir`."""
    limits = thresholds(base)
    try:
        diag = read_json(diagnostics) if diagnostics.exists() else None
    except ValueError:
        diag = None
    check = check_fit(fit_id, diag, limits)
    write_text(out_dir / "fit-summary.md", markdown([check], limits))
    data = asdict(check)
    data["thresholds"] = limits
    data["headline"] = {
        k: (None if isinstance(v, float) and math.isnan(v) else v)
        for k, v in check.headline.items()
    }
    write_json(out_dir / "fit-summary.json", _clean(data))
    return check


def _clean(x: Any) -> Any:
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return x


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point. Exits 1 on fail and 2 on error."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--id", required=True)
    p.add_argument("--diagnostics", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--policy", type=Path, default=None)
    p.add_argument("--no-fail", action="store_true")
    args = p.parse_args(argv)
    base = None
    if args.policy:
        from .policy import load_policy

        base = load_policy(args.policy).convergence
    check = summarise(args.id, args.diagnostics, args.out, base)
    append_summary((args.out / "fit-summary.md").read_text())
    set_outputs({"status": check.status})
    print(f"{args.id}: {check.status}")
    if args.no_fail:
        return 0
    return {"fail": 1, "error": 2}.get(check.status, 0)


if __name__ == "__main__":
    raise SystemExit(main())
