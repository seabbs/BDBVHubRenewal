"""Stand-in for the main repo's `bdbv-hub-renewal` CLI, for the dry run.

It implements the interface in INTEGRATION.md on fixture data, so the
step scripts run end to end without the model. Behaviour per candidate
is set by `FAKE_SCENARIOS`, e.g. `main=fail,retry-1=pass`, read with the
candidate id the step script exports as `CAND_ID`. Scenarios:

  pass         converged fit, plausible forecast
  warn         converged but past the warning thresholds
  loose        past the retry thresholds (R-hat 1.2), not broken
  fail         clearly broken (bulk ESS 8, R-hat 1.4, divergences)
  implausible  converged, but the forecast explodes
  crash        the fit command fails
  negative     the nowcast writes a negative value (format check)
  ppcwarn      converged, but the predictive intervals cover about 60%
               of the observations (PPC warns)
  ppcfail      converged, but they cover about 30% (PPC fails)
  noppc        converged, but the fit writes no ppc.json (PPC error)
  standin      the config runs a stand-in (persistence): no sampler, no
               posterior, and `stand_in` in diagnostics.json and fit.json

Unknown candidates default to `pass`. `persistence=implausible` breaks
the persistence baseline too, which is how the dry run reaches a skip.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import random
import re
import struct
import sys
import zlib
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

HERE = Path(__file__).parent
OBSERVED = Path(
    os.environ.get("FAKE_OBSERVED", HERE / "hub" / "time-series.csv")
)
TARGET = "wk inc confirmed cases"
BACK_WEEKS = 7  # weeks submitted before the reference date
HORIZON = 3


def scenario() -> str:
    table = dict(
        item.split("=", 1)
        for item in os.environ.get("FAKE_SCENARIOS", "").split(",")
        if "=" in item
    )
    return table.get(os.environ.get("CAND_ID", ""), "pass")


def observed(as_of: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = defaultdict(dict)
    with OBSERVED.open() as fh:
        for row in csv.DictReader(fh):
            if row["as_of"] == as_of and row["target"] == TARGET:
                out[row["location"]][row["target_end_date"]] = float(
                    row["observation"]
                )
    return out


def reference_date(as_of: str) -> dt.date:
    """First Saturday whose window (closing Tuesday) is after `as_of`."""
    day = dt.date.fromisoformat(as_of)
    ref = day + dt.timedelta(days=(5 - day.weekday()) % 7)
    while ref - dt.timedelta(days=4) < day:
        ref += dt.timedelta(days=7)
    return ref


def fake_ppc(as_of: str, sc: str, config: str) -> dict:
    """A `ppc.json` whose PIT values, and so coverage, the scenario sets.

    Rows are ordered by week then province and row `i` is given a PIT.
    For `pass` the PITs are a low-discrepancy sequence, close to uniform.
    For `ppcwarn` (`ppcfail`) 4 (7) rows in 10 sit in the outer 1% and
    the rest inside the 90% interval. The median and intervals are then
    placed around the observed count to match: a normal predictive
    distribution with a scale of 15% of the count (at least 2).
    """
    obs = observed(as_of)
    cutoff = (dt.date.fromisoformat(as_of) - dt.timedelta(weeks=4)).isoformat()
    tails = {"ppcwarn": 4, "ppcfail": 7}.get(sc, 0)
    z = NormalDist().inv_cdf
    rows = []
    for stream in ("total", "observed"):
        keys = [
            (w, loc)
            for w in sorted({w for loc in obs for w in obs[loc]})
            for loc in sorted(obs)
            if loc != "CD" and w in obs[loc]
        ]
        for i, (week, loc) in enumerate(keys):
            golden = (0.5 + i * 0.6180339887) % 1
            if i % 10 < tails:
                pit = 0.005 if i % 2 else 0.995
            elif tails:
                pit = 0.1 + 0.8 * golden
            else:
                pit = min(max(golden, 0.005), 0.995)
            y = obs[loc][week]
            scale = max(2.0, 0.15 * y)
            med = y - z(pit) * scale
            row = {
                "location": loc,
                "stream": stream,
                "target_end_date": week,
                "observed": round(y),
                "median": med,
            }
            for level in (50, 90):
                half = z(0.5 + level / 200) * scale
                row[f"lower_{level}"] = med - half
                row[f"upper_{level}"] = med + half
            row["pit"] = pit
            row["complete"] = week <= cutoff
            rows.append(row)
    return {
        "schema_version": 1,
        "available": True,
        "model": "renewal",
        "config": config,
        "intervals": [0.5, 0.9],
        "rows": rows,
    }


def cmd_fit(args: argparse.Namespace) -> int:
    sc = scenario()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "progress.log").write_text("iteration=2000 divergences=0\n")
    if sc == "crash":
        print("fake fit: crashed", file=sys.stderr)
        return 1
    if sc == "standin":
        text = (
            f"the `{args.config}` config runs the persistence baseline in "
            "place of the model"
        )
        diag = {"schema_version": 1, "sampler": "none", "stand_in": text}
        ppc = {
            "schema_version": 1,
            "available": False,
            "reason": "the stand-in has no posterior",
        }
        for name, data in (
            ("diagnostics.json", diag),
            ("fit.json", {"stand_in": text}),
            ("ppc.json", ppc),
        ):
            (out / name).write_text(json.dumps(data))
        return 0
    rng = random.Random(1)
    sampler = json.loads(args.sampler_json or "{}")
    chains = int(sampler.get("num_chains", 4))
    # As NumPyro: with fewer XLA host devices than chains, the chains run
    # one after another, which the CI must never do. JAX on CPU has one
    # device unless XLA_FLAGS says otherwise.
    flag = re.search(
        r"--xla_force_host_platform_device_count=(\d+)",
        os.environ.get("XLA_FLAGS", ""),
    )
    devices = int(flag.group(1)) if flag else 1
    if devices < chains:
        print(
            f"fake fit: {chains} chains on {devices} XLA devices would run "
            "sequentially",
            file=sys.stderr,
        )
        return 1
    draws = int(sampler.get("num_samples", 1000))
    rhat, ess, div = {
        "warn": (1.07, 80.0, 15),
        # Past the retry thresholds (refit once), not clearly broken.
        "loose": (1.2, 40.0, 80),
        # Clearly broken: almost no effective draws.
        "fail": (1.4, 8.0, 400),
    }.get(sc, (1.004, 900.0, 0))
    params = [
        {
            "name": name,
            "index": i,
            "rhat": rhat - 0.002 * rng.random() * (i + 1),
            "ess_bulk": ess + 50 * i + 10 * rng.random(),
            "ess_tail": ess + 60 * i,
        }
        for name in ("log_rt_national", "log_rt_dev", "pi_missing")
        for i in range(3)
    ]
    diag = {
        "schema_version": 1,
        "model": "renewal",
        "config": args.config,
        "sampler": "nuts",
        "num_chains": chains,
        "num_samples": draws,
        "max_tree_depth": int(sampler.get("max_tree_depth", 10)),
        "runtime_seconds": 12.5,
        "parameters": params,
        "chains": [
            {
                "chain": c,
                "draws": draws,
                "divergences": div // chains,
                "step_size": 0.05,
                "tree_depth_hits": 0,
                "mean_accept": 0.85,
                "ebfmi": 0.9,
            }
            for c in range(chains)
        ],
    }
    (out / "diagnostics.json").write_text(json.dumps(diag, indent=2))
    if sc != "noppc":
        ppc = fake_ppc(args.vintage, sc, args.config)
        (out / "ppc.json").write_text(json.dumps(ppc, indent=1))
    return 0


def cmd_nowcast(args: argparse.Namespace) -> int:
    sc = scenario()
    as_of = args.vintage
    obs = observed(as_of)
    ref = reference_date(as_of)
    weeks = [
        (ref + dt.timedelta(days=7 * k)).isoformat()
        for k in range(-BACK_WEEKS, HORIZON + 1)
    ]
    provinces = sorted(loc for loc in obs if loc != "CD")
    rng = random.Random(7 if args.model == "persistence" else 11)
    rows = []
    totals: dict[tuple[int, str], float] = defaultdict(float)
    for loc in provinces:
        complete = sorted(obs[loc])[:-4]
        base = sum(obs[loc][w] for w in complete[-2:]) / 2
        for s in range(1, 101):
            noise = math.exp(rng.gauss(0, 0.15))
            for k, w in enumerate(weeks):
                growth = 1.0
                if sc == "implausible" and w > ref.isoformat():
                    growth = 10.0 ** (k - BACK_WEEKS)
                v = round(base * noise * growth)
                if sc == "negative" and s == 1 and k == 0:
                    v = -1
                totals[(s, w)] += v
                rows.append([loc, s, w, v])
    for (s, w), v in totals.items():
        # `badnational`: the national rows do not add up, and there is no
        # national.json, so the gate rebuilds them.
        rows.append(["CD", s, w, v + (3 if sc == "badnational" else 0)])
    mid = f"epiforecasts-{args.model}"
    path = Path(args.out) / "model-output" / mid / f"{ref}-{mid}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "reference_date",
                "target",
                "target_end_date",
                "location",
                "output_type",
                "output_type_id",
                "value",
            ]
        )
        for loc, s, week, v in rows:
            w.writerow([ref.isoformat(), TARGET, week, loc, "sample", s, v])
    sidecar = {
        "schema_version": 1,
        "national": "CD",
        "modelled_provinces": provinces,
        "unassigned": [],
    }
    if sc != "badnational":
        (Path(args.out) / "national.json").write_text(json.dumps(sidecar))
    if sc == "latestream":
        (Path(args.out) / "handled.json").write_text(
            json.dumps(
                [
                    "delays: no delay data for this vintage, so used the "
                    "2026-10-24 vintage, the latest available"
                ]
            )
        )
    print(f"Wrote {path}")
    return 0


def cmd_coherence(args: argparse.Namespace) -> int:
    """National against the sum of provinces, per stream and week.

    The fixture has confirmed onsets only, so `observed` is the fixture,
    `imputed` is zero and `total` is their sum. Its national row is the
    sum of the provinces; `FAKE_COHERENCE_GAP=<week>:<gap>[,...]` adds
    `gap` to the national observed (and so total) count of a week.
    """
    obs = observed(args.vintage)
    gaps = {
        w: float(g)
        for item in os.environ.get("FAKE_COHERENCE_GAP", "").split(",")
        if ":" in item
        for w, g in [item.split(":", 1)]
    }
    provinces = [loc for loc in obs if loc != "CD"]
    rows = []
    streams = {}
    for stream in ("observed", "imputed", "total"):
        found = []
        for week in sorted(obs["CD"]):
            known = obs["CD"][week] + gaps.get(week, 0.0)
            nat = 0.0 if stream == "imputed" else known
            prov = (
                0.0
                if stream == "imputed"
                else sum(obs[p].get(week, 0.0) for p in provinces)
            )
            if nat != prov:
                found.append(
                    {
                        "target_end_date": week,
                        "stream": stream,
                        "national": nat,
                        "provinces": prov,
                        "gap": nat - prov,
                    }
                )
        rows += found
        streams[stream] = {
            "weeks_with_gap": len(found),
            "max_abs_gap": max((abs(r["gap"]) for r in found), default=0),
            "total_gap": sum(r["gap"] for r in found),
        }
    data = {
        "schema_version": 1,
        "as_of": args.vintage,
        "national": "CD",
        "coherent": not rows,
        "streams": streams,
        "rows": rows,
        "handling": "The fixture model fits the provinces and adds the gap "
        "as an unassigned national component.",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(data, indent=2))
    return 0


def png(width: int = 64, height: int = 32) -> bytes:
    """A small valid PNG: a grey band, so the file opens anywhere."""
    raw = b"".join(
        b"\x00" + bytes([200 if 8 < y < 24 else 255] * width * 3)
        for y in range(height)
    )

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return (
            struct.pack(">I", len(data))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def cmd_plot(args: argparse.Namespace) -> int:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_bytes(png())
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="bdbv-hub-renewal")
    p.add_argument("--version", action="store_true")
    sub = p.add_subparsers(dest="command")
    d = sub.add_parser("data")
    d.add_argument("--ref", default="main")
    f = sub.add_parser("fit")
    f.add_argument("--model", required=True)
    f.add_argument("--config", default="main")
    f.add_argument("--vintage", required=True)
    f.add_argument("--sampler-json", default="{}")
    f.add_argument("--out", required=True)
    n = sub.add_parser("nowcast")
    n.add_argument("--model", required=True)
    n.add_argument("--config", default="main")
    n.add_argument("--fit", default=None)
    n.add_argument("--vintage", required=True)
    n.add_argument("--out", required=True)
    pl = sub.add_parser("plot-submission")
    pl.add_argument("--submission", required=True)
    pl.add_argument("--vintage", required=True)
    pl.add_argument("--out", required=True)
    pp = sub.add_parser("plot-ppc")
    pp.add_argument("--ppc", required=True)
    pp.add_argument("--out", required=True)
    sub.add_parser("chosen-model")
    b = sub.add_parser("bvd")
    b.add_argument("--vintages", nargs="*", default=None)
    sub.add_parser("hub-id")
    co = sub.add_parser("coherence")
    co.add_argument("--vintage", required=True)
    co.add_argument("--out", required=True)
    args = p.parse_args()
    if args.version:
        print("0.1.0+fixture")
        return 0
    return {
        "data": lambda _a: 0,
        "fit": cmd_fit,
        "nowcast": cmd_nowcast,
        "plot-submission": cmd_plot,
        "plot-ppc": cmd_plot,
        "coherence": cmd_coherence,
        "chosen-model": lambda _a: (
            print(os.environ.get("FAKE_CHOSEN_MODEL") or "renewal") or 0
        ),
        "bvd": lambda _a: 0,
        "hub-id": lambda _a: print("epiforecasts-seabbsrenewal") or 0,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
