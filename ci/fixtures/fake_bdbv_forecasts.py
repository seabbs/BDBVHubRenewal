"""Stand-in for the main repo's `bdbv-forecasts` CLI, for the dry run.

It implements the interface in INTEGRATION.md on fixture data, so the
step scripts run end to end without the model. Behaviour per candidate
is set by `FAKE_SCENARIOS`, e.g. `main=fail,retry-1=pass`, read with the
candidate id the step script exports as `CAND_ID`. Scenarios:

  pass         converged fit, plausible forecast
  warn         converged but past the warning thresholds
  fail         not converged (R-hat 1.4, divergences)
  implausible  converged, but the forecast explodes
  crash        the fit command fails
  negative     the nowcast writes a negative value (format check)

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
import struct
import sys
import zlib
from collections import defaultdict
from pathlib import Path

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


def cmd_fit(args: argparse.Namespace) -> int:
    sc = scenario()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "progress.log").write_text("iteration=2000 divergences=0\n")
    if sc == "crash":
        print("fake fit: crashed", file=sys.stderr)
        return 1
    rng = random.Random(1)
    sampler = json.loads(args.sampler_json or "{}")
    chains = int(sampler.get("num_chains", 4))
    draws = int(sampler.get("num_samples", 1000))
    rhat, ess, div = {
        "warn": (1.07, 80.0, 15),
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
        rows.append(["CD", s, w, v])
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
    (Path(args.out) / "national.json").write_text(json.dumps(sidecar))
    print(f"Wrote {path}")
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
    p = argparse.ArgumentParser(prog="bdbv-forecasts")
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
    args = p.parse_args()
    if args.version:
        print("0.1.0+fixture")
        return 0
    return {
        "data": lambda _a: 0,
        "fit": cmd_fit,
        "nowcast": cmd_nowcast,
        "plot-submission": cmd_plot,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
