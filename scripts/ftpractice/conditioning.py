"""Check whether interval width tracks each held-out simulation's noise.

Joins the held-out quantiles (`heldout_quantiles.py`) to each simulation's
grid factors (`settings-<seed>.csv`) and to its realised spread: the
ratio of the final 7-day count to the count reported by the cut-off, for
the week ending at the cut-off, per place and weekly cut-off. The same
ratio is computed for the German development half from the hub's
triangle, so the sims' spread and Germany's sit on one scale.

    python scripts/ftpractice/conditioning.py QUANTILES_DIR SIM_DIR \
        GERMAN_TRIANGLE OUT_DIR --seeds 32000 32039

Writes OUT_DIR/per-sim.csv, OUT_DIR/summary.md and prints the summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MIN_FINAL = 10
"""Rows with a smaller final count are left out of relative widths."""

NOWCAST = {"daily": range(-7, 0), "3day": range(-2, 0), "weekly": range(-1, 0)}
"""Steps before the cut-off taken as its last week, per pure cadence."""

FORECAST = {"daily": range(0, 14), "3day": range(0, 5), "weekly": range(0, 2)}
"""Steps after the cut-off taken as its first two weeks."""


def spread(r: np.ndarray) -> float:
    """Return the 90% range of ratios over their median."""
    r = r[np.isfinite(r)]
    if len(r) < 5:
        return np.nan
    lo, mid, hi = np.quantile(r, [0.05, 0.5, 0.95])
    return float((hi - lo) / mid)


def sim_ratios(sim_dir: Path, seeds: range) -> pd.DataFrame:
    """Return final/reported 7-day ratios at weekly cut-offs, per place."""
    out = []
    for seed in seeds:
        c = pd.read_csv(sim_dir / f"counts-{seed}.csv.gz")
        c = c[c["timestep"] == "day"]
        if c.empty:
            continue
        c["simulation"] = c["outbreak"].str.rsplit("-", n=1).str[0]
        ref = pd.to_datetime(c["reference_date"])
        c["t"] = (ref - ref.min()).dt.days
        c["d"] = (pd.to_datetime(c["report_date"]) - ref).dt.days
        for (sim, loc), g in c.groupby(["simulation", "location"]):
            t0 = g["t"].min()
            n_t, n_d = g["t"].max() - t0 + 1, g["d"].max() + 1
            m = np.zeros((n_t, n_d))
            np.add.at(m, (g["t"] - t0, g["d"]), g["count"])
            cum = np.cumsum(m, axis=1)
            for cut in range(13, n_t, 7):
                rows = np.arange(cut - 6, cut + 1)
                final = m[rows].sum()
                if final < MIN_FINAL:
                    continue
                lag = np.minimum(cut - rows, n_d - 1)
                seen = cum[rows, lag].sum()
                out.append((sim, loc, final / max(seen, 1)))
    return pd.DataFrame(out, columns=["simulation", "location", "ratio"])


def german_ratios(path: Path, first: str, last: str) -> pd.DataFrame:
    """Return the German final/reported 7-day ratios at horizon 0."""
    wide = pd.read_csv(path)
    vals = [c for c in wide.columns if c.startswith("value_")]
    wide["date"] = pd.to_datetime(wide["date"])
    out = []
    for key, g in wide.groupby(["location", "age_group"]):
        g = g.set_index("date").sort_index()
        m = g[vals].fillna(0).to_numpy()
        dates = g.index
        for cut in pd.date_range(first, last):
            idx = np.where(
                (dates > cut - pd.Timedelta(days=7)) & (dates <= cut)
            )[0]
            if len(idx) < 7:
                continue
            final = m[idx].sum()
            if final < MIN_FINAL:
                continue
            lag = (cut - dates[idx]).days.to_numpy()
            seen = sum(
                m[i, : lag_i + 1].sum()
                for i, lag_i in zip(idx, lag, strict=True)
            )
            out.append(("/".join(key), final / max(seen, 1)))
    return pd.DataFrame(out, columns=["series", "ratio"])


def widths(q: pd.DataFrame, steps: dict[str, range]) -> pd.DataFrame:
    """Return each row's relative 90% width and coverage at `steps`."""
    keep = np.zeros(len(q), bool)
    for source, r in steps.items():
        keep |= (q["source"] == source) & q["step"].isin(list(r))
    q = q[keep & (q["final"] >= MIN_FINAL)].copy()
    q["width90"] = (q["q0.95"] - q["q0.05"]) / q["final"]
    q["cover90"] = (q["final"] >= q["q0.05"]) & (q["final"] <= q["q0.95"])
    q["cover50"] = (q["final"] >= q["q0.25"]) & (q["final"] <= q["q0.75"])
    return q


def main() -> None:
    """Join widths to factors and realised spread, and summarise."""
    p = argparse.ArgumentParser()
    p.add_argument("quantiles", type=Path)
    p.add_argument("sim_dir", type=Path)
    p.add_argument("triangle", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--seeds", type=int, nargs=2, required=True)
    p.add_argument("--german-first", default="2021-11-22")
    p.add_argument("--german-last", default="2022-02-09")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = range(args.seeds[0], args.seeds[1] + 1)

    settings = pd.concat(
        pd.read_csv(args.sim_dir / f"settings-{s}.csv") for s in seeds
    ).rename(columns={"outbreak": "simulation"})
    factors = settings[
        ["simulation", "sqrt_phi", "shocks", "delay_change", "spread", "cases"]
    ].copy()
    factors["cases"] = pd.to_numeric(factors["cases"], errors="coerce")

    ratios_path = args.out / "sim-ratios.csv"
    if ratios_path.exists():
        ratios = pd.read_csv(ratios_path)
    else:
        ratios = sim_ratios(args.sim_dir, seeds)
        ratios.to_csv(ratios_path, index=False)
    # Own spread: each place's spread over its cut-offs, the median over
    # a simulation's places, so it is what one series' context can show.
    by_place = ratios.groupby(["simulation", "location"])["ratio"].apply(
        lambda r: spread(r.to_numpy())
    )
    own = by_place.groupby("simulation").median()
    factors = factors.merge(
        own.rename("own_spread"), on="simulation", how="left"
    )
    factors["quiet"] = (
        (factors["shocks"] == "none")
        & (factors["delay_change"] == "static")
        & (factors["sqrt_phi"] <= factors["sqrt_phi"].quantile(0.25))
    )
    factors["static_noshock"] = (factors["shocks"] == "none") & (
        factors["delay_change"] == "static"
    )
    german = german_ratios(args.triangle, args.german_first, args.german_last)
    germany = german["ratio"].to_numpy()
    german_own = (
        german.groupby("series")["ratio"]
        .apply(lambda r: spread(r.to_numpy()))
        .median()
    )

    lines = [
        f"Held-out sims: {len(factors)}; realised spread = 90% range of "
        "final/reported for the week to the cut-off, over its median.",
        f"Germany (dev half, horizon 0, {len(germany)} place-age-dates): "
        f"spread {spread(germany):.2f}, median ratio "
        f"{np.median(germany):.2f}; own spread (median over series) "
        f"{german_own:.2f}.",
        f"Sims pooled: spread {spread(ratios['ratio'].to_numpy()):.2f}, "
        f"median ratio {ratios['ratio'].median():.2f}.",
    ]
    for name, sub in (
        ("quiet", factors["quiet"]),
        ("static+no shock", factors["static_noshock"]),
        ("rest", ~factors["static_noshock"]),
    ):
        ids = set(factors.loc[sub, "simulation"])
        r = ratios.loc[ratios["simulation"].isin(ids), "ratio"].to_numpy()
        lines.append(f"Sims {name}: spread {spread(r):.2f} ({sub.sum()} sims)")
    by_change = {
        k: spread(
            ratios.loc[
                ratios["simulation"].isin(
                    set(factors.loc[factors["delay_change"] == k, "simulation"])
                ),
                "ratio",
            ].to_numpy()
        )
        for k in sorted(factors["delay_change"].unique())
    }
    lines.append(
        "Realised spread by delay_change: "
        + ", ".join(f"{k} {v:.2f}" for k, v in by_change.items())
    )

    per_sim, rows = [], []
    for f in sorted(args.quantiles.glob("*.csv.gz")):
        name = f.name.removesuffix(".csv.gz")
        q = pd.read_csv(f)
        for part, steps in (("nowcast", NOWCAST), ("forecast", FORECAST)):
            w = widths(q, steps)
            s = (
                w.groupby("simulation")
                .agg(
                    width90=("width90", "median"),
                    cover90=("cover90", "mean"),
                    cover50=("cover50", "mean"),
                    n=("width90", "size"),
                )
                .reset_index()
                .merge(factors, on="simulation")
            )
            s.insert(0, "part", part)
            s.insert(0, "checkpoint", name)
            per_sim.append(s)
            # Row level: does width follow the row's own error, and are
            # the narrowest and widest tenths of rows equally covered?
            err = (w["final"] - w["q0.5"]).abs() / w["final"]
            decile = pd.qcut(w["width90"], 10, labels=False, duplicates="drop")
            by_decile = w.groupby(decile)["cover90"].mean()
            rec = {
                "checkpoint": name,
                "part": part,
                "rows": len(w),
                "width90": w["width90"].median(),
                "cover90": w["cover90"].mean(),
                "cover50": w["cover50"].mean(),
                "rho_abs_err": w["width90"].corr(err, "spearman"),
                "cover90_narrow10": by_decile.iloc[0],
                "cover90_wide10": by_decile.iloc[-1],
                "rho_sqrt_phi": s["width90"].corr(s["sqrt_phi"], "spearman"),
                "rho_own_spread": s["width90"].corr(
                    s["own_spread"], "spearman"
                ),
                "quiet": s.loc[s["quiet"], "width90"].median(),
                "static_noshock": s.loc[
                    s["static_noshock"], "width90"
                ].median(),
                "rest": s.loc[~s["static_noshock"], "width90"].median(),
                "shock_none": s.loc[s["shocks"] == "none", "width90"].median(),
                "shock_any": s.loc[s["shocks"] != "none", "width90"].median(),
            }
            for k in by_change:
                rec[f"dc_{k}"] = s.loc[
                    s["delay_change"] == k, "width90"
                ].median()
            rows.append(rec)
    per = pd.concat(per_sim, ignore_index=True)
    per.to_csv(args.out / "per-sim.csv", index=False)
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "summary.csv", index=False)
    rho = factors["own_spread"].corr(factors["sqrt_phi"], "spearman")
    lines.append(f"Own spread vs sqrt_phi (sims themselves): rho {rho:.2f}")
    own_by = factors.groupby("delay_change")["own_spread"].median()
    lines.append(
        "Sims' own spread by delay_change (median): "
        + ", ".join(f"{k} {v:.2f}" for k, v in own_by.items())
        + f"; all {factors['own_spread'].median():.2f}"
    )
    (args.out / "spreads.json").write_text(
        json.dumps(
            {
                "germany_pooled": spread(germany),
                "germany_own": float(german_own),
                "sims_pooled": spread(ratios["ratio"].to_numpy()),
                "sims_own": float(factors["own_spread"].median()),
            }
        )
    )
    text = "\n".join(lines) + "\n\n" + table.round(3).to_markdown(index=False)
    (args.out / "summary.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
