"""Where fine-tuning Chronos-2 helps and where it fails: the tables.

Germany (development half): randomwalknowcast, the hub MeanEnsemble,
and the daily and weekly ragged Chronos-2 models zero-shot, fine-tuned
at round 1 and at each complete r3 checkpoint, on the cells every model
made. BDBV: the release backtest's complete target at each checkpoint.

Usage: python scripts/ft_review.py RESULTS CHECKPOINTS BDBV OUT

RESULTS holds the German `forecasts.csv.gz` and `teams-forecasts.csv.gz`;
CHECKPOINTS `<hash>/<date>.csv.gz`; BDBV `zero-shot/` and `<hash>/` with
`nowcast-scores.csv`. Writes Markdown tables to OUT.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from germany_chronos_dev import scores as germany_scores

from bdbvhubrenewal.evaluation import germany as de

READ = {"dtype": {"origin": str, "target_end_date": str}}
BASE = "randomwalknowcast"
CELL = ["origin", "location", "target_end_date"]


def _label(model: str, ckpt: str) -> tuple[str, str]:
    """Return (source, variant) of a model name."""
    if model in (BASE, "MeanEnsemble"):
        return "-", model
    source = "daily" if "_daily_" in model else "weekly"
    return source, ckpt if model.endswith("_ft") else "zero-shot"


def german_forecasts(results: Path, ckpt_dir: Path) -> pd.DataFrame:
    """Return every German model's quantiles, labelled by source/variant."""
    config = de.load_germany_config("config/germany.toml")
    ours = pd.read_csv(results / "forecasts.csv.gz", **READ)
    teams = pd.read_csv(results / "teams-forecasts.csv.gz", **READ)
    keep = [
        BASE,
        "fm_chronos_daily_ragged",
        "fm_chronos_weekly_ragged",
        "fm_chronos_daily_ragged_ft",
        "fm_chronos_weekly_ragged_ft",
    ]
    parts = [
        ours.loc[ours["model"].isin(keep)].assign(ckpt="round 1"),
        teams.loc[teams["model"] == "MeanEnsemble"].assign(ckpt="-"),
    ]
    n_dates = len(de.nowcast_dates(config, held_out=False))
    checkpoints = pd.read_csv("evidence/checkpoints.csv", dtype=str)
    for ckpt, outbreaks in checkpoints.itertuples(index=False):
        files = sorted((ckpt_dir / ckpt).glob("*.csv.gz"))
        if len(files) < n_dates:
            continue
        fc = pd.concat(pd.read_csv(f, **READ) for f in files)
        parts.append(fc.assign(ckpt=f"r3 n{outbreaks}"))
    fc = pd.concat(parts, ignore_index=True)
    fc = fc.loc[fc["origin"] < config.split_date]
    src, var = zip(
        *(_label(m, c) for m, c in zip(fc["model"], fc["ckpt"], strict=True)),
        strict=True,
    )
    fc["source"], fc["variant"] = src, var
    fc["model"] = fc["source"] + "/" + fc["variant"]
    return fc.drop(columns="ckpt")


def german_scores(fc: pd.DataFrame) -> pd.DataFrame:
    """Score every forecast, with its median and the cell's phase and size."""
    config = de.load_germany_config("config/germany.toml")
    sc = germany_scores(
        fc.drop(columns=["source", "variant"]), config.truth_delay
    )
    med = fc.loc[fc["quantile"] == 0.5, ["model", *CELL, "value"]]
    sc = sc.merge(med.rename(columns={"value": "median"}), on=["model", *CELL])
    n = sc.groupby(CELL)["model"].nunique()
    sc = sc.set_index(CELL).loc[n.index[n == n.max()]].reset_index()
    sc["source"] = sc["model"].str.split("/").str[0]
    sc["variant"] = sc["model"].str.split("/").str[1]
    # Phase at the origin from the truth: the sum ending on the nowcast
    # date against the one 14 days before.
    now = sc.loc[sc["horizon"] == 0].drop_duplicates(["origin", "location"])
    truth = sc.drop_duplicates(["location", "target_end_date"]).set_index(
        ["location", "target_end_date"]
    )["truth"]
    back = (pd.to_datetime(now["origin"]) - pd.Timedelta(days=14)).dt.strftime(
        "%Y-%m-%d"
    )
    before = truth.reindex(list(zip(now["location"], back, strict=True)))
    ratio = now["truth"].to_numpy() / before.to_numpy()
    phase = np.select(
        [ratio > 1.15, ratio < 0.87], ["rising", "falling"], "peak/flat"
    )
    phases = pd.DataFrame(
        {"origin": now["origin"], "location": now["location"], "phase": phase}
    )
    sc = sc.merge(phases, on=["origin", "location"], how="left")
    # Series size: tertile of the cell's truth within its level.
    sc["size"] = sc.groupby("pool")["truth"].transform(
        lambda t: pd.qcut(t, 3, labels=["small", "middle", "large"])
    )
    sc["rel_err"] = sc["median"] / sc["truth"].where(sc["truth"] > 0) - 1
    return sc


def relative(sc: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Relative WIS against randomwalknowcast and against zero-shot."""
    wis = sc.groupby([*by, "model"])["wis"].mean().unstack("model")
    rows = []
    for model in wis.columns:
        source, variant = model.split("/")
        row = wis[model] / wis[f"-/{BASE}"]
        out = row.rename("vs_rwnc").to_frame()
        zs = f"{source}/zero-shot"
        if source != "-" and variant != "zero-shot" and zs in wis:
            out["vs_zero_shot"] = wis[model] / wis[zs]
        out["model"] = model
        rows.append(out.reset_index())
    return pd.concat(rows, ignore_index=True)


def table(df: pd.DataFrame, index: list[str], col: str) -> str:
    """Pivot one column of `relative` to Markdown, models as columns."""
    t = df.pivot_table(index=index, columns="model", values=col, observed=True)
    return t.round(2).to_markdown()


def failure(sc: pd.DataFrame) -> pd.DataFrame:
    """WIS components, bias and coverage by model."""
    g = sc.groupby("model")
    out = g[["wis", "dispersion", "overprediction", "underprediction"]].mean()
    for c in ("dispersion", "overprediction", "underprediction"):
        out[f"{c}_share"] = out[c] / out["wis"]
    out["over_share_of_cells"] = g["rel_err"].apply(lambda e: (e > 0).mean())
    out["median_rel_err"] = g["rel_err"].median()
    out["coverage_50"] = g["coverage_50"].mean()
    out["coverage_95"] = g["coverage_95"].mean()
    return out


def bdbv(root: Path) -> pd.DataFrame:
    """Return BDBV complete-target scores of every checkpoint, labelled."""
    checkpoints = pd.read_csv("evidence/checkpoints.csv", dtype=str)
    names = {"zero-shot": "zero-shot"} | {
        c: f"r3 n{o}" for c, o in checkpoints.itertuples(index=False)
    }
    parts = []
    for ckpt, name in names.items():
        path = root / ckpt / "nowcast-scores.csv"
        if not path.exists():
            continue
        s = pd.read_csv(path)
        s = s.loc[s["target"] == "complete"]
        chronos = s["model"].str.startswith("fm_chronos")
        ft = s["model"].str.endswith("_ft")
        mine = chronos & (ft if ckpt != "zero-shot" else ~ft)
        if ckpt == "zero-shot":
            mine |= s["model"] == BASE
        s = s.loc[mine].copy()
        src = np.where(s["model"].str.contains("daily"), "daily", "weekly")
        s["model"] = np.where(
            s["model"] == BASE,
            f"-/{BASE}",
            np.char.add(src.astype(str), "/" + name),
        )
        parts.append(s)
    s = pd.concat(parts, ignore_index=True)
    s["pool"] = np.where(s["location"] == "CD", "national", "provinces")
    return s.drop_duplicates(["model", "origin", "location", "target_end_date"])


def main() -> None:
    """Write the review's tables."""
    results, ckpts, bdbv_root, out = map(Path, sys.argv[1:5])
    fc = german_forecasts(results, ckpts)
    sc = german_scores(fc)
    sc.to_csv(out.with_suffix(".germany-scores.csv.gz"), index=False)
    md = ["# Tables", ""]
    md += [
        f"Germany: {sc.groupby('model').size().iloc[0]} matched cells "
        "per model.",
        "",
    ]
    for by, name in (
        (["pool"], "level"),
        (["horizon"], "horizon (days back)"),
        (["phase"], "phase at origin"),
        (["pool", "size"], "series size"),
    ):
        rel = relative(sc, by)
        md += [f"## Germany by {name}: WIS relative to randomwalknowcast", ""]
        md += [table(rel, by, "vs_rwnc"), ""]
        md += [
            f"## Germany by {name}: fine-tuned WIS relative to zero-shot",
            "",
        ]
        md += [
            table(rel.dropna(subset=["vs_zero_shot"]), by, "vs_zero_shot"),
            "",
        ]
    md += ["## Germany failure modes (all cells)", ""]
    md += [failure(sc).round(3).to_markdown(), ""]
    turn = sc.loc[sc["horizon"] >= -7]
    md += ["## Germany median relative error at horizons 0 and -7 by phase", ""]
    md += [
        turn.pivot_table(
            index="phase", columns="model", values="rel_err", aggfunc="median"
        )
        .round(3)
        .to_markdown(),
        "",
    ]
    b = bdbv(bdbv_root)
    k = ["origin", "truth_vintage", "location", "target_end_date"]
    n = b.groupby(k)["model"].nunique()
    m = b.set_index(k).loc[n.index[n == n.max()]].reset_index()
    md += [f"## BDBV ({len(m) // m['model'].nunique()} matched cells)", ""]
    wis = m.groupby(["pool", "model"])["wis"].mean().unstack("model")
    md += [(wis.div(wis[f"-/{BASE}"], axis=0)).round(2).to_markdown(), ""]
    comp = m.groupby("model")[
        [
            "wis",
            "dispersion",
            "overprediction",
            "underprediction",
            "bias",
            "coverage_50",
            "coverage_90",
        ]
    ].mean()
    md += [comp.round(3).to_markdown(), ""]
    # Chronos-only cells (two origins): fine-tuned against zero-shot.
    c = b.loc[b["model"] != f"-/{BASE}"]
    n = c.groupby(k)["model"].nunique()
    c = c.set_index(k).loc[n.index[n == n.max()]].reset_index()
    w = c.groupby(["pool", "model"])["wis"].mean().unstack("model")
    for src in ("daily", "weekly"):
        cols = [x for x in w.columns if x.startswith(src)]
        w[cols] = w[cols].div(w[f"{src}/zero-shot"], axis=0)
    md += [
        f"## BDBV Chronos cells ({len(c) // c['model'].nunique()}, two "
        "origins): WIS relative to the same source's zero-shot",
        "",
        w.round(2).to_markdown(),
        "",
    ]
    out.write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
