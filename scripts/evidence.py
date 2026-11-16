"""Evidence for #199: the Chronos-2 models with and without the copula.

Usage, from the fm-copula worktree's root:
  evidence.py OUT draws MODEL...   Chronos-2's marginal draws at every
                                   origin, kept in OUT/draws.
  evidence.py OUT score            Pair the draws without and with the
                                   copula, estimate it, and score both on
                                   the release backtest's rolling truth.
Both arms pair the same marginal draws, so they differ only by pairing.
"""

import dataclasses
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import scoringrules as sr

from bdbvhubrenewal.data import load_daily, load_delays, load_target
from bdbvhubrenewal.evaluation.backtest import load_config
from bdbvhubrenewal.evaluation.nowcast import (
    resolve_truth_delay,
    rolling_truth,
    weeks_back_from_cutoff,
)
from bdbvhubrenewal.evaluation.scoring import score_forecasts
from bdbvhubrenewal.evaluation.store import origin_tasks
from bdbvhubrenewal.forecasters import NATIONAL, get_model
from bdbvhubrenewal.forecasters.copula import (
    GaussianCopula,
    estimate_copula,
    normal_scores,
)
from bdbvhubrenewal.forecasters.foundation import chronos_sampler

MODELS = [
    *(
        f"fm_chronos_{s}_{lay}"
        for s in ("weekly", "daily")
        for lay in ("complete", "ragged")
    ),
    "fm_chronos_weekly_ragged_ft",
    "fm_chronos_daily_ragged_ft",
]
out = Path(sys.argv[1])
(out / "draws").mkdir(parents=True, exist_ok=True)
d = "data"
target, daily = load_target(d), load_daily(d)
config = load_config()
tasks = origin_tasks(target, load_delays(d), config, daily)


def cached(path, checkpoint, groups, horizon, n, seed):
    if path.exists():
        return [np.load(path)]
    draws = chronos_sampler(groups, horizon, n, seed, checkpoint=checkpoint)
    np.save(path, draws[0])
    return draws


def forecast(name, origin, copula):
    model = get_model(name)
    path = out / "draws" / f"{name}_{origin}.npy"
    model = dataclasses.replace(
        model,
        sample=partial(cached, path, model.checkpoint),
        copula=copula,
    )
    rows = model.forecast(tasks[origin])
    return rows.assign(model=name, origin=origin)


if sys.argv[2] == "draws":
    import torch

    torch.set_num_threads(4)
    for name in sys.argv[3:]:
        for origin in tasks:
            try:
                forecast(name, origin, GaussianCopula())
            except Exception as err:  # a failed fit, as the store keeps
                print(name, origin, "failed", repr(err), flush=True)
                continue
            print(name, origin, "drawn", flush=True)
    sys.exit()

real = target.loc[target["kind"] == "real"]
delay = resolve_truth_delay(config, target)
truth = rolling_truth(real, delay)[["location", "target_end_date", "truth"]]
print("truth delay", delay, "days")
models = [m for m in MODELS if any((out / "draws").glob(f"{m}_*.npy"))]


def run(copula):
    frames = [
        forecast(m, o, copula)
        for m in models
        for o in tasks
        if (out / "draws" / f"{m}_{o}.npy").exists()
    ]
    return pd.concat(frames).merge(truth, on=["location", "target_end_date"])


flat = run(GaussianCopula())
weekly = flat.loc[flat["model"].str.startswith("fm_chronos_weekly")]
scores = normal_scores(
    weekly.drop(columns="truth"), truth, np.random.default_rng(199)
)
fit = estimate_copula(scores)
print("estimated", fit, "from", len(scores), "normal scores")
(out / "copula.json").write_text(json.dumps(dataclasses.asdict(fit)))
for name, x in scores.groupby("model"):
    print(" ", name, estimate_copula(x))
arms = {
    "independent": flat,
    "copula": run(fit),
    **{
        f"fixed_{t}_{p}_{n}": run(GaussianCopula(t, p, n))
        for t in (0.8, 0.9, 0.95)
        for p in (0.5, 0.7, 0.9)
        for n in (0.7, 0.9, 0.95)
    },
}


def coverage(samples, obs, level):
    lo, hi = np.quantile(samples, [(1 - level) / 2, (1 + level) / 2])
    return lo <= obs <= hi


def sums(rows, by):
    """Per-trajectory sums and their truth, over the rows of each `by`."""
    s = rows.groupby([*by, "sample"])["value"].sum().rename("value")
    t = (
        rows.drop_duplicates([*by, "location", "target_end_date"])
        .groupby(by)["truth"]
        .sum()
    )
    return s.reset_index().merge(t.reset_index(), on=by)


def covered(rows, by):
    g = sums(rows, by).groupby(by)
    return g.apply(
        lambda x: pd.Series(
            {
                f"cov{int(lv * 100)}": coverage(
                    x["value"], x["truth"].iloc[0], lv
                )
                for lv in (0.5, 0.9)
            }
        ),
        include_groups=False,
    ).mean()


def joint(rows):
    """Energy and variogram scores of each origin's place-week vector."""
    res = []
    for origin, x in rows.groupby("origin"):
        wide = x.pivot_table(
            index="sample",
            columns=["location", "target_end_date"],
            values="value",
        )
        obs = (
            x.drop_duplicates(["location", "target_end_date"])
            .set_index(["location", "target_end_date"])["truth"]
            .reindex(wide.columns)
            .to_numpy()
        )
        f = wide.to_numpy()
        res.append(
            {
                "es": float(sr.energy_score(obs, f)),
                "es_log": float(sr.energy_score(np.log1p(obs), np.log1p(f))),
                "vs": float(sr.variogram_score(obs, f, p=0.5)),
                "vs_log": float(
                    sr.variogram_score(np.log1p(obs), np.log1p(f), p=0.5)
                ),
            }
        )
    return pd.DataFrame(res).mean()


table, wis = [], {}
keys = ["model", "origin", "location", "target_end_date"]
for arm, rows in arms.items():
    rows = rows.assign(
        kind="real", vintage=rows["origin"], reference_date=rows["origin"]
    )
    s = score_forecasts(
        rows.drop(columns="truth").assign(horizon=0),
        truth,
        keys=[
            "model",
            "kind",
            "vintage",
            "reference_date",
            "location",
            "target_end_date",
        ],
    )
    wis[arm] = s.set_index(["model", "vintage", "location", "target_end_date"])[
        "wis"
    ]
    for model, m in rows.groupby("model"):
        prov = m.loc[m["location"] != NATIONAL]
        nat = m.loc[m["location"] == NATIONAL]
        row = {
            "model": model,
            "arm": arm,
            "wis": s.loc[s["model"] == model, "wis"].mean(),
        }
        row |= covered(m, ["origin", "location"]).add_prefix("weeks_")
        row |= covered(prov, ["origin", "target_end_date"]).add_prefix(
            "provsum_"
        )
        # National's draws against the sum of the provinces' truths.
        prov_truth = (
            prov.drop_duplicates(["origin", "location", "target_end_date"])
            .groupby(["origin", "target_end_date"], as_index=False)["truth"]
            .sum()
        )
        row |= covered(
            nat.drop(columns="truth").merge(
                prov_truth, on=["origin", "target_end_date"]
            ),
            ["origin", "target_end_date"],
        ).add_prefix("natvsprov_")
        row |= joint(m)
        table.append(row)

table = pd.DataFrame(table)
diff = (
    pd.concat([(wis[k] - wis["independent"]).abs() for k in arms])
    .groupby("model")
    .max()
)
same = {}
for model in models:
    a, *rest = (
        arms[k]
        .loc[arms[k]["model"] == model]
        .sort_values([*keys, "value"])["value"]
        .to_numpy()
        for k in arms
    )
    same[model] = all(np.array_equal(a, x) for x in rest)
table["max_wis_diff"] = table["model"].map(diff)
table["marginals_identical"] = table["model"].map(same)
pd.set_option("display.width", 250)
print(table.round(3).to_string(index=False))
table.to_csv(out / "copula-evidence.csv", index=False)
grid = table.groupby("arm")[["es", "es_log", "vs", "vs_log"]].mean()
print((grid / grid.loc["independent"]).sort_values("es_log").round(3))
n_cells = arms["independent"].drop_duplicates(keys).groupby("model").size()
print("scored cells per model\n", n_cells.to_string())
