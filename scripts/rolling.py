"""Evidence for #221: rolling-window copula on BDBV and Germany.

Usage, from the fm-copula-rolling worktree's root:
  rolling.py OUT bdbv DRAWS    BDBV release backtest: randomwalknowcast
                               and the Chronos-2 models (marginal draws
                               cached in DRAWS by #199's evidence.py).
  rolling.py OUT germany DATA  Germany's development half, randomwalknowcast.
Arms: independent (each model's own pairing; Chronos-2 with none), fixed
(the prior copula) and rolling (estimated at each origin, shrunk to it).
"""

import dataclasses
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import scoringrules as sr

from bdbvhubrenewal.evaluation.scoring import score_forecasts
from bdbvhubrenewal.forecasters import NATIONAL, get_model
from bdbvhubrenewal.forecasters.copula import (
    PRIOR,
    GaussianCopula,
    RollingCopula,
    pair_samples,
)

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
case = sys.argv[2]
SEED = 221


def coverage(samples, obs, level):
    lo, hi = np.quantile(samples, [(1 - level) / 2, (1 + level) / 2])
    return lo <= obs <= hi


def covered(rows, by):
    """Coverage of per-trajectory sums over `by` groups, and their count."""
    s = rows.groupby([*by, "sample"])["value"].sum().rename("value")
    t = (
        rows.drop_duplicates([*by, "location", "target_end_date"])
        .groupby(by)["truth"]
        .sum()
    )
    x = s.reset_index().merge(t.reset_index(), on=by)
    res = x.groupby(by).apply(
        lambda g: pd.Series(
            {
                f"cov{int(lv * 100)}": coverage(
                    g["value"], g["truth"].iloc[0], lv
                )
                for lv in (0.5, 0.9)
            }
        ),
        include_groups=False,
    )
    return pd.Series({**res.mean().to_dict(), "n": len(res)})


def joint(rows):
    res = []
    for _, x in rows.groupby(["model", "origin"]):
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
        lo, lf = np.log1p(obs), np.log1p(f)
        res.append(
            {
                "es": float(sr.energy_score(obs, f)),
                "es_log": float(sr.energy_score(lo, lf)),
                "vs_log": float(sr.variogram_score(lo, lf, p=0.5)),
            }
        )
    return pd.DataFrame(res).mean()


def metrics(arms, truth, national, group_of=lambda m: m):
    """One row per model and arm; `group_of` maps a model label to the
    model it reports under."""
    table = []
    keys = ["model", "origin", "location", "target_end_date"]
    base = arms["independent"]
    for arm, rows in arms.items():
        rows = rows.merge(truth, on=["location", "target_end_date"])
        s = score_forecasts(
            rows.drop(columns=["truth", "truth_vintage"]).assign(horizon=0),
            truth[["location", "target_end_date", "truth"]],
            keys=keys,
        )
        rows = rows.assign(group=rows["model"].map(group_of))
        s = s.assign(group=s["model"].map(group_of))
        for group, m in rows.groupby("group"):
            sm = s.loc[s["group"] == group]
            is_nat = m["location"] == national
            prov = m.loc[~is_nat]
            nat = m.loc[is_nat]
            row = {
                "model": group,
                "arm": arm,
                "wis_places": sm.loc[sm["location"] != national, "wis"].mean(),
                "wis_national": sm.loc[
                    sm["location"] == national, "wis"
                ].mean(),
            }
            row |= covered(m, ["model", "origin", "location"]).add_prefix(
                "weeks_"
            )
            row |= covered(
                prov, ["model", "origin", "target_end_date"]
            ).add_prefix("provsum_")
            if not nat.empty:
                prov_truth = (
                    prov.drop_duplicates(
                        ["model", "origin", "location", "target_end_date"]
                    )
                    .groupby(
                        ["model", "origin", "target_end_date"], as_index=False
                    )["truth"]
                    .sum()
                )
                row |= covered(
                    nat.drop(columns="truth").merge(
                        prov_truth, on=["model", "origin", "target_end_date"]
                    ),
                    ["model", "origin", "target_end_date"],
                ).add_prefix("natvsprov_")
            row |= joint(m)
            # Each place's draws, compared with the independent arm's.
            mine = m.loc[~is_nat]
            ref = base.assign(group=base["model"].map(group_of))
            ref = ref.loc[
                (ref["group"] == group) & (ref["location"] != national)
            ]
            a = mine.sort_values([*keys, "value"])["value"].to_numpy()
            b = (
                ref.merge(mine[keys].drop_duplicates(), on=keys)
                .sort_values([*keys, "value"])["value"]
                .to_numpy()
            )
            row["places_identical"] = bool(np.array_equal(a, b))
            table.append(row)
    return pd.DataFrame(table)


def report(arms, truth, national, fits, group_of=lambda m: m):
    table = metrics(arms, truth, national, group_of)
    pd.set_option("display.width", 250)
    print(table.round(3).to_string(index=False))
    table.to_csv(out / f"{case}-evidence.csv", index=False)
    fits.to_csv(out / f"{case}-fits.csv", index=False)
    print(fits.round(3).to_string(index=False))


if case == "bdbv":
    from bdbvhubrenewal.data import load_daily, load_delays, load_target
    from bdbvhubrenewal.evaluation.backtest import load_config, run_forecasts
    from bdbvhubrenewal.evaluation.nowcast import (
        resolve_truth_delay,
        rolling_truth,
    )
    from bdbvhubrenewal.evaluation.store import origin_tasks
    from bdbvhubrenewal.forecasters.foundation import chronos_sampler

    draws = Path(sys.argv[3])
    d = "data"
    target, daily = load_target(d), load_daily(d)
    config = load_config()
    tasks = origin_tasks(target, load_delays(d), config, daily)
    real = target.loc[target["kind"] == "real"]
    delay = resolve_truth_delay(config, target)
    truth = rolling_truth(real, delay)
    print("truth delay", delay)

    def cached(path, checkpoint, groups, horizon, n, seed):
        return [np.load(path)]

    def chronos(name, copula):
        frames = []
        for origin, task in tasks.items():
            path = draws / f"{name}_{origin}.npy"
            if not path.exists():
                continue
            model = get_model(name)
            model = dataclasses.replace(
                model,
                sample=partial(cached, path, model.checkpoint),
                copula=copula,
            )
            frames.append(
                model.forecast(task).assign(model=name, origin=origin)
            )
        return pd.concat(frames, ignore_index=True)

    models = sorted({p.stem.rsplit("_", 1)[0] for p in draws.glob("*.npy")})
    rwnc = run_forecasts([get_model("randomwalknowcast")], tasks.values())
    rwnc = rwnc.loc[rwnc["as_of"].isna()].rename(columns={"vintage": "origin"})
    rwnc = rwnc[
        ["model", "origin", "location", "target_end_date", "sample", "value"]
    ]
    independent = pd.concat(
        [rwnc, *(chronos(m, GaussianCopula()) for m in models)],
        ignore_index=True,
    )
    fixed_chronos = pd.concat(
        [chronos(m, PRIOR) for m in models], ignore_index=True
    )
    rng = np.random.default_rng(SEED)
    fixed_rwnc = pd.concat(
        [pair_samples(r, PRIOR, rng) for _, r in rwnc.groupby("origin")],
        ignore_index=True,
    )
    fixed = pd.concat([fixed_rwnc, fixed_chronos], ignore_index=True)
    rolling_in = pd.concat([rwnc, fixed_chronos], ignore_index=True)
    rolled, fits = RollingCopula(PRIOR).pair(rolling_in, truth, seed=SEED)
    w25, _ = RollingCopula(PRIOR, weight=25.0).pair(
        rolling_in, truth, seed=SEED
    )
    arms = {
        "independent": independent,
        "fixed": fixed,
        "rolling": rolled,
        "rolling_w25": w25,
    }
    report(arms, truth, NATIONAL, fits)

if case == "germany":
    from bdbvhubrenewal.data.germany import (
        AGE_GROUPS,
        GERMANY,
        hub_forecasts,
        hub_target,
        hub_triangle,
        nowcast_tasks,
    )
    from bdbvhubrenewal.evaluation import germany as de
    from bdbvhubrenewal.evaluation.backtest import run_forecasts
    from bdbvhubrenewal.evaluation.nowcast import rolling_truth

    cfg = de.load_germany_config("config/germany.toml")
    dates = de.nowcast_dates(cfg)
    assert dates[0] == "2021-11-22" and dates[-1] == "2022-02-09", dates[-1]
    triangle = hub_triangle(Path(sys.argv[3]) / Path(cfg.triangle).name)
    model = get_model("randomwalknowcast")
    cache = out / "germany-rwnc"
    cache.mkdir(exist_ok=True)
    # Optional "i/k": draw only every k-th date from the i-th, then stop.
    part = sys.argv[4] if len(sys.argv) > 4 else None
    todo = dates
    if part:
        i, k = (int(x) for x in part.split("/"))
        todo = dates[i::k]
    for date in todo:
        path = cache / f"{date}.pkl"
        if path.exists():
            continue
        frames = []
        for task in nowcast_tasks(triangle, date, cfg.n_samples, cfg.seed):
            s = run_forecasts([model], [task])
            if s.empty:
                continue
            h = hub_forecasts(s, task, date)
            ages = h["location"].isin(AGE_GROUPS).any()
            frames.append(
                h.assign(
                    model=f"randomwalknowcast|{'ages' if ages else 'states'}"
                )
            )
        pd.concat(frames, ignore_index=True).rename(
            columns={"forecast_date": "origin"}
        ).to_pickle(path)
        print(date, flush=True)
    if part:
        sys.exit()
    samples = pd.concat(
        [pd.read_pickle(cache / f"{d}.pkl") for d in dates],
        ignore_index=True,
    )
    samples = samples[
        ["model", "origin", "location", "target_end_date", "sample", "value"]
    ]
    ends = sorted(set(samples["target_end_date"]))
    truth = rolling_truth(hub_target(triangle, ends), cfg.truth_delay)
    rng = np.random.default_rng(SEED)
    fixed = pd.concat(
        [
            pair_samples(r, PRIOR, rng, national=GERMANY)
            for _, r in samples.groupby(["model", "origin"])
        ],
        ignore_index=True,
    )
    rolled, fits = RollingCopula(PRIOR, national=GERMANY).pair(
        samples, truth, seed=SEED
    )
    w25, _ = RollingCopula(PRIOR, weight=25.0, national=GERMANY).pair(
        samples, truth, seed=SEED
    )
    arms = {
        "independent": samples,
        "fixed": fixed,
        "rolling": rolled,
        "rolling_w25": w25,
    }
    report(arms, truth, GERMANY, fits)
