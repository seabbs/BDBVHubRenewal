# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
# Synced from the prototype by scripts/sync_pyrenew.py. Edit there.
"""Plug the model into the `bdbv_hub_renewal` interfaces.

- `PyRenewProvinceModel(config).forecast(task)` is a `ForecastModel`.
- `fit_renewal(task, sampler, fit_dir, config)` and
  `forecast_renewal(task, fit_dir)` are the `fit` and `forecast` of a
  `bdbv_hub_renewal.fitting.FitConfig`.
- `ppc_rows` gives the inputs of `bdbv_hub_renewal.ppc.summarise_draws`.
"""

from __future__ import annotations

import datetime as dt
import json
import pickle
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pandas as pd
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin

from bdbv_hub_renewal.observation import hub_delay_pmf
from bdbv_hub_renewal.ppc import summarise_draws, write_ppc

from .latent import adjacency_kernel
from .model import MAIN, ModelConfig, Vintage, build_model, fit
from .model import forecast_trajectories as _trajectories

NATIONAL = "CD"
START_COUNT = 5
"""The week grid starts one week before national weekly onsets first
reach this count. Earlier weeks are returned as observed."""


CENTRES = Path(__file__).parent.parent / "province_centres.csv"
"""Population-weighted centres of all 26 provinces (see the file header)."""


def province_centres() -> pd.DataFrame:
    """Read the province centres shipped with the package.

    Returns
    -------
    pandas.DataFrame
        Columns `location`, `latitude`, `longitude`, `worldpop`, `zones`.
    """
    return pd.read_csv(CENTRES, comment="#")


def gravity_kernel(
    locations: tuple[str, ...], table: pd.DataFrame, decay_km: float = 300.0
) -> np.ndarray:
    r"""Row-normalised gravity kernel between provinces.

    $K_{qp} \propto N_p \exp(-D_{qp} / \text{decay})$ for $q \ne p$, as
    `bdbv_hub_renewal.simulate.gravity_kernel`, with populations $N$ from
    `table` and great-circle distances $D$ between the centres in
    `province_centres()`, which cover all 26 provinces.

    Returns
    -------
    numpy.ndarray
        Rows sum to one, zero diagonal.

    Raises
    ------
    ValueError
        If a province has no centre or population, or the kernel is not
        finite.
    """
    centres = province_centres().set_index("location").reindex(list(locations))
    pops = table.set_index("location")["population"].reindex(list(locations))
    if (
        centres[["latitude", "longitude"]].isna().any().any()
        or pops.isna().any()
    ):
        raise ValueError("Missing province centre or population.")
    lat = np.radians(centres["latitude"].to_numpy(float))
    lon = np.radians(centres["longitude"].to_numpy(float))
    a = (
        np.sin((lat[:, None] - lat[None, :]) / 2) ** 2
        + np.cos(lat[:, None])
        * np.cos(lat[None, :])
        * np.sin((lon[:, None] - lon[None, :]) / 2) ** 2
    )
    dist = 2 * 6371 * np.arcsin(np.sqrt(a))
    k = pops.to_numpy(float)[None, :] * np.exp(-dist / decay_km)
    np.fill_diagonal(k, 0.0)
    k = k / k.sum(axis=1, keepdims=True)
    if not np.all(np.isfinite(k)):
        raise ValueError("Importation kernel is not finite.")
    return k


BVD_PROVINCES = {
    "ituri": "CD-IT",
    "nord_kivu": "CD-NK",
    "haut_uele": "CD-HU",
    "sud_kivu": "CD-SK",
    "tshopo": "CD-TO",
    "bas_uele": "CD-BU",
    "sud_ubangi": "CD-SU",
    "nord_ubangi": "CD-NU",
}
"""BVDOutbreakSize patch keys to hub province codes."""
BVD_SHIFT = 6
"""Days from BVD's infection-time Rt to our onset-time Rt (incubation)."""
BVD_TAU_MIN = 0.05
Z90 = 1.6448536


def bvd_rows(
    estimates: pd.DataFrame,
    timeseries: pd.DataFrame,
    draws: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Tidy BVDOutbreakSize summaries of one release into model rows.

    Reads `province_estimates.csv`, `province_timeseries.csv` and,
    optionally, `posterior_draws.csv`. The national term uses the draws
    when given (mean and SD of log `R_T`), otherwise the `National` row
    of the time series at `as_of`. Each patch carries the time series'
    own national row as its reference, so the deviation uses one table.
    Pooled patches (`pooled = 1`) share one row and list their members.

    Returns
    -------
    pandas.DataFrame
        Columns `group`, `members`, `as_of`, `log_mean`, `log_sd`,
        `ref_log_mean`, `ref_log_sd`.

    Raises
    ------
    ValueError
        On an unknown BVD patch key.
    """
    as_of = str(estimates["as_of"].iloc[0])
    nat = timeseries.loc[
        (timeseries["area"] == "National")
        & (timeseries["date"] == as_of)
        & (timeseries["series"] == "rt")
    ].iloc[0]
    ref_mean = float(np.log(nat["median"]))
    ref_sd = float(np.log(nat["upper_90"] / nat["lower_90"]) / (2 * Z90))
    if draws is not None:
        log_rt = np.log(draws["R_T"].to_numpy(float))
        nat_mean, nat_sd = float(log_rt.mean()), float(log_rt.std(ddof=1))
    else:
        nat_mean, nat_sd = ref_mean, ref_sd
    rows = [("national", "", as_of, nat_mean, nat_sd, nat_mean, nat_sd)]
    unknown = set(estimates["province"]) - set(BVD_PROVINCES)
    if unknown:
        raises = sorted(unknown)
        raise ValueError(f"Unknown BVD patch keys {raises}.")
    by_patch = estimates.groupby("patch", sort=False)
    for patch, g in by_patch:
        members = ";".join(BVD_PROVINCES[k] for k in g["province"])
        row = g.iloc[0]
        rows.append(
            (
                str(patch),
                members,
                as_of,
                float(np.log(row["R_T_median"])),
                float(np.log(row["R_T_upper"] / row["R_T_lower"]) / (2 * Z90)),
                ref_mean,
                ref_sd,
            )
        )
    return pd.DataFrame(
        rows,
        columns=[
            "group",
            "members",
            "as_of",
            "log_mean",
            "log_sd",
            "ref_log_mean",
            "ref_log_sd",
        ],
    )


def bvd_inputs(
    rows: pd.DataFrame,
    names: list[str],
    start: dt.date,
    masked: tuple[str, ...] = (),
) -> dict | None:
    """Map tidy BVD rows onto the modelled provinces.

    Groups with no modelled member are dropped. Leaving a province out
    drops the national term and its patch, since both carry its data.

    Returns
    -------
    dict or None
        `day`, `groups`, `values`, `scales` and `kinds` for
        `RtPseudoObservation`, or None if nothing is left.
    """
    day = (pd.Timestamp(rows["as_of"].iloc[0]).date() - start).days + BVD_SHIFT
    groups, values, scales, kinds = [], [], [], []
    for row in rows.itertuples():
        if row.group == "national":
            if masked:
                continue
            member = np.ones(len(names), bool)
            value, scale, kind = row.log_mean, row.log_sd, 0
        else:
            codes = row.members.split(";")
            if set(codes) & set(masked):
                continue
            member = np.array([p in codes for p in names])
            value = row.log_mean - row.ref_log_mean
            scale = np.sqrt(
                max(row.log_sd**2 - row.ref_log_sd**2, BVD_TAU_MIN**2)
            )
            kind = 1
        if member.any():
            groups.append(member)
            values.append(value)
            scales.append(scale)
            kinds.append(kind)
    if not groups:
        return None
    return {
        "day": day,
        "groups": np.array(groups),
        "values": np.array(values),
        "scales": np.array(scales),
        "kinds": np.array(kinds),
    }


def _saturday(day: str | dt.date) -> dt.date:
    d = pd.Timestamp(day).date()
    return d + dt.timedelta(days=(5 - d.weekday()) % 7)


def vintage_from_tables(
    target: pd.DataFrame,
    delays: pd.DataFrame | None,
    vintage: str,
    locations: tuple[str, ...],
    table: pd.DataFrame,
    config: ModelConfig = MAIN,
    horizon_weeks: int = 3,
    masked: tuple[str, ...] = (),
    bvd: pd.DataFrame | None = None,
) -> Vintage:
    """Build a `Vintage` from the target and delay rows of one vintage.

    The active set is the provinces in `locations` with any case in this
    vintage (every province if `config.active == "all"`). The week grid
    starts one week before national weekly onsets first reach
    `START_COUNT` and ends with the week holding the vintage date.

    Parameters
    ----------
    target : pandas.DataFrame
        Rows as in `bdbv_hub_renewal.data.tidy_target`.
    delays : pandas.DataFrame or None
        Delay rows (`location`, `onset_week`, `delay`) at this vintage.
    vintage : str
        The `as_of` date.
    locations : tuple of str
        Provinces.
    table : pandas.DataFrame
        Locations table (`bdbv_hub_renewal.data.load_locations`).
    config : ModelConfig
        Model configuration.
    horizon_weeks : int
        Forecast weeks after the vintage week.
    masked : tuple of str
        Provinces whose likelihood is switched off (leave one out). They
        stay in the latent process and their delays are left out.
    bvd : pandas.DataFrame, optional
        BVDOutbreakSize rows from `bvd_rows` for the release chosen for
        this vintage (`ForecastTask.auxiliary["bvd"]`).

    Returns
    -------
    Vintage
        Model inputs.
    """
    rows = target.loc[
        (target["vintage"] == vintage) & target["location"].isin(locations)
    ]
    wide = rows.pivot_table(
        index="target_end_date",
        columns="location",
        values=["observed", "imputed", "total"],
        aggfunc="sum",
    ).fillna(0)
    cases = wide["total"].sum(axis=0)
    if config.active == "all":
        active = tuple(locations)
    else:
        active = tuple(p for p in locations if cases.get(p, 0) > 0)
    idle = tuple(p for p in locations if p not in active)
    national = wide["total"].reindex(columns=list(active)).fillna(0).sum(1)
    big = int(np.argmax(national.to_numpy() >= START_COUNT))
    first = _saturday(national.index[max(0, big - 1)])
    start = first - dt.timedelta(days=6)
    cutoff = (pd.Timestamp(vintage).date() - start).days
    n_weeks = cutoff // 7 + 1
    weeks = [
        (first + dt.timedelta(weeks=i)).isoformat() for i in range(n_weeks)
    ]

    def grid(column: str, locs: tuple[str, ...]) -> np.ndarray:
        out = wide[column].reindex(index=weeks, columns=list(locs))
        return out.fillna(0).to_numpy(dtype=float)

    counts = np.zeros((n_weeks, config.dmax + 1))
    pmf = np.full(config.dmax + 1, 1 / (config.dmax + 1))
    if delays is not None and len(delays):
        pmf = hub_delay_pmf(delays, config.dmax)
        keep = delays.loc[
            (delays["delay"] >= 0)
            & (delays["delay"] <= config.dmax)
            & delays["location"].isin(active)
            & ~delays["location"].isin(masked)
        ]
        week = np.array(
            [
                (pd.Timestamp(w).date() - first).days // 7
                for w in keep["onset_week"]
            ],
            dtype=int,
        )
        ok = (week >= 0) & (week < n_weeks)
        np.add.at(counts, (week[ok], keep["delay"].to_numpy()[ok]), 1)

    known, imputed = grid("observed", active), grid("imputed", active)
    pops = table.set_index("location").loc[list(active), "population"]
    names = list(active)
    if config.national:
        known = known.sum(axis=1, keepdims=True)
        imputed = imputed.sum(axis=1, keepdims=True)
        pops = pd.Series([pops.sum()])
        names = [NATIONAL]
    if len(names) == 1:
        kernel = None
    elif config.coupling == "adjacency":
        kernel = adjacency_kernel(np.ones((len(active), len(active))))
    else:
        kernel = gravity_kernel(active, table)
    import_kernel, inactive = None, None
    if idle and not config.national:
        import_kernel = gravity_kernel(active + idle, table)[
            : len(active), len(active) :
        ]
        inactive = grid("total", idle)
    return Vintage(
        start_date=start,
        cutoff=cutoff,
        known=known,
        imputed=imputed,
        delay_counts=counts,
        hub_delay_pmf=pmf,
        populations=pops.to_numpy(dtype=float),
        kernel=kernel,
        horizon_weeks=horizon_weeks,
        names=names,
        inactive=inactive,
        inactive_names=[] if config.national else list(idle),
        import_kernel=import_kernel,
        fit_mask=np.array([p not in masked for p in names]),
        bvd=None
        if bvd is None or config.national
        else bvd_inputs(bvd, names, start, masked),
    )


def ppc_rows(
    data: Vintage, insample: Mapping[str, np.ndarray]
) -> tuple[pd.DataFrame, np.ndarray]:
    """Observed rows and the matching in-sample predictive draws.

    One row per location, stream and data week for the streams the model
    fits (`observed` and `imputed`, or `total`), plus national rows as
    the sum of the province draws within each draw. These are the
    `observed` and `draws` arguments of
    `bdbv_hub_renewal.ppc.summarise_draws`.

    Parameters
    ----------
    data : Vintage
        The fitted vintage.
    insample : mapping
        `forecast_trajectories(...)["insample"]`: draws of the vintage
        counts by stream, `(n_draws, n_weeks + horizon, P)`.

    Returns
    -------
    tuple
        `observed` (columns `location`, `stream`, `target_end_date`,
        `observed`) and `draws`, `(n_draws, len(observed))`.
    """
    first = data.start_date + dt.timedelta(days=6)
    weeks = [
        (first + dt.timedelta(weeks=i)).isoformat() for i in range(data.n_weeks)
    ]
    data_by_stream = {
        "known": data.known,
        "imputed": data.imputed,
        "total": data.known + data.imputed,
    }
    hub_name = {"known": "observed", "imputed": "imputed", "total": "total"}
    rows, cols = [], []
    for stream, x in insample.items():
        draws = x[:, : data.n_weeks]
        obs = data_by_stream[stream]
        for j, loc in enumerate(data.names):
            for i, week in enumerate(weeks):
                rows.append((loc, hub_name[stream], week, obs[i, j]))
                cols.append(draws[:, i, j])
        if data.names != [NATIONAL]:
            for i, week in enumerate(weeks):
                rows.append((NATIONAL, hub_name[stream], week, obs[i].sum()))
                cols.append(draws[:, i].sum(axis=1))
    observed = pd.DataFrame(
        rows, columns=["location", "stream", "target_end_date", "observed"]
    )
    return observed, np.column_stack(cols)


def diagnostics(
    model, config_name: str, seconds: float, sampler: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the `diagnostics.json` record of a fit.

    One row per scalar element of every sampled parameter (the
    deterministic trajectories are left out) and one row per chain.

    Returns
    -------
    dict
        Schema version 1 of `ci/INTEGRATION.md` in the main repository.
    """
    trace = jax.tree.map(
        np.asarray, model.mcmc.get_samples(group_by_chain=True)
    )
    sampled = sorted(model.mcmc.last_state.z)
    params = []
    for name in sampled:
        x = trace[name]
        flat = x.reshape(x.shape[0], x.shape[1], -1)
        rhat = split_gelman_rubin(flat) if x.shape[0] > 1 else None
        ess = effective_sample_size(flat)
        for i in range(flat.shape[2]):
            constant = bool(np.ptp(flat[:, :, i]) == 0)
            params.append(
                {
                    "name": name,
                    "index": i,
                    "rhat": None
                    if rhat is None or constant
                    else float(rhat[i]),
                    "ess_bulk": None if constant else float(ess[i]),
                }
            )
    extra = {
        k: np.asarray(v)
        for k, v in model.mcmc.get_extra_fields(group_by_chain=True).items()
    }
    depth = int(sampler.get("max_tree_depth", 10))
    chains = []
    for c in range(extra["diverging"].shape[0]):
        row = {
            "chain": c,
            "draws": int(extra["diverging"].shape[1]),
            "divergences": int(extra["diverging"][c].sum()),
        }
        if "energy" in extra:
            e = extra["energy"][c]
            row |= {
                "step_size": float(extra["adapt_state.step_size"][c][-1]),
                "tree_depth_hits": float(
                    np.mean(extra["num_steps"][c] >= 2**depth - 1)
                ),
                "mean_accept": float(extra["accept_prob"][c].mean()),
                "ebfmi": float(
                    np.sum(np.diff(e) ** 2) / np.sum((e - e.mean()) ** 2)
                ),
            }
        chains.append(row)
    worst = sorted(
        (p for p in params if p["rhat"] is not None), key=lambda p: -p["rhat"]
    )[:10]
    return {
        "schema_version": 1,
        "model": "renewal",
        "config": config_name,
        "sampler": "nuts",
        "num_chains": len(chains),
        "num_samples": int(extra["diverging"].shape[1]),
        "max_tree_depth": depth,
        "runtime_seconds": round(seconds, 1),
        "parameters": params,
        "chains": chains,
        "worst_rhat": worst,
    }


def check_fit(
    model, data: Vintage, draws: dict, max_rhat: float = 1.05
) -> list[str]:
    """Return the convergence and plausibility checks a fit fails.

    Checks: divergent transitions under 1%, split R-hat of the sampled
    parameters under `max_rhat` (with more than one chain), and the
    median national forecast under ten times the largest weekly national
    count seen so far.

    Returns
    -------
    list of str
        Failed checks, empty if all pass.
    """
    record = diagnostics(model, "check", 0.0, {})
    failures = []
    div = sum(c["divergences"] for c in record["chains"])
    n = sum(c["draws"] for c in record["chains"])
    if div / n > 0.01:
        failures.append(f"divergent share {div / n:.3f}")
    rhats = [p["rhat"] for p in record["parameters"] if p["rhat"] is not None]
    if rhats and max(rhats) > max_rhat:
        failures.append(f"max R-hat {max(rhats):.3f}")
    seen = np.nansum(data.known + data.imputed, axis=1).max()
    ahead = np.median(draws["national"][:, -data.horizon_weeks :], axis=0)
    if not np.all(np.isfinite(ahead)) or ahead.max() > 10 * max(seen, 1):
        failures.append("implausible forecast")
    return failures


def _horizon(task) -> int:
    last = max(_saturday(d) for d in task.target_end_dates)
    return max(1, (last - _saturday(task.vintage)).days // 7)


def samples_table(task, data: Vintage, draws: dict, config: ModelConfig):
    """Province sample rows and the unassigned national component.

    Modelled provinces take their trajectories, provinces with no cases
    take their importation draws, and weeks before the model grid take
    the (complete) data in every sample. Provinces not reported (when
    `config.report_inactive` is false) go to the unassigned component,
    so national sample k is always the sum of everything in sample k.

    Returns
    -------
    tuple
        Province samples (`location`, `target_end_date`, `sample`,
        `value`) and the unassigned frame (`target_end_date`, `sample`,
        `value`) or None.
    """
    first = data.start_date + dt.timedelta(days=6)
    rows = task.target.loc[task.target["vintage"] == task.vintage]
    observed = rows.pivot_table(
        index="target_end_date", columns="location", values="total"
    )
    series = {
        loc: draws["province"][:, :, j] for j, loc in enumerate(data.names)
    }
    for j, loc in enumerate(data.inactive_names):
        series[loc] = draws["inactive"][:, :, j]
    n = task.n_samples
    report = [p for p in task.locations if p in data.names]
    if config.report_inactive:
        report = list(task.locations)
    out, unassigned = [], []
    for date in task.target_end_dates:
        i = (_saturday(date) - first).days // 7
        for loc in task.locations:
            if i < 0:  # before the model grid: the complete data
                value = observed[loc].get(date, 0) if loc in observed else 0
                values = np.full(n, float(np.nan_to_num(value)))
            elif loc in series:
                values = series[loc][:, i]
            else:
                values = np.zeros(n)
            frame = pd.DataFrame(
                {
                    "location": loc,
                    "target_end_date": date,
                    "sample": np.arange(1, n + 1),
                    "value": np.asarray(values, dtype=float),
                }
            )
            (out if loc in report else unassigned).append(frame)
    samples = pd.concat(out, ignore_index=True)
    extra = None
    if unassigned:
        extra = (
            pd.concat(unassigned)
            .groupby(["target_end_date", "sample"], as_index=False)["value"]
            .sum()
        )
    return samples, extra


@dataclass
class PyRenewProvinceModel:
    """The province model behind the `bdbv_hub_renewal` model protocol.

    If `fallback` is set and the main fit fails `check_fit`, the task is
    refitted with the fallback configuration. `last_run` records which
    configuration was used and why, with the fitted model and draws.

    Attributes
    ----------
    name : str
        Model name.
    config : ModelConfig
        Structural variant.
    table : pandas.DataFrame or None
        Locations table. Defaults to `bdbv_hub_renewal.data.load_locations()`.
    fallback : ModelConfig or None
        Configuration to use when `config` fails its checks.
    num_warmup, num_samples, num_chains, max_tree_depth : int
        NUTS settings.
    masked : tuple of str
        Provinces with the likelihood switched off (leave one out).
    bvd : pandas.DataFrame or None
        BVDOutbreakSize rows (`bvd_rows`), overriding
        `task.auxiliary["bvd"]`.
    """

    name: str = "renewal"
    config: ModelConfig = MAIN
    table: pd.DataFrame | None = None
    fallback: ModelConfig | None = None
    num_warmup: int = 500
    num_samples: int = 500
    num_chains: int = 2
    max_tree_depth: int = 10
    masked: tuple[str, ...] = ()
    bvd: pd.DataFrame | None = None
    last_run: dict = field(default_factory=dict)

    def _table(self) -> pd.DataFrame:
        if self.table is None:
            from bdbv_hub_renewal.data import load_locations

            self.table = load_locations()
        return self.table

    def run(self, task, config: ModelConfig):
        """Fit one configuration to a task and draw trajectories.

        Returns
        -------
        tuple
            `(data, model, draws, seconds)`.
        """
        data = vintage_from_tables(
            task.target,
            task.delays,
            task.vintage,
            tuple(task.locations),
            self._table(),
            config,
            _horizon(task),
            self.masked,
            (
                self.bvd
                if self.bvd is not None
                else getattr(task, "auxiliary", {}).get("bvd")
            )
            if config.bvd
            else None,
        )
        model = build_model(data, config)
        start = time.perf_counter()
        fit(
            model,
            data,
            self.num_warmup,
            self.num_samples,
            self.num_chains,
            seed=task.seed,
            max_tree_depth=self.max_tree_depth,
        )
        jax.block_until_ready(model.mcmc.get_samples())
        seconds = time.perf_counter() - start
        draws = _trajectories(model, data, task.n_samples, seed=task.seed)
        return data, model, draws, seconds

    def forecast_with_unassigned(self, task):
        """Return the sample table and the unassigned national component.

        Returns
        -------
        tuple
            Province samples and the unassigned frame (or None).
        """
        config = self.config
        data, model, draws, seconds = self.run(task, config)
        self.last_run = {
            "config": config,
            "failures": check_fit(model, data, draws),
        }
        if self.last_run["failures"] and self.fallback is not None:
            config = self.fallback
            data, model, draws, seconds = self.run(task, config)
            self.last_run |= {
                "config": config,
                "fallback_failures": check_fit(model, data, draws),
            }
        self.last_run |= {
            "seconds": seconds,
            "data": data,
            "model": model,
            "draws": draws,
        }
        return samples_table(task, data, draws, config)

    def forecast(self, task) -> pd.DataFrame:
        """Return joint sample trajectories for a `ForecastTask`.

        Returns
        -------
        pandas.DataFrame
            Columns `location`, `target_end_date`, `sample`, `value`.
        """
        return self.forecast_with_unassigned(task)[0]


def fit_renewal(
    task,
    sampler: Mapping[str, Any],
    fit_dir: Path | str,
    config: ModelConfig = MAIN,
    config_name: str = "main",
    table: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """`FitConfig.fit`: fit, write trajectories, `ppc.json` and diagnostics.

    The forecast trajectories are drawn here and saved to
    `trajectories.pkl`, so `forecast_renewal` only reads them.

    Returns
    -------
    dict
        The `diagnostics.json` record, also written to `fit_dir`.
    """
    fit_dir = Path(fit_dir)
    fit_dir.mkdir(parents=True, exist_ok=True)
    runner = PyRenewProvinceModel(
        config=config,
        table=table,
        num_warmup=int(sampler.get("num_warmup", 1000)),
        num_samples=int(sampler.get("num_samples", 1000)),
        num_chains=int(sampler.get("num_chains", 4)),
        max_tree_depth=int(sampler.get("max_tree_depth", 10)),
    )
    task = replace(task, seed=int(sampler.get("seed", task.seed)))
    samples, unassigned = runner.forecast_with_unassigned(task)
    run = runner.last_run
    record = diagnostics(run["model"], config_name, run["seconds"], sampler)
    observed, draws = ppc_rows(run["data"], run["draws"]["insample"])
    write_ppc(
        summarise_draws(
            observed, draws, as_of=task.vintage, config=config_name
        ),
        fit_dir / "ppc.json",
    )
    with open(fit_dir / "trajectories.pkl", "wb") as f:
        pickle.dump({"samples": samples, "unassigned": unassigned}, f)
    (fit_dir / "diagnostics.json").write_text(json.dumps(record, indent=1))
    return record


def forecast_renewal(task, fit_dir: Path | str):
    """`FitConfig.forecast`: read the saved trajectories.

    Returns
    -------
    bdbv_hub_renewal.fitting.Forecast
        Province samples and the unassigned component.
    """
    from bdbv_hub_renewal.fitting import Forecast

    with open(Path(fit_dir) / "trajectories.pkl", "rb") as f:
        saved = pickle.load(f)
    samples = saved["samples"]
    weeks = samples["target_end_date"].isin(task.target_end_dates)
    return Forecast(samples=samples.loc[weeks], unassigned=saved["unassigned"])
