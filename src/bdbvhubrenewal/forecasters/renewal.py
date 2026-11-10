# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""The PyRenew renewal model behind the repository's interfaces.

The model itself (JAX, NumPyro, PyRenew) is in `bdbvhubrenewal.model`.
It is imported lazily, so this module and the registry work without the
`model` extra.
This module owns everything around it:

- the variants (`VARIANTS`), registered as models for the backtest;
- `vintage_inputs`: a `ForecastTask` as the model's `Vintage`, with the
  leave-one-province-out mask;
- `samples_table`: trajectories as the hub sample table, with the
  unassigned national component;
- `ppc_rows`, `parameter_diagnostics`, `chain_diagnostics`: the CI's
  `ppc.json` and `diagnostics.json`;
- `delay_summary` and `fit_delays`: each fit's learnt delay, written as
  `delay-summary.csv` beside its forecasts;
- `fit_renewal` and `forecast_renewal`: the `fit` and `forecast` of the
  `main` and `bvd` fit configurations
  (`bdbvhubrenewal.submission.fitting`).

This module reads only `ModelConfig`, `MAIN`, `BVD`, `build_model`,
`model_args`, `model.EXTRA_FIELDS`, `forecast_trajectories` and
`adapter.vintage_from_tables` from the subpackage, so a new model version
drops in if it keeps those. Sampling is here (`run_nuts`), so the CI's
sampler table applies in full.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import multiprocessing
import os
import time
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from bdbvhubrenewal.data.tidy import write_csv
from bdbvhubrenewal.forecasters.base import (
    NATIONAL,
    SAMPLE_COLUMNS,
    ForecastTask,
    register_model,
)

if TYPE_CHECKING:
    from bdbvhubrenewal.data.bvd import BvdRelease
    from bdbvhubrenewal.submission.fitting import Forecast

__all__ = [
    "DEFAULT_SAMPLER",
    "DELAY_SUMMARY_FILE",
    "FIT_VARIANTS",
    "HUB_MODEL",
    "SAMPLER_ENV",
    "TRAJECTORY_DRAW_FILE",
    "VARIANTS",
    "RenewalModel",
    "RenewalRun",
    "Variant",
    "bvd_record",
    "bvd_release",
    "bvd_rows",
    "chain_diagnostics",
    "data_gap",
    "delay_summary",
    "ebfmi",
    "engine",
    "fit_delays",
    "fit_renewal",
    "fit_summary",
    "forecast_renewal",
    "horizon_weeks",
    "model_config",
    "nuts_kernel",
    "parameter_diagnostics",
    "ppc_rows",
    "report_cutoffs",
    "run_nuts",
    "sampler_settings",
    "samples_table",
    "vintage_inputs",
    "worst_rhat",
]

HUB_MODEL: Final = "renewal"
"""The model every fit configuration reports in its records.

It is the internal key. The hub id is set apart, in `config/hub.toml`."""

SAMPLER_ENV: Final = "BDBV_RENEWAL_SAMPLER"
"""Environment variable with JSON sampler settings for registered models."""

DEFAULT_SAMPLER: Final[Mapping[str, Any]] = {
    "num_chains": 4,
    "num_warmup": 1000,
    "num_samples": 1000,
    "target_accept": 0.9,
    "max_tree_depth": 10,
    "seed": 1,
    # Parallel chains (pmap over host devices) leak memory on Linux CPU
    # with JAX 0.12; vectorized chains stay flat, about a third slower.
    "chain_method": "vectorized",
}
"""NUTS settings when neither the CI nor `SAMPLER_ENV` gives them."""

PROVINCE_WORKERS_ENV: Final = "BDBV_PROVINCE_WORKERS"
"""Environment variable with the number of provinces a per-province
variant fits at once (`province_workers`)."""

MAX_PROVINCE_WORKERS: Final = 4
"""Most provinces fitted at once by default, so a large shared host is
not filled by one fit."""


def _cores() -> int:
    """Return the CPUs this process may run on."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - not Linux
        return os.cpu_count() or 1


def province_workers(num_chains: int, n_provinces: int) -> int:
    """Return how many provinces a per-province variant fits at once.

    `PROVINCE_WORKERS_ENV` if set, else one fit per `num_chains` cores,
    at most `MAX_PROVINCE_WORKERS`: with parallel chains (the policy's
    setting) each chain then has a core. Never more than the provinces.

    Parameters
    ----------
    num_chains : int
        Chains of each fit, run in parallel.
    n_provinces : int
        Provinces to fit.

    Returns
    -------
    int
        At least 1.

    Raises
    ------
    ValueError
        If `PROVINCE_WORKERS_ENV` is not a positive integer.
    """
    env = os.environ.get(PROVINCE_WORKERS_ENV)
    if env:
        if not env.isdigit() or int(env) < 1:
            raise ValueError(
                f"{PROVINCE_WORKERS_ENV} must be a positive integer, "
                f"got {env!r}."
            )
        workers = int(env)
    else:
        workers = min(MAX_PROVINCE_WORKERS, _cores() // max(1, num_chains))
    return max(1, min(workers, n_provinces))


RECENT_WEEKS: Final = 4
"""Data weeks behind the share that carries the national data gap forward."""


def engine() -> ModuleType:
    """Import the PyRenew model subpackage.

    Returns
    -------
    module
        `bdbvhubrenewal.model`.
    """
    from bdbvhubrenewal import model

    return model


@dataclass(frozen=True)
class Variant:
    """A model variant: a base configuration and what it changes.

    Attributes
    ----------
    name : str
        Registry key and hub model abbreviation.
    description : str
        One line for tables and docs.
    base : str
        `MAIN` or `BVD`, the configuration in `bdbvhubrenewal.model`.
    overrides : mapping
        `ModelConfig` fields to change.
    per_province : bool
        Fit each province on its own series, in a loop, with the same
        model code and no pooling or coupling across provinces.
    """

    name: str
    description: str
    base: str = "MAIN"
    overrides: Mapping[str, Any] = field(default_factory=dict)
    per_province: bool = False


VARIANTS: Final[dict[str, Variant]] = {
    v.name: v
    for v in (
        Variant(
            "renewal",
            "Province renewal (PyRenew): pooled Rt, gravity importation, "
            "depletion, modelled delays and imputation.",
        ),
        Variant(
            "renewalsp",
            "Single patch: renewal fitted to each province's series on its "
            "own, with no pooling or coupling; national is their sum.",
            # One series cannot inform the share at risk.
            overrides={"national": True, "depletion": False},
            per_province=True,
        ),
        Variant(
            "renewalbvd",
            "Province renewal informed by BVDOutbreakSize Rt, as a tempered "
            "pseudo-observation.",
            base="BVD",
        ),
    )
}
"""Model variants, by name."""

FIT_VARIANTS: Final[dict[str, str]] = {
    "main": "renewal",
    "renewalsp": "renewalsp",
    "bvd": "renewalbvd",
}
"""The variant each fit configuration of the CI runs. The chain is `bvd`,
`renewalsp`, `main`, then the random walks (`ci/policy.toml`)."""


def model_config(variant: Variant) -> Any:
    """Build the `ModelConfig` of a variant.

    Parameters
    ----------
    variant : Variant
        The variant.

    Returns
    -------
    bdbvhubrenewal.model.ModelConfig
        Its base configuration with the overrides applied.
    """
    return dataclasses.replace(
        getattr(engine(), variant.base), **dict(variant.overrides)
    )


def sampler_settings(
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return NUTS settings.

    `DEFAULT_SAMPLER`, then any JSON in the `SAMPLER_ENV` environment
    variable, then `overrides`.

    Parameters
    ----------
    overrides : mapping, optional
        Settings that win, such as the CI's sampler table.

    Returns
    -------
    dict
        The settings.
    """
    env = json.loads(os.environ.get(SAMPLER_ENV) or "{}")
    return {**DEFAULT_SAMPLER, **env, **dict(overrides or {})}


def nuts_kernel(model_fn: Any, sampler: Mapping[str, Any]) -> Any:
    """Build the NUTS kernel for a sampler table.

    Each chain starts from its own prior draw, so chains start apart and
    R-hat can show a chain that stays in another mode.

    Parameters
    ----------
    model_fn : callable
        The NumPyro model.
    sampler : mapping
        `target_accept` and `max_tree_depth`.

    Returns
    -------
    numpyro.infer.NUTS
        The kernel.
    """
    from numpyro.infer import NUTS, init_to_sample

    return NUTS(
        model_fn,
        target_accept_prob=float(sampler["target_accept"]),
        max_tree_depth=int(sampler["max_tree_depth"]),
        init_strategy=init_to_sample,
    )


def run_nuts(model: Any, data: Any, sampler: Mapping[str, Any]) -> Any:
    """Fit a built model by NUTS, keeping the fields the diagnostics need.

    Parameters
    ----------
    model : pyrenew.model.MultiSignalModel
        From `build_model`.
    data : bdbvhubrenewal.model.Vintage
        The inputs.
    sampler : mapping
        The full sampler table (`sampler_settings`).

    Returns
    -------
    pyrenew.model.MultiSignalModel
        The model, with `model.mcmc` holding the posterior.
    """
    import jax
    from jax import random
    from numpyro.infer import MCMC

    # Each fit compiles its own model, so cached executables from earlier
    # fits never help and exhaust LLVM memory over many locations.
    jax.clear_caches()  # type: ignore[no-untyped-call]
    pr = engine()
    args = pr.model_args(model, data)
    model.validate_data(
        **{k: v for k, v in args.items() if k != "population_size"}
    )
    model.mcmc = MCMC(
        nuts_kernel(model.sample, sampler),
        num_warmup=int(sampler["num_warmup"]),
        num_samples=int(sampler["num_samples"]),
        num_chains=int(sampler["num_chains"]),
        chain_method=str(sampler.get("chain_method", "vectorized")),
        progress_bar=False,
    )
    model.mcmc.run(
        random.key(int(sampler["seed"])),
        extra_fields=pr.model.EXTRA_FIELDS,
        **args,
    )
    return model


def _saturday(day: str | dt.date) -> dt.date:
    d = pd.Timestamp(day).date()
    return d + dt.timedelta(days=(5 - d.weekday()) % 7)


def horizon_weeks(task: ForecastTask) -> int:
    """Weeks the model must run past the week holding the vintage.

    Parameters
    ----------
    task : ForecastTask
        The task.

    Returns
    -------
    int
        At least one.
    """
    last = max(_saturday(d) for d in task.target_end_dates)
    return max(1, (last - _saturday(task.cutoff)).days // 7)


def _weeks(first: dt.date, n: int) -> list[str]:
    return [(first + dt.timedelta(weeks=i)).isoformat() for i in range(n)]


def bvd_release(
    task: ForecastTask, cache_dir: Path | str | None = None
) -> BvdRelease | None:
    """Return the BVDOutbreakSize release chosen for a task's vintage.

    The newest release that existed at the vintage's cut-off and whose
    joint fit passed the R-hat gate (`bdbvhubrenewal.data.bvd`).

    Parameters
    ----------
    task : ForecastTask
        The task.
    cache_dir : pathlib.Path or str, optional
        The release cache. Defaults to `bvd/` in the data directory
        (`$BDBV_DATA_DIR` or `data/`), filled by `bdbvhubrenewal bvd`.

    Returns
    -------
    bdbvhubrenewal.data.bvd.BvdRelease or None
        The release with its choice, or None if none qualifies.
    """
    from bdbvhubrenewal.data import bvd

    data_dir = Path(os.environ.get("BDBV_DATA_DIR") or "data")
    cache = Path(cache_dir) if cache_dir is not None else data_dir / "bvd"
    return bvd.bvd_for_vintage(task.vintage, cache)


def bvd_rows(
    task: ForecastTask, cache_dir: Path | str | None = None
) -> pd.DataFrame | None:
    """BVDOutbreakSize rows for a task's vintage, from the release cache.

    Parameters
    ----------
    task : ForecastTask
        The task.
    cache_dir : pathlib.Path or str, optional
        The release cache, as in `bvd_release`.

    Returns
    -------
    pandas.DataFrame or None
        `bdbvhubrenewal.data.bvd.model_rows` of the release chosen for the
        vintage, or None if none qualifies.
    """
    from bdbvhubrenewal.data import bvd

    release = bvd_release(task, cache_dir)
    return None if release is None else bvd.model_rows(release)


def bvd_record(release: BvdRelease | None) -> dict[str, Any]:
    """Describe the BVDOutbreakSize release used, for the fit diagnostics.

    Parameters
    ----------
    release : bdbvhubrenewal.data.bvd.BvdRelease or None
        From `bvd_release`.

    Returns
    -------
    dict
        `tag`, `as_of`, the joint fit's `joint_max_rhat`, the `max_rhat`
        limit and the newer releases `skipped` with their reasons, or
        only `tag: None` if no release qualified.
    """
    if release is None:
        return {"tag": None}
    record: dict[str, Any] = {
        "tag": release.tag,
        "as_of": release.as_of.isoformat(),
    }
    if release.choice is not None:
        record.update(release.choice.record())
    return record


def vintage_inputs(
    task: ForecastTask,
    config: Any,
    table: pd.DataFrame | None = None,
    bvd: pd.DataFrame | None = None,
) -> Any:
    """Turn a forecast task into the model's `Vintage`.

    A thin call to the model's own `vintage_from_tables`, which chooses
    the active set (the provinces with cases) and the week grid. Provinces
    in `task.mask` have their likelihood switched off and their delays
    left out. They stay in the latent process, so they are still
    forecast.

    Parameters
    ----------
    task : ForecastTask
        The task.
    config : bdbvhubrenewal.model.ModelConfig
        The model configuration.
    table : pandas.DataFrame, optional
        The locations table (populations). Defaults to
        `bdbvhubrenewal.data.load_locations()`.
    bvd : pandas.DataFrame, optional
        BVDOutbreakSize rows (`bvd_rows`), for a variant with the BVD
        term. Without them that variant is the main model.

    Returns
    -------
    bdbvhubrenewal.model.Vintage
        The model inputs.
    """
    from bdbvhubrenewal.data import load_locations
    from bdbvhubrenewal.data.releases import release_pairs
    from bdbvhubrenewal.model.inputs import vintage_from_tables

    pairs = None
    if task.delay_history is not None:
        pairs = release_pairs(task.delay_history, task.vintage)
    return vintage_from_tables(
        task.target,
        task.delays,
        task.vintage,
        tuple(task.locations),
        load_locations() if table is None else table,
        config,
        horizon_weeks(task),
        tuple(task.mask),
        bvd=bvd,
        release_pairs=pairs,
    )


def samples_table(
    task: ForecastTask,
    data: Any,
    draws: Mapping[str, Any],
) -> pd.DataFrame:
    """Province sample rows of the modelled provinces.

    Weeks before the model grid take the data in every sample.

    Parameters
    ----------
    task : ForecastTask
        The task.
    data : bdbvhubrenewal.model.Vintage
        The fitted inputs.
    draws : mapping
        `forecast_trajectories` output, with `province`
        `(n, n_weeks + horizon, P)`.

    Returns
    -------
    pandas.DataFrame
        Province samples, columns `SAMPLE_COLUMNS`.
    """
    first = data.start_date + dt.timedelta(days=6)
    rows = task.target.loc[task.target["vintage"] == task.vintage]
    observed = rows.pivot_table(
        index="target_end_date", columns="location", values="total"
    )
    n = task.n_samples
    out: list[pd.DataFrame] = []
    for date in task.target_end_dates:
        i = (_saturday(date) - first).days // 7
        for j, loc in enumerate(data.names):
            if i < 0:  # before the model grid: the data
                value = observed[loc].get(date, 0) if loc in observed else 0
                values = np.full(n, float(np.nan_to_num(value)))
            else:
                values = np.asarray(draws["province"])[:n, i, j]
            frame = pd.DataFrame(
                {
                    "location": loc,
                    "target_end_date": date,
                    "sample": np.arange(1, n + 1),
                    "value": np.asarray(values, dtype=float),
                }
            )
            out.append(frame)
    return pd.concat(out, ignore_index=True).loc[:, list(SAMPLE_COLUMNS)]


def _silent_walk(task: ForecastTask) -> tuple[pd.DataFrame, set[str]]:
    """Return `randomwalknowcast`'s draws of the silent places, and them.

    The walk finds the places with no case in its walk window and gives
    them its introduction layer, pooled over all the task's places.
    """
    from bdbvhubrenewal.forecasters.randomwalk import RandomWalkNowcast

    walk = RandomWalkNowcast()
    origin = walk.origin_at(task, errors=False)
    silent = {
        str(loc)
        for loc, quiet in zip(origin.snap.locations, origin.silent, strict=True)
        if quiet and loc in task.locations
    }
    draws = walk.forecast(replace(task, report_by={}))
    return draws.loc[
        draws["location"].isin(silent), list(SAMPLE_COLUMNS)
    ], silent


def report_cutoffs(task: ForecastTask, data: Any) -> dict[int, str]:
    """Days from the model start to each later vintage in `task.report_by`.

    Parameters
    ----------
    task : ForecastTask
        The task.
    data : bdbvhubrenewal.model.Vintage
        The fitted inputs.

    Returns
    -------
    dict
        The vintage date (ISO) by its cut-off on the model axis, the
        `as_of` cut-offs of `forecast_trajectories`.
    """
    return {
        (dt.date.fromisoformat(cut) - data.start_date).days: d
        for d, cut in task.report_by.items()
    }


def _as_reported(task: ForecastTask, run: RenewalRun) -> list[pd.DataFrame]:
    """Province samples of the counts reported by each later vintage."""
    out = []
    for cut, date in report_cutoffs(task, run.data).items():
        weeks = tuple(w for w in task.target_end_dates if w <= date)
        if not weeks:
            continue
        sub = replace(task, target_end_dates=weeks)
        part = samples_table(sub, run.data, run.draws["as_reported"][cut])
        out.append(part.assign(as_of=date))
    return out


def data_gap(task: ForecastTask, samples: pd.DataFrame) -> pd.DataFrame | None:
    r"""National cases in the data that no province has, by week and sample.

    In weeks with data, the gap is $g(w) = \max\{0, Y_{CD}(w) - \sum_p
    Y_p(w)\}$ from the vintage, the same in every sample. Later weeks carry
    the recent share forward: $g_k(w) = \rho \sum_p \tilde y_{p,k}(w)$, with
    $\rho$ the gap over the provinces' total in the last `RECENT_WEEKS`
    data weeks and $\tilde y_{p,k}$ sample $k$ of province $p$.

    Parameters
    ----------
    task : ForecastTask
        The task.
    samples : pandas.DataFrame
        Province samples (`SAMPLE_COLUMNS`).

    Returns
    -------
    pandas.DataFrame or None
        Columns `target_end_date`, `sample` and `value`, or None if the
        national data equal the sum of the provinces.
    """
    rows = task.target.loc[task.target["vintage"] == task.vintage]
    wide = rows.pivot_table(
        index="target_end_date", columns="location", values="total"
    ).fillna(0)
    if NATIONAL not in wide:
        return None
    provinces = wide.drop(columns=NATIONAL).sum(axis=1)
    gap = (wide[NATIONAL] - provinces).clip(lower=0)
    if not (gap > 0).any():
        return None
    recent = gap.index[-RECENT_WEEKS:]
    total = float(provinces.loc[recent].sum())
    share = float(gap.loc[recent].sum()) / total if total > 0 else 0.0
    by = samples.groupby(["target_end_date", "sample"], as_index=False).agg(
        value=("value", "sum")
    )
    last = str(gap.index.max())
    later = by["target_end_date"].astype(str) > last
    known = by["target_end_date"].astype(str).map(gap).fillna(0.0)
    by["value"] = np.where(later, share * by["value"], known)
    return by.reset_index(drop=True)


def ppc_rows(
    data: Any, insample: Mapping[str, Any]
) -> tuple[pd.DataFrame, NDArray[np.float64]]:
    """Observed rows and the matching in-sample predictive draws.

    One row per location, stream and data week for the streams the model
    fits (`observed` and `imputed`, or `total`), plus national rows as
    the sum of the province draws within each draw. These are the
    `observed` and `draws` arguments of
    `bdbvhubrenewal.evaluation.ppc.summarise_draws`.

    Parameters
    ----------
    data : bdbvhubrenewal.model.Vintage
        The fitted inputs.
    insample : mapping
        Draws of the vintage counts by stream (`known`, `imputed` or
        `total`), each `(n_draws, n_weeks + horizon, P)`.

    Returns
    -------
    tuple
        `observed` (columns `location`, `stream`, `target_end_date`,
        `observed`) and `draws`, `(n_draws, len(observed))`.
    """
    first = data.start_date + dt.timedelta(days=6)
    weeks = _weeks(first, data.n_weeks)
    by_stream = {
        "known": data.known,
        "imputed": data.imputed,
        "total": data.known + data.imputed,
    }
    hub_name = {"known": "observed", "imputed": "imputed", "total": "total"}
    rows: list[tuple[str, str, str, float]] = []
    cols: list[NDArray[np.float64]] = []
    for stream, x in insample.items():
        draws = np.asarray(x, dtype=float)[:, : data.n_weeks]
        obs = by_stream[stream]
        for j, loc in enumerate(data.names):
            for i, week in enumerate(weeks):
                rows.append((loc, hub_name[stream], week, float(obs[i, j])))
                cols.append(draws[:, i, j])
        if len(data.names) > 1:
            for i, week in enumerate(weeks):
                rows.append(
                    (NATIONAL, hub_name[stream], week, float(obs[i].sum()))
                )
                cols.append(draws[:, i].sum(axis=1))
    observed = pd.DataFrame(
        rows, columns=["location", "stream", "target_end_date", "observed"]
    )
    return observed, np.column_stack(cols)


def ebfmi(energy: NDArray[np.float64]) -> float:
    r"""Energy Bayesian fraction of missing information of one chain.

    Parameters
    ----------
    energy : numpy.ndarray
        Hamiltonian energy of each draw.

    Returns
    -------
    float
        $\sum_n (E_n - E_{n-1})^2 / \sum_n (E_n - \bar E)^2$. Below 0.3
        suggests the momentum resampling explores the energy poorly.
    """
    e = np.asarray(energy, dtype=float)
    return float(np.sum(np.diff(e) ** 2) / np.sum((e - e.mean()) ** 2))


DELAY_SUMMARY_FILE: Final = "delay-summary.csv"
"""The learnt delay of each fit, written beside its forecasts."""

_DELAY_QUANTILES: Final = (0.05, 0.5, 0.95)


def delay_summary(
    beta: NDArray[np.floating],
    shift: NDArray[np.floating],
    first_day: dt.date,
    last_day: int,
) -> pd.DataFrame:
    r"""Summarise a fit's onset-to-report delay over posterior draws.

    The delay pmf of onset day $s$ is `hazard_to_pmf` of
    $\beta(d) + \gamma(s)$, as in `HazardDelay`, from the recorded
    `delay_beta` and `delay_shift` draws.

    Parameters
    ----------
    beta : numpy.ndarray
        `delay_beta` draws, `(draws, dmax)`.
    shift : numpy.ndarray
        `delay_shift` draws, `(draws, days)`, on the model axis.
    first_day : datetime.date
        Date of the model axis's first day.
    last_day : int
        Index on the axis of the last onset day at the vintage.

    Returns
    -------
    pandas.DataFrame
        Columns `kind`, `x`, `quantity` and `q5`, `q50`, `q95` over the
        draws. `kind` `"cumulative"` has, by delay day `x`, the share
        reported by then (`reported_share`) for onsets in the last week.
        `kind` `"onset_week"` has, by onset week ending `x`, the median
        delay in days (`median_delay`), the mean delay in days
        (`mean_delay`) and the share reported within 7 days
        (`share_within_7`).
    """
    import jax.numpy as jnp

    from bdbvhubrenewal.model.hazard_delay import hazard_to_pmf

    days = pd.date_range(first_day, periods=last_day + 1, freq="D")
    # The last day of each onset week (to Saturday) up to the vintage.
    week_end = days + pd.to_timedelta((5 - days.dayofweek) % 7, unit="D")
    last = (
        pd.Series(np.arange(len(days)), index=week_end).groupby(level=0).max()
    )
    logit = beta[:, None, :] + shift[:, last.to_numpy()][:, :, None]
    cdf = np.cumsum(np.asarray(hazard_to_pmf(jnp.asarray(logit))), axis=-1)
    qs = list(_DELAY_QUANTILES)

    def rows(
        values: NDArray[np.floating], x: list[Any], kind: str, name: str
    ) -> pd.DataFrame:
        q = np.quantile(values, qs, axis=0)
        return pd.DataFrame(
            {
                "kind": kind,
                "x": x,
                "quantity": name,
                "q5": q[0],
                "q50": q[1],
                "q95": q[2],
            }
        )

    weeks = [d.date().isoformat() for d in last.index]
    return pd.concat(
        [
            rows(
                cdf[:, -1, :],
                list(range(cdf.shape[-1])),
                "cumulative",
                "reported_share",
            ),
            rows(
                (cdf >= 0.5).argmax(axis=-1).astype(float),
                weeks,
                "onset_week",
                "median_delay",
            ),
            rows(cdf[:, :, 7], weeks, "onset_week", "share_within_7"),
            rows((1 - cdf).sum(axis=-1), weeks, "onset_week", "mean_delay"),
        ],
        ignore_index=True,
    )


def fit_delays(fit: RenewalFit, task: ForecastTask) -> pd.DataFrame:
    """Return the delay summaries of a fit's runs, as one table.

    Parameters
    ----------
    fit : RenewalFit
        The fit.
    task : ForecastTask
        Its task.

    Returns
    -------
    pandas.DataFrame
        `delay_summary` rows with `model`, `vintage` and `location`:
        each province's delay, and the shared delay of a fit of
        several provinces.
    """
    return pd.concat(
        [
            run.delay.rename(columns={"place": "location"}).assign(
                model=run.variant.name, vintage=task.vintage
            )
            for run in fit.runs.values()
        ],
        ignore_index=True,
    )[
        [
            "model",
            "vintage",
            "location",
            "kind",
            "x",
            "quantity",
            "q5",
            "q50",
            "q95",
        ]
    ]


def fit_summary(run: RenewalRun, sampler: Mapping[str, Any]) -> dict[str, Any]:
    """Summarise one fit: sampler settings, time and convergence.

    Parameters
    ----------
    run : RenewalRun
        The fit.
    sampler : mapping
        Its sampler settings.

    Returns
    -------
    dict
        `num_chains`, `num_warmup`, `num_samples`, `max_tree_depth`,
        `seconds`, `max_rhat`, `min_ess_bulk`, and the shares of draws
        that diverged (`divergent_share`) or hit the tree depth cap
        (`depth_share`).
    """
    depth = int(
        sampler.get("max_tree_depth", DEFAULT_SAMPLER["max_tree_depth"])
    )
    chains = chain_diagnostics(run.extra, depth)
    params = parameter_diagnostics(run.trace)
    rhat = [p["rhat"] for p in params if p["rhat"] is not None]
    ess = [p["ess_bulk"] for p in params if p["ess_bulk"] is not None]
    draws = sum(c["draws"] for c in chains)
    hits = sum(c.get("tree_depth_hits", 0) for c in chains)
    return {
        **{
            k: int(sampler.get(k, DEFAULT_SAMPLER[k]))
            for k in ("num_chains", "num_warmup", "num_samples")
        },
        "max_tree_depth": depth,
        "seconds": float(run.seconds),
        "max_rhat": max(rhat, default=float("nan")),
        "min_ess_bulk": min(ess, default=float("nan")),
        "divergent_share": sum(c["divergences"] for c in chains) / draws,
        "depth_share": hits / draws,
    }


def chain_diagnostics(
    extra: Mapping[str, Any], max_tree_depth: int
) -> list[dict[str, Any]]:
    """Per-chain sampler diagnostics, as BVDOutbreakSize reports them.

    Parameters
    ----------
    extra : mapping
        NUTS extra fields by chain, `(n_chains, n_draws)`: `diverging`,
        and, if kept, `num_steps`, `accept_prob`, `energy` and
        `adapt_state.step_size`.
    max_tree_depth : int
        The tree depth cap.

    Returns
    -------
    list of dict
        One row per chain: `chain`, `draws`, `divergences`, and
        `step_size`, `tree_depth_hits` (draws at the cap),
        `tree_depth_share`, `mean_accept` and `ebfmi` when available.
    """
    x = {k: np.asarray(v) for k, v in extra.items()}
    diverging = x["diverging"]
    chains = []
    for c in range(diverging.shape[0]):
        row: dict[str, Any] = {
            "chain": c,
            "draws": int(diverging.shape[1]),
            "divergences": int(diverging[c].sum()),
        }
        if "num_steps" in x:
            hits = x["num_steps"][c] >= 2**max_tree_depth - 1
            row["tree_depth_hits"] = int(hits.sum())
            row["tree_depth_share"] = float(hits.mean())
        if "adapt_state.step_size" in x:
            row["step_size"] = float(x["adapt_state.step_size"][c][-1])
        if "accept_prob" in x:
            row["mean_accept"] = float(x["accept_prob"][c].mean())
        if "energy" in x:
            row["ebfmi"] = ebfmi(x["energy"][c])
        chains.append(row)
    return chains


def _rank_normal(x: NDArray[np.float64]) -> NDArray[np.float64]:
    from scipy import stats

    ranks = stats.rankdata(x, axis=None).reshape(x.shape)
    return np.asarray(stats.norm.ppf((ranks - 0.375) / (x.size + 0.25)))


def parameter_diagnostics(
    trace: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """R-hat and effective sample sizes of every sampled scalar.

    Rank-normalised split R-hat and bulk ESS, and tail ESS as the smaller
    ESS of the 5% and 95% quantile indicators (Vehtari et al. 2021).
    Undefined values, such as for a constant, are None.

    Parameters
    ----------
    trace : mapping
        Draws of each sampled site by chain, `(n_chains, n_draws, ...)`.

    Returns
    -------
    list of dict
        One row per scalar element, sites in name order: `name`,
        `index` (flat, 0 for a scalar), `rhat`, `ess_bulk` and `ess_tail`.
    """
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin

    def none_if_bad(value: float) -> float | None:
        return float(value) if np.isfinite(value) else None

    params = []
    for name in sorted(trace):
        x = np.asarray(trace[name], dtype=float)
        flat = x.reshape(x.shape[0], x.shape[1], -1)
        for i in range(flat.shape[2]):
            draws = flat[:, :, i]
            row: dict[str, Any] = {"name": name, "index": i}
            if np.ptp(draws) == 0:
                row |= {"rhat": None, "ess_bulk": None, "ess_tail": None}
            else:
                z = _rank_normal(draws)
                lo, hi = np.quantile(draws, [0.05, 0.95])
                tails = [
                    effective_sample_size((draws <= q).astype(float))
                    for q in (lo, hi)
                ]
                row |= {
                    "rhat": none_if_bad(split_gelman_rubin(z)),
                    "ess_bulk": none_if_bad(effective_sample_size(z)),
                    "ess_tail": none_if_bad(min(tails)),
                }
            params.append(row)
    return params


def worst_rhat(
    params: list[dict[str, Any]], n: int = 10
) -> list[dict[str, Any]]:
    """Return the `n` parameters with the highest R-hat.

    Parameters
    ----------
    params : list of dict
        `parameter_diagnostics` rows.
    n : int
        How many to keep.

    Returns
    -------
    list of dict
        The rows, worst first.
    """
    rated = [p for p in params if p["rhat"] is not None]
    return sorted(rated, key=lambda p: -float(p["rhat"]))[:n]


@dataclass(frozen=True)
class RenewalRun:
    """One fit of a variant to a task.

    Attributes
    ----------
    variant : Variant
        The variant.
    data : bdbvhubrenewal.model.Vintage
        The model inputs.
    draws : dict
        `forecast_trajectories` output.
    trace : dict
        Sampled sites by chain.
    extra : dict
        NUTS extra fields by chain.
    seconds : float
        Sampling time.
    cpu_seconds : float
        Process CPU time over the same span, across all threads. Near
        `num_chains` times `seconds` when the chains run in parallel.
    delay : pandas.DataFrame
        `delay_summary` of each province of the fit (`place`), and the
        shared delay of a fit of several provinces.
    bvd : dict or None
        The BVDOutbreakSize release used (`bvd_record`), for a variant
        with the BVD term.
    """

    variant: Variant
    data: Any
    draws: Mapping[str, Any]
    trace: Mapping[str, Any]
    extra: Mapping[str, Any]
    seconds: float
    cpu_seconds: float
    delay: pd.DataFrame
    bvd: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class RenewalFit:
    """The fits of a variant to a task and the samples they give.

    Attributes
    ----------
    runs : dict
        `RenewalRun` by province for a per-province variant, else the
        one run of the provinces together, `"all"`.
    samples : pandas.DataFrame
        Province samples (`SAMPLE_COLUMNS`).
    unassigned : pandas.DataFrame or None
        National cases in no reported province.
    as_reported : pandas.DataFrame or None
        Samples of the counts reported by each date in `task.report_by`
        (`SAMPLE_COLUMNS` and `as_of`), for the weeks up to that date.
    """

    runs: Mapping[str, RenewalRun]
    samples: pd.DataFrame
    unassigned: pd.DataFrame | None = None
    as_reported: pd.DataFrame | None = None


@dataclass
class RenewalModel:
    """A renewal variant as a `ForecastModel`.

    Attributes
    ----------
    variant : Variant
        The variant.
    sampler : dict
        NUTS settings (`sampler_settings`).
    progress : pathlib.Path or None
        File to append progress lines to.
    supports_mask : bool
        The model honours `ForecastTask.mask`.
    name : str
        The variant name.
    delay_summary : pandas.DataFrame or None
        `fit_delays` of the last `forecast`.
    """

    variant: Variant
    sampler: dict[str, Any] = field(default_factory=sampler_settings)
    progress: Path | None = None
    supports_mask: bool = field(init=False, default=True)
    name: str = field(init=False)
    fit_diagnostics: list[dict[str, Any]] = field(
        init=False, default_factory=list
    )
    delay_summary: pd.DataFrame | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        """Name the model after its variant and say if it takes a mask."""
        self.name = self.variant.name
        # A province fitted on its own cannot be forecast without its data.
        self.supports_mask = not self.variant.per_province

    def _log(self, line: str) -> None:
        if self.progress is not None:
            with self.progress.open("a") as fh:
                fh.write(line + "\n")

    def run(self, task: ForecastTask) -> RenewalRun:
        """Fit the variant to a task and draw trajectories.

        Parameters
        ----------
        task : ForecastTask
            The task.

        Returns
        -------
        RenewalRun
            The fit and its draws.
        """
        pr = engine()
        config = model_config(self.variant)
        rows, used = None, None
        if config.bvd:
            from bdbvhubrenewal.data import bvd

            release = bvd_release(task)
            used = bvd_record(release)
            if release is not None:
                rows = bvd.model_rows(release)
            rhat = used.get("joint_max_rhat")
            self._log(
                f"variant={self.name} bvd={used['tag'] or 'none'} "
                f"bvd_as_of={used.get('as_of', 'none')} "
                f"bvd_joint_rhat={'none' if rhat is None else f'{rhat:.3f}'}"
            )
        data = vintage_inputs(task, config, bvd=rows)
        model = pr.build_model(data, config)
        s = self.sampler
        self._log(f"variant={self.name} sampling start")
        start, cpu_start = time.perf_counter(), time.process_time()
        run_nuts(model, data, {"seed": task.seed, **s})
        # Pulling the draws to the host waits for the sampler.
        np.asarray(next(iter(model.mcmc.get_samples().values())))
        seconds = time.perf_counter() - start
        cpu_seconds = time.process_time() - cpu_start
        self._log(
            f"variant={self.name} sampling done seconds={seconds:.0f} "
            f"cpu_seconds={cpu_seconds:.0f}"
        )
        draws = pr.forecast_trajectories(
            model,
            data,
            task.n_samples,
            seed=task.seed,
            as_of=tuple(report_cutoffs(task, data)),
        )
        by_chain = model.mcmc.get_samples(group_by_chain=True)
        trace = {
            k: np.asarray(by_chain[k])
            for k in model.mcmc.last_state.z
            if k in by_chain
        }
        extra = {
            k: np.asarray(v)
            for k, v in model.mcmc.get_extra_fields(group_by_chain=True).items()
        }
        posterior = model.mcmc.get_samples()
        n_init = model.latent.n_initialization_points
        beta = np.asarray(posterior["delay_beta"])
        # Each province's delay, and the shared one they pool towards
        # when a fit has more than one.
        shifts = {
            name: np.asarray(posterior["delay_shift"])[..., p]
            for p, name in enumerate(data.names)
        }
        if len(shifts) > 1:
            shifts["shared"] = np.asarray(posterior["delay_shared_shift"])[
                ..., 0
            ]
        delay = pd.concat(
            [
                delay_summary(
                    beta,
                    shift,
                    data.start_date - dt.timedelta(days=n_init),
                    n_init + data.cutoff,
                ).assign(place=place)
                for place, shift in shifts.items()
            ],
            ignore_index=True,
        )
        return RenewalRun(
            self.variant,
            data,
            draws,
            trace,
            extra,
            seconds,
            cpu_seconds,
            delay,
            used,
        )

    def forecast_fit(self, task: ForecastTask) -> RenewalFit:
        """Fit the variant to a task and return its samples.

        A per-province variant fits each province on its own, apart from
        the provinces the random walks find silent (no case in their
        walk window), which take `randomwalknowcast`'s draws: exactly 0
        until an introduction, then its walk. Otherwise the provinces are
        fitted together. National is the sum of the provinces plus the
        unassigned part, the national cases no province has
        (`data_gap`).

        Parameters
        ----------
        task : ForecastTask
            The task.

        Returns
        -------
        RenewalFit
            The runs, province samples and national or unassigned parts.
        """
        frames: list[pd.DataFrame] = []
        later: list[pd.DataFrame] = []
        tasks = {"all": task}
        if self.variant.per_province:
            walk, silent = _silent_walk(task)
            frames.append(walk)
            later.extend(
                walk.loc[walk["target_end_date"] <= d].assign(as_of=d)
                for d in task.report_by
            )
            tasks = {
                loc: replace(task, locations=(loc,), mask=())
                for loc in task.locations
                if loc not in silent
            }
        runs = self._run_all(tasks)
        for key, run in runs.items():
            frames.append(samples_table(tasks[key], run.data, run.draws))
            later.extend(_as_reported(tasks[key], run))
        samples = pd.concat(frames, ignore_index=True)
        return RenewalFit(
            runs,
            samples,
            data_gap(task, samples),
            as_reported=pd.concat(later, ignore_index=True) if later else None,
        )

    def _run_all(self, tasks: Mapping[str, ForecastTask]) -> dict[str, Any]:
        """Run every task, largest first, `province_workers` at a time.

        Returns
        -------
        dict
            `RenewalRun` by key, in the order of `tasks`.
        """

        def size(key: str) -> float:
            target = tasks[key].target
            own = target["location"].isin(tasks[key].locations)
            return float(target.loc[own, "total"].sum())

        order = sorted(tasks, key=size, reverse=True)
        chains = int(self.sampler.get("num_chains", 1))
        workers = province_workers(chains, len(tasks))
        if workers == 1:
            done = {key: self.run(tasks[key]) for key in order}
        else:
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(workers, mp_context=context) as pool:
                futures = {
                    key: pool.submit(self.run, tasks[key]) for key in order
                }
                try:
                    done = {key: f.result() for key, f in futures.items()}
                except BaseException:
                    # Fail now, not after the queued fits have run.
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise
        return {key: done[key] for key in tasks}

    def forecast(self, task: ForecastTask) -> pd.DataFrame:
        """Return joint sample trajectories for a task.

        Parameters
        ----------
        task : ForecastTask
            The task.

        Returns
        -------
        pandas.DataFrame
            Columns `SAMPLE_COLUMNS`. With `task.report_by`, also `as_of`
            rows (`RenewalFit.as_reported`).

        Examples
        --------
        Fitting takes minutes per task on a CPU.

        >>> from bdbvhubrenewal.forecasters import get_model
        >>> model = get_model("renewal")
        >>> samples = model.forecast(task)
        >>> model.fit_diagnostics  # divergences and R-hat of the fit
        """
        fit = self.forecast_fit(task)
        self.delay_summary = fit_delays(fit, task)
        self.fit_diagnostics = [
            {"fit": key, **fit_summary(run, self.sampler)}
            for key, run in fit.runs.items()
        ]
        samples = fit.samples
        if not task.report_by:
            return samples
        later = [] if fit.as_reported is None else [fit.as_reported]
        return pd.concat(
            [samples.assign(as_of=np.nan), *later], ignore_index=True
        )


SAMPLES_FILE: Final = "samples.csv.gz"
UNASSIGNED_FILE: Final = "unassigned.csv.gz"
TRAJECTORY_DRAW_FILE: Final = "trajectory_draw.csv"
"""The posterior draw behind each trajectory, by fit, so a later stage
(health zones) can pair trajectories with posterior draws."""
TRAJECTORIES_FILE: Final = "trajectories.json"


def fit_renewal(
    task: ForecastTask,
    sampler: Mapping[str, Any],
    fit_dir: Path | str,
    *,
    variant: str,
    config_name: str,
) -> dict[str, Any]:
    """Fit a variant for the CI and write the fit directory.

    Trajectories are drawn here, so `forecast_renewal` only reads them.
    Writes `samples.csv.gz`, `delay-summary.csv`, `unassigned.csv.gz`
    (if any), `trajectories.json` and `ppc.json`. `run_fit` writes the returned
    record to `diagnostics.json`.

    Parameters
    ----------
    task : ForecastTask
        The task.
    sampler : mapping
        The candidate's sampler table from `ci/policy.toml`.
    fit_dir : pathlib.Path or str
        The fit directory.
    variant : str
        Key in `VARIANTS`.
    config_name : str
        The fit configuration, for the records.

    Returns
    -------
    dict
        The `diagnostics.json` record (schema 1, with `worst_rhat`, and
        `bvd`, the BVDOutbreakSize release used, for a variant with the
        BVD term).
    """
    from bdbvhubrenewal.evaluation.ppc import summarise_draws, write_ppc

    out = Path(fit_dir)
    out.mkdir(parents=True, exist_ok=True)
    settings = sampler_settings(sampler)
    task = replace(task, seed=int(settings.get("seed", task.seed)))
    model = RenewalModel(VARIANTS[variant], settings, out / "progress.log")
    fit = model.forecast_fit(task)
    write_csv(fit.samples, out / SAMPLES_FILE)
    write_csv(fit_delays(fit, task), out / DELAY_SUMMARY_FILE)
    if fit.unassigned is not None:
        write_csv(fit.unassigned, out / UNASSIGNED_FILE)
    pairs = [
        pd.DataFrame(
            {
                "location": key,
                "sample": np.arange(1, len(run.draws["draw"]) + 1),
                "draw": np.asarray(run.draws["draw"], dtype=int),
            }
        )
        for key, run in fit.runs.items()
        if "draw" in run.draws
    ]
    if pairs:
        write_csv(
            pd.concat(pairs, ignore_index=True), out / TRAJECTORY_DRAW_FILE
        )
    modelled = sorted(set(fit.samples["location"]))
    (out / TRAJECTORIES_FILE).write_text(
        json.dumps({"variant": variant, "modelled": modelled}, indent=1) + "\n"
    )
    by_run = [
        ppc_rows(run.data, run.draws["insample"]) for run in fit.runs.values()
    ]
    write_ppc(
        summarise_draws(
            pd.concat([obs for obs, _ in by_run], ignore_index=True),
            np.column_stack([dr for _, dr in by_run]),
            as_of=task.vintage,
            model=HUB_MODEL,
            config=config_name,
        ),
        out / "ppc.json",
    )
    depth = int(settings["max_tree_depth"])
    several = len(fit.runs) > 1
    params: list[dict[str, Any]] = []
    chains: list[dict[str, Any]] = []
    for key, run in fit.runs.items():
        prefix = f"{key}:" if several else ""
        for row in parameter_diagnostics(run.trace):
            params.append({**row, "name": prefix + row["name"]})
        for row in chain_diagnostics(run.extra, depth):
            extra = {"location": key} if several else {}
            chains.append({**row, **extra, "chain": len(chains)})
    record: dict[str, Any] = {
        "schema_version": 1,
        "model": HUB_MODEL,
        "config": config_name,
        "variant": variant,
        "sampler": "nuts",
        "num_chains": len(chains),
        "num_samples": chains[0]["draws"] if chains else 0,
        "max_tree_depth": depth,
        "sampling_seconds": round(sum(r.seconds for r in fit.runs.values()), 1),
        "sampling_cpu_seconds": round(
            sum(r.cpu_seconds for r in fit.runs.values()), 1
        ),
        "parameters": params,
        "chains": chains,
        "worst_rhat": worst_rhat(params),
    }
    if model_config(VARIANTS[variant]).bvd:
        # The BVDOutbreakSize release behind the BVD term and its R-hat.
        used = [r.bvd for r in fit.runs.values() if r.bvd is not None]
        record["bvd"] = dict(used[0]) if used else bvd_record(None)
    return record


def forecast_renewal(task: ForecastTask, fit_dir: Path | str) -> Forecast:
    """Read the trajectories `fit_renewal` saved.

    Parameters
    ----------
    task : ForecastTask
        The task.
    fit_dir : pathlib.Path or str
        The fit directory.

    Returns
    -------
    bdbvhubrenewal.submission.fitting.Forecast
        Province samples, the unassigned component and the reported
        provinces.
    """
    from bdbvhubrenewal.submission.fitting import Forecast

    path = Path(fit_dir)
    samples = pd.read_csv(path / SAMPLES_FILE, dtype={"target_end_date": str})
    samples = samples.loc[
        samples["target_end_date"].isin(task.target_end_dates)
    ]
    unassigned = None
    if (path / UNASSIGNED_FILE).exists():
        unassigned = pd.read_csv(
            path / UNASSIGNED_FILE, dtype={"target_end_date": str}
        )
        unassigned = unassigned.loc[
            unassigned["target_end_date"].isin(task.target_end_dates)
        ].reset_index(drop=True)
    meta = json.loads((path / TRAJECTORIES_FILE).read_text())
    return Forecast(
        samples=samples.reset_index(drop=True),
        unassigned=unassigned,
        modelled=tuple(meta["modelled"]),
    )


def _factory(variant: Variant) -> RenewalModel:
    return RenewalModel(variant)


for _variant in VARIANTS.values():
    register_model(
        _variant.name,
        partial(_factory, _variant),
        _variant.description,
        fitted=True,
    )
