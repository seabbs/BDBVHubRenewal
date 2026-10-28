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
- `fit_renewal` and `forecast_renewal`: the `fit` and `forecast` of the
  `main` and `bvd` fit configurations
  (`bdbvhubrenewal.submission.fitting`).

This module reads only `ModelConfig`, `MAIN`, `BVD`, `build_model`,
`model_args`, `model.EXTRA_FIELDS`, `forecast_trajectories` and
`adapter.vintage_from_tables` from the subpackage, so a new model version
drops in if it keeps those. Sampling is here (`run_nuts`), so the CI's
sampler table, including jittered inits, applies in full.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from bdbvhubrenewal.forecasters.base import (
    NATIONAL,
    SAMPLE_COLUMNS,
    ForecastTask,
    register_model,
)

if TYPE_CHECKING:
    from bdbvhubrenewal.submission.fitting import Forecast

__all__ = [
    "DEFAULT_SAMPLER",
    "FIT_VARIANTS",
    "HUB_MODEL",
    "JITTER_RADIUS",
    "SAMPLER_ENV",
    "SINGLE_PATCH_MIN_CASES",
    "TRAJECTORY_DRAW_FILE",
    "VARIANTS",
    "RenewalModel",
    "RenewalRun",
    "Variant",
    "bvd_rows",
    "chain_diagnostics",
    "data_gap",
    "ebfmi",
    "engine",
    "fit_renewal",
    "fit_summary",
    "forecast_renewal",
    "horizon_weeks",
    "init_strategy",
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

SINGLE_PATCH_MIN_CASES: Final = 10
"""Fewest cases for a location to get its own single-patch fit. Below it
the location gets Poisson draws at a Gamma-distributed weekly rate
(`bdbvhubrenewal.model.inputs.idle_draws`)."""

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
    register : bool
        Register it as a model for the backtest.
    per_location : bool
        Fit each location, the nation included, on its own series, with no
        pooling or coupling. National is then its own fit.
    """

    name: str
    description: str
    base: str = "MAIN"
    overrides: Mapping[str, Any] = field(default_factory=dict)
    register: bool = True
    per_location: bool = False


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
            "Single patch: one renewal fitted to each location's series on "
            "its own, the nation included, with no pooling or coupling.",
            overrides={
                "national": True,
                "coupling": "none",
                "inactive": "none",
                "report_inactive": True,
                # One series cannot inform the share at risk.
                "depletion": False,
            },
            per_location=True,
        ),
        Variant(
            "renewalbvd",
            "Province renewal informed by BVDOutbreakSize Rt, as a tempered "
            "pseudo-observation.",
            base="BVD",
        ),
        Variant(
            "renewalhold",
            "Province renewal with Rt held at its last value past the data.",
            overrides={"future_rt": "hold"},
        ),
        Variant(
            "renewalpo",
            "Province renewal with one importation share per origin, "
            "partially pooled.",
            overrides={"importation": "per_origin"},
        ),
    )
}
"""Model variants, by name."""

FIT_VARIANTS: Final[dict[str, str]] = {
    "main": "renewal",
    "renewalsp": "renewalsp",
    "bvd": "renewalbvd",
}
"""The variant each fit configuration of the CI runs. The chain is `main`,
a retry, then `renewalsp`, then persistence. `bvd` is a shadow candidate:
fitted weekly, never submitted until it is chosen."""


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


JITTER_RADIUS: Final = 1.0
"""Half-width of the uniform jittered inits, on the unconstrained scale."""


def init_strategy(sampler: Mapping[str, Any]) -> Any:
    """Return NumPyro's init strategy for a sampler table.

    `init = "jitter"` (the CI's retry) starts each chain at its own
    uniform draw within `JITTER_RADIUS` of zero on the unconstrained
    scale. Otherwise chains start at the prior median, as in BVDOutbreakSize.

    Parameters
    ----------
    sampler : mapping
        The sampler table.

    Returns
    -------
    callable
        A NumPyro init strategy.

    Raises
    ------
    ValueError
        On an unknown `init`.
    """
    from numpyro.infer import init_to_median, init_to_uniform

    init = sampler.get("init", "median")
    if init == "jitter":
        return init_to_uniform(radius=JITTER_RADIUS)
    if init == "median":
        return init_to_median(num_samples=15)
    raise ValueError(f"Unknown init {init!r}: use 'median' or 'jitter'.")


def nuts_kernel(model_fn: Any, sampler: Mapping[str, Any]) -> Any:
    """Build the NUTS kernel for a sampler table.

    Parameters
    ----------
    model_fn : callable
        The NumPyro model.
    sampler : mapping
        `target_accept`, `max_tree_depth` and optionally `init`.

    Returns
    -------
    numpyro.infer.NUTS
        The kernel.
    """
    from numpyro.infer import NUTS

    return NUTS(
        model_fn,
        target_accept_prob=float(sampler["target_accept"]),
        max_tree_depth=int(sampler["max_tree_depth"]),
        init_strategy=init_strategy(sampler),
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
    return max(1, (last - _saturday(task.vintage)).days // 7)


def _weeks(first: dt.date, n: int) -> list[str]:
    return [(first + dt.timedelta(weeks=i)).isoformat() for i in range(n)]


def bvd_rows(
    task: ForecastTask, cache_dir: Path | str | None = None
) -> pd.DataFrame | None:
    """BVDOutbreakSize rows for a task's vintage, from the release cache.

    Parameters
    ----------
    task : ForecastTask
        The task.
    cache_dir : pathlib.Path or str, optional
        The release cache. Defaults to `bvd/` in the data directory
        (`$BDBV_DATA_DIR` or `data/`), filled by `bdbvhubrenewal bvd`.

    Returns
    -------
    pandas.DataFrame or None
        `bdbvhubrenewal.data.bvd.model_rows` of the release chosen for the
        vintage, or None if none qualifies.
    """
    from bdbvhubrenewal.data import bvd

    data_dir = Path(os.environ.get("BDBV_DATA_DIR") or "data")
    cache = Path(cache_dir) if cache_dir is not None else data_dir / "bvd"
    release = bvd.bvd_for_vintage(task.vintage, cache)
    return None if release is None else bvd.model_rows(release)


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
    from bdbvhubrenewal.data.reconstruct import publication_lag_pmf
    from bdbvhubrenewal.model.inputs import vintage_from_tables

    lag = None
    if task.delay_history is not None:
        lag = publication_lag_pmf(task.delay_history, task.vintage)
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
        publication_lag=lag,
    )


def samples_table(
    task: ForecastTask,
    data: Any,
    draws: Mapping[str, Any],
    *,
    report_all: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Province sample rows and the unassigned national component.

    Modelled provinces take their trajectories, provinces with no cases
    take their importation draws, and weeks before the model grid take
    the data in every sample. With `report_all` false, provinces the
    model did not fit go to the unassigned component, so national sample
    k is always the sum of everything in sample k.

    Parameters
    ----------
    task : ForecastTask
        The task.
    data : bdbvhubrenewal.model.Vintage
        The fitted inputs.
    draws : mapping
        `forecast_trajectories` output: `province`, `inactive`,
        `national`, each `(n, n_weeks + horizon, ...)`.
    report_all : bool
        Report every province of the task.

    Returns
    -------
    tuple
        Province samples (columns `SAMPLE_COLUMNS`) and the unassigned
        frame (`target_end_date`, `sample`, `value`) or None.
    """
    first = data.start_date + dt.timedelta(days=6)
    rows = task.target.loc[task.target["vintage"] == task.vintage]
    observed = rows.pivot_table(
        index="target_end_date", columns="location", values="total"
    )
    series = {
        loc: np.asarray(draws["province"])[:, :, j]
        for j, loc in enumerate(data.names)
    }
    for j, loc in enumerate(data.inactive_names):
        series[loc] = np.asarray(draws["inactive"])[:, :, j]
    report = list(task.locations)
    if not report_all:
        report = [p for p in task.locations if p in data.names]
    n = task.n_samples
    out: list[pd.DataFrame] = []
    unassigned: list[pd.DataFrame] = []
    for date in task.target_end_dates:
        i = (_saturday(date) - first).days // 7
        for loc in task.locations:
            if i < 0:  # before the model grid: the data
                value = observed[loc].get(date, 0) if loc in observed else 0
                values = np.full(n, float(np.nan_to_num(value)))
            elif loc in series:
                values = series[loc][:n, i]
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
    samples = pd.concat(out, ignore_index=True).loc[:, list(SAMPLE_COLUMNS)]
    extra = None
    if unassigned:
        extra = (
            pd.concat(unassigned)
            .groupby(["target_end_date", "sample"], as_index=False)
            .agg(value=("value", "sum"))
        )
    return samples, extra


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
        The vintage date (ISO) by its day on the model axis, the `as_of`
        cut-offs of `forecast_trajectories`.
    """
    return {
        (dt.date.fromisoformat(d) - data.start_date).days: d
        for d in task.report_by
    }


def _as_reported(
    task: ForecastTask, run: RenewalRun, report_all: bool
) -> list[pd.DataFrame]:
    """Province samples of the counts reported by each later vintage."""
    out = []
    for cut, date in report_cutoffs(task, run.data).items():
        weeks = tuple(w for w in task.target_end_dates if w <= date)
        if not weeks:
            continue
        sub = replace(task, target_end_dates=weeks)
        part, _ = samples_table(
            sub, run.data, run.draws["as_reported"][cut], report_all=report_all
        )
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


def _idle_samples(
    task: ForecastTask, loc: str, rows: pd.DataFrame
) -> pd.DataFrame:
    """Draws for a location with too few cases for its own fit."""
    from bdbvhubrenewal.model.inputs import idle_draws

    counts = rows.loc[rows["location"] == loc, "total"].to_numpy(dtype=float)
    n, weeks = task.n_samples, list(task.target_end_dates)
    draws = idle_draws(counts, len(weeks), n, task.rng())
    return pd.DataFrame(
        {
            "location": loc,
            "target_end_date": np.repeat(weeks, n),
            "sample": np.tile(np.arange(1, n + 1), len(weeks)),
            "value": draws.T.reshape(-1),
        }
    )


def _add_unassigned(
    a: pd.DataFrame | None, b: pd.DataFrame | None
) -> pd.DataFrame | None:
    if a is None or b is None:
        return b if a is None else a
    both = pd.concat([a, b], ignore_index=True)
    return both.groupby(["target_end_date", "sample"], as_index=False).agg(
        value=("value", "sum")
    )


def ppc_rows(
    data: Any, insample: Mapping[str, Any], *, add_national: bool = True
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
    add_national : bool
        Add national rows as the sum of the province draws. A
        single-location fit has none.

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
        if add_national and list(data.names) != [NATIONAL]:
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
    """

    variant: Variant
    data: Any
    draws: Mapping[str, Any]
    trace: Mapping[str, Any]
    extra: Mapping[str, Any]
    seconds: float


@dataclass(frozen=True)
class RenewalFit:
    """The fits of a variant to a task and the samples they give.

    Attributes
    ----------
    runs : dict
        `RenewalRun` by location for a per-location variant, or one run,
        `"all"`.
    samples : pandas.DataFrame
        Province samples (`SAMPLE_COLUMNS`).
    unassigned : pandas.DataFrame or None
        National cases in no reported province.
    national : pandas.DataFrame or None
        The national fit's samples, for a per-location variant.
    as_reported : pandas.DataFrame or None
        Samples of the counts reported by each date in `task.report_by`
        (`SAMPLE_COLUMNS` and `as_of`), for the weeks up to that date.
        A per-location variant includes its national fit.
    """

    runs: Mapping[str, RenewalRun]
    samples: pd.DataFrame
    unassigned: pd.DataFrame | None = None
    national: pd.DataFrame | None = None
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
        The model honours `ForecastTask.mask`. A single-patch variant fits
        the national sum, so it cannot.
    name : str
        The variant name.
    """

    variant: Variant
    sampler: dict[str, Any] = field(default_factory=sampler_settings)
    progress: Path | None = None
    supports_mask: bool = field(init=False)
    name: str = field(init=False)
    fit_diagnostics: list[dict[str, Any]] = field(
        init=False, default_factory=list
    )

    def __post_init__(self) -> None:
        """Name the model after its variant and say if it takes a mask."""
        self.name = self.variant.name
        self.supports_mask = not (
            self.variant.per_location
            or self.variant.overrides.get("national", False)
        )

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
        rows = bvd_rows(task) if config.bvd else None
        if config.bvd:
            tag = "none" if rows is None else rows["as_of"].iloc[0]
            self._log(f"variant={self.name} bvd_as_of={tag}")
        data = vintage_inputs(task, config, bvd=rows)
        model = pr.build_model(data, config)
        s = self.sampler
        self._log(f"variant={self.name} sampling start")
        start = time.perf_counter()
        run_nuts(model, data, {"seed": task.seed, **s})
        # Pulling the draws to the host waits for the sampler.
        np.asarray(next(iter(model.mcmc.get_samples().values())))
        seconds = time.perf_counter() - start
        self._log(f"variant={self.name} sampling done seconds={seconds:.0f}")
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
        return RenewalRun(self.variant, data, draws, trace, extra, seconds)

    def forecast_fit(self, task: ForecastTask) -> RenewalFit:
        """Fit the variant to a task and return its samples.

        A per-location variant fits each location with cases, the nation
        included, on its own. Its national samples are that fit, and
        there is no unassigned part. Otherwise the provinces are fitted
        together and the unassigned part holds the provinces not reported
        and the data's national gap (`data_gap`).

        Parameters
        ----------
        task : ForecastTask
            The task.

        Returns
        -------
        RenewalFit
            The runs, province samples and national or unassigned parts.
        """
        report_all = bool(model_config(self.variant).report_inactive)
        if not self.variant.per_location:
            run = self.run(task)
            samples, unassigned = samples_table(
                task, run.data, run.draws, report_all=report_all
            )
            unassigned = _add_unassigned(unassigned, data_gap(task, samples))
            later = _as_reported(task, run, report_all)
            return RenewalFit(
                {"all": run},
                samples,
                unassigned,
                as_reported=pd.concat(later, ignore_index=True)
                if later
                else None,
            )
        rows = task.target.loc[task.target["vintage"] == task.vintage]
        cases = rows.groupby("location")["total"].sum()
        runs: dict[str, RenewalRun] = {}
        frames: list[pd.DataFrame] = []
        reported: list[pd.DataFrame] = []
        for loc in (*task.locations, NATIONAL):
            if cases.get(loc, 0) <= 0:
                continue
            if cases.get(loc, 0) < SINGLE_PATCH_MIN_CASES:
                idle = _idle_samples(task, loc, rows)
                frames.append(idle)
                # Too few cases to model reporting: the same draws.
                reported.extend(
                    idle.loc[idle["target_end_date"] <= d].assign(as_of=d)
                    for d in task.report_by
                )
                continue
            sub = replace(task, locations=(loc,), mask=())
            runs[loc] = self.run(sub)
            part, _ = samples_table(
                sub, runs[loc].data, runs[loc].draws, report_all=True
            )
            frames.append(part)
            reported.extend(_as_reported(sub, runs[loc], report_all=True))
        both = pd.concat(frames, ignore_index=True)
        nat = both["location"] == NATIONAL
        national = both.loc[nat, ["target_end_date", "sample", "value"]]
        return RenewalFit(
            runs,
            both.loc[~nat].reset_index(drop=True),
            None,
            national.reset_index(drop=True) if nat.any() else None,
            pd.concat(reported, ignore_index=True) if reported else None,
        )

    def forecast(self, task: ForecastTask) -> pd.DataFrame:
        """Return joint sample trajectories for a task.

        Parameters
        ----------
        task : ForecastTask
            The task.

        Returns
        -------
        pandas.DataFrame
            Columns `SAMPLE_COLUMNS`, with national rows for a
            per-location variant, whose national is its own fit. With
            `task.report_by`, also `as_of` rows (`RenewalFit.as_reported`).
        """
        fit = self.forecast_fit(task)
        self.fit_diagnostics = [
            {"fit": key, **fit_summary(run, self.sampler)}
            for key, run in fit.runs.items()
        ]
        samples = fit.samples
        if fit.national is not None:
            national = fit.national.assign(location=NATIONAL)
            samples = pd.concat(
                [samples, national[list(SAMPLE_COLUMNS)]], ignore_index=True
            )
        if not task.report_by:
            return samples
        later = [] if fit.as_reported is None else [fit.as_reported]
        return pd.concat(
            [samples.assign(as_of=np.nan), *later], ignore_index=True
        )


SAMPLES_FILE: Final = "samples.csv.gz"
UNASSIGNED_FILE: Final = "unassigned.csv.gz"
NATIONAL_FILE: Final = "national.csv.gz"
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
    Writes `samples.csv.gz`, `unassigned.csv.gz` (if any),
    `trajectories.json` and `ppc.json`. `run_fit` writes the returned
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
        The `diagnostics.json` record (schema 1, with `worst_rhat`).
    """
    from bdbvhubrenewal.evaluation.ppc import summarise_draws, write_ppc

    out = Path(fit_dir)
    out.mkdir(parents=True, exist_ok=True)
    settings = sampler_settings(sampler)
    task = replace(task, seed=int(settings.get("seed", task.seed)))
    model = RenewalModel(VARIANTS[variant], settings, out / "progress.log")
    fit = model.forecast_fit(task)
    fit.samples.to_csv(out / SAMPLES_FILE, index=False)
    if fit.unassigned is not None:
        fit.unassigned.to_csv(out / UNASSIGNED_FILE, index=False)
    if fit.national is not None:
        fit.national.to_csv(out / NATIONAL_FILE, index=False)
    pairs = [
        pd.DataFrame(
            {
                "location": loc,
                "sample": np.arange(1, len(run.draws["draw"]) + 1),
                "draw": np.asarray(run.draws["draw"], dtype=int),
            }
        )
        for loc, run in fit.runs.items()
        if "draw" in run.draws
    ]
    if pairs:
        pd.concat(pairs, ignore_index=True).to_csv(
            out / TRAJECTORY_DRAW_FILE, index=False
        )
    modelled = sorted(set(fit.samples["location"]))
    (out / TRAJECTORIES_FILE).write_text(
        json.dumps({"variant": variant, "modelled": modelled}, indent=1) + "\n"
    )
    per_location = VARIANTS[variant].per_location
    observed, draws = [], []
    for run in fit.runs.values():
        obs, dr = ppc_rows(
            run.data, run.draws["insample"], add_national=not per_location
        )
        observed.append(obs)
        draws.append(dr)
    write_ppc(
        summarise_draws(
            pd.concat(observed, ignore_index=True),
            np.column_stack(draws),
            as_of=task.vintage,
            model=HUB_MODEL,
            config=config_name,
        ),
        out / "ppc.json",
    )
    depth = int(settings["max_tree_depth"])
    params: list[dict[str, Any]] = []
    chains: list[dict[str, Any]] = []
    for loc, run in fit.runs.items():
        prefix = f"{loc}:" if per_location else ""
        for row in parameter_diagnostics(run.trace):
            params.append({**row, "name": prefix + row["name"]})
        for row in chain_diagnostics(run.extra, depth):
            extra = {"location": loc} if per_location else {}
            chains.append({**row, **extra, "chain": len(chains)})
    return {
        "schema_version": 1,
        "model": HUB_MODEL,
        "config": config_name,
        "variant": variant,
        "sampler": "nuts",
        "num_chains": len(chains),
        "num_samples": chains[0]["draws"] if chains else 0,
        "max_tree_depth": depth,
        "sampling_seconds": round(sum(r.seconds for r in fit.runs.values()), 1),
        "parameters": params,
        "chains": chains,
        "worst_rhat": worst_rhat(params),
        "warnings": lag_warnings([run.data for run in fit.runs.values()]),
    }


def lag_warnings(inputs: Sequence[Any]) -> list[str]:
    """Warn when a fit used the prior publication lag, not an estimate.

    Parameters
    ----------
    inputs : sequence of bdbvhubrenewal.model.Vintage
        The model inputs of each run.

    Returns
    -------
    list of str
        One `name: message` warning, or none.
    """
    if any(getattr(v, "publication_lag_pmf", None) is None for v in inputs):
        return [
            "publication_lag: no earlier pair of delay files, so the fit "
            "used PRIOR_PUBLICATION_LAG"
        ]
    return []


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
    national = None
    if (path / NATIONAL_FILE).exists():
        national = pd.read_csv(
            path / NATIONAL_FILE, dtype={"target_end_date": str}
        )
        national = national.loc[
            national["target_end_date"].isin(task.target_end_dates)
        ].reset_index(drop=True)
    meta = json.loads((path / TRAJECTORIES_FILE).read_text())
    return Forecast(
        national=national,
        samples=samples.reset_index(drop=True),
        unassigned=unassigned,
        modelled=tuple(meta["modelled"]),
    )


def _factory(variant: Variant) -> RenewalModel:
    return RenewalModel(variant)


for _variant in VARIANTS.values():
    if _variant.register:
        register_model(
            _variant.name,
            partial(_factory, _variant),
            _variant.description,
            fitted=True,
        )
