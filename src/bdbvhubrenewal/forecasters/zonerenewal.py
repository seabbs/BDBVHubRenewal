# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""`renewal` and `renewalbvd` fitted on health zones (opt-in).

With `[renewal] zones = true` in `config/hub.toml`, or `BDBV_RENEWAL_ZONES`
set, `enable_zones` registers zone versions of `renewal` and
`renewalbvd` in place of the province ones (`ZoneRenewalModel`). They fit
the zones with a case in the random walks' walk window in one model,
nested in their provinces (`model.zones`), give the silent zones
`randomwalknowcast`'s introduction layer, and sum the zones to the
provinces, sample by sample. National is the sum of the provinces.

Nothing the province models run imports this module, so with the setting
off their code, and their stored fit keys, do not change. With it on,
the registry maps `renewal` and `renewalbvd` to this module, so their
keys follow its code.
"""

from __future__ import annotations

import datetime as dt
import os
import time
import tomllib
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from bdbvhubrenewal.data.tidy import write_csv
from bdbvhubrenewal.forecasters.base import (
    SAMPLE_COLUMNS,
    ForecastTask,
    register_model,
)
from bdbvhubrenewal.forecasters.renewal import (
    VARIANTS,
    RenewalFit,
    RenewalModel,
    RenewalRun,
    Variant,
    _as_reported,
    _silent_walk,
    bvd_record,
    bvd_release,
    delay_summary,
    engine,
    horizon_weeks,
    model_config,
    report_cutoffs,
    run_nuts,
    samples_table,
)

__all__ = [
    "ZONES_ENV",
    "ZONES_FILE",
    "ZONE_VARIANTS",
    "ZoneRenewalFit",
    "ZoneRenewalModel",
    "enable_zones",
    "sum_zones",
    "task_zones",
    "zone_fits",
    "zone_target",
    "zone_vintage_inputs",
]

ZONE_VARIANTS: Final = ("renewal", "renewalbvd")
"""The variants that fit health zones when `zone_fits` says so."""

ZONES_ENV: Final = "BDBV_RENEWAL_ZONES"
"""Environment variable that overrides `[renewal] zones` in
`config/hub.toml`: `1` or `true` for zone fits, `0` or `false` for
province fits."""

HUB_CONFIG: Final = Path("config/hub.toml")

ZONES_FILE: Final = "zones.csv.gz"
"""Zone samples in a zone fit's directory, summing to its provinces."""


def zone_fits(path: Path | str = HUB_CONFIG) -> bool:
    """Return whether `renewal` and `renewalbvd` fit health zones.

    `ZONES_ENV` if set, else `[renewal] zones` in `config/hub.toml`,
    else False: the province fits stay the default until the setting is
    flipped.

    Parameters
    ----------
    path : pathlib.Path or str
        The hub config.

    Returns
    -------
    bool
        True for zone fits.

    Raises
    ------
    ValueError
        If `ZONES_ENV` is not a boolean word.
    """
    env = os.environ.get(ZONES_ENV)
    if env:
        words = {"1": True, "true": True, "0": False, "false": False}
        if env.lower() not in words:
            raise ValueError(f"{ZONES_ENV} must be 1, 0, true or false.")
        return words[env.lower()]
    config = Path(path)
    if not config.exists():
        return False
    raw = tomllib.loads(config.read_text())
    return bool(raw.get("renewal", {}).get("zones"))


def _data_dir() -> Path:
    return Path(os.environ.get("BDBV_DATA_DIR") or "data")


def _daily_zones() -> pd.DataFrame:
    from bdbvhubrenewal.data import load_daily_zones

    return load_daily_zones(_data_dir())


def task_zones(task: ForecastTask) -> tuple[str, ...]:
    """Return the health zones of a task's provinces.

    The hub's zone codes (`hub-config/tasks.json` in the data directory),
    or the zones in the daily zone table if there is no hub config.

    Parameters
    ----------
    task : ForecastTask
        A province task.

    Returns
    -------
    tuple of str
        Zone codes, sorted.
    """
    from bdbvhubrenewal.data.zones import hub_zones, province_of

    tasks_json = _data_dir() / "hub-config" / "tasks.json"
    if tasks_json.exists():
        codes = hub_zones(tasks_json)
    else:
        codes = tuple(sorted(set(_daily_zones()["location"])))
    return tuple(z for z in codes if province_of(z) in task.locations)


def zone_target(task: ForecastTask) -> pd.DataFrame:
    """Weekly known and imputed onsets of each zone at a task's vintage.

    Summed by epi week from the daily zone table, as the province target
    is from the daily province table.

    Parameters
    ----------
    task : ForecastTask
        A province task.

    Returns
    -------
    pandas.DataFrame
        Columns `vintage`, `location`, `target_end_date`, `observed`,
        `imputed`, `total` and `cutoff`, for zone weeks with an onset.
    """
    daily = _daily_zones()
    rows = daily.loc[daily["vintage"].astype(str) == task.vintage]
    onset = pd.to_datetime(rows["onset_date"])
    end = onset + pd.to_timedelta((5 - onset.dt.day_of_week) % 7, unit="D")
    out = (
        rows.assign(target_end_date=end.dt.date.astype(str))
        .groupby(["location", "target_end_date"], as_index=False)[
            ["observed", "imputed"]
        ]
        .sum()
    )
    out["total"] = out["observed"] + out["imputed"]
    out["vintage"] = task.vintage
    out["cutoff"] = str(rows["cutoff"].iloc[0]) if len(rows) else task.cutoff
    return out.loc[out["total"] > 0].reset_index(drop=True)


def _zone_walk_task(task: ForecastTask) -> ForecastTask:
    """Return the random walks' zone task, which finds the silent zones."""
    from bdbvhubrenewal.data.zones import weekly_zone_counts
    from bdbvhubrenewal.forecasters.zones import zone_task

    daily = _daily_zones()
    counts = weekly_zone_counts(daily, task.vintage)
    history = (
        daily
        if task.daily_history is None
        else pd.concat([task.daily_history, daily], ignore_index=True)
    )
    return zone_task(task, counts, task_zones(task), history)


def sum_zones(samples: pd.DataFrame) -> pd.DataFrame:
    """Sum zone samples to their provinces, sample by sample.

    Parameters
    ----------
    samples : pandas.DataFrame
        Zone samples (`SAMPLE_COLUMNS`), optionally with `as_of`.

    Returns
    -------
    pandas.DataFrame
        Province samples with the same columns.
    """
    from bdbvhubrenewal.data.zones import province_of

    keys = [c for c in samples.columns if c not in ("location", "value")]
    out = (
        samples.assign(location=samples["location"].map(province_of))
        .groupby(["location", *keys], as_index=False, dropna=False)["value"]
        .sum()
    )
    return pd.DataFrame(out).loc[:, list(samples.columns)]


def zone_vintage_inputs(
    task: ForecastTask, config: Any, bvd: pd.DataFrame | None = None
) -> Any:
    """Turn a zone task into the zone model's `ZoneVintage`.

    Parameters
    ----------
    task : ForecastTask
        A task whose locations are zones and whose target is
        `zone_target`. Its mask names provinces.
    config : bdbvhubrenewal.model.ModelConfig
        The model configuration.
    bvd : pandas.DataFrame, optional
        BVDOutbreakSize rows (`bvd_rows`).

    Returns
    -------
    bdbvhubrenewal.model.zones.ZoneVintage
        The model inputs.
    """
    from bdbvhubrenewal.data.releases import release_pairs
    from bdbvhubrenewal.model.zones import zone_table, zone_vintage_from_tables

    pairs = None
    if task.delay_history is not None:
        pairs = release_pairs(task.delay_history, task.vintage)
    return zone_vintage_from_tables(
        task.target,
        task.delays,
        task.vintage,
        tuple(task.locations),
        zone_table(tuple(task.locations)),
        config,
        horizon_weeks(task),
        tuple(task.mask),
        bvd=bvd,
        release_pairs=pairs,
    )


@dataclass(frozen=True)
class ZoneRenewalFit(RenewalFit):
    """A `RenewalFit` with the zone samples it sums to its provinces.

    Attributes
    ----------
    zones : pandas.DataFrame or None
        Final-count samples of every zone of the task's provinces
        (`SAMPLE_COLUMNS`). They sum to `samples`.
    """

    zones: pd.DataFrame | None = None


@dataclass
class ZoneRenewalModel(RenewalModel):
    """A renewal variant fitted on health zones, as a `ForecastModel`.

    Attributes
    ----------
    zones : bool
        Always True: in the representation, so it keys a stored fit
        apart from the province model.
    zone_samples : pandas.DataFrame or None
        Zone samples of the last `forecast`, which sum to its province
        samples.
    """

    zones: bool = field(init=False, default=True)
    zone_samples: pd.DataFrame | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def run(self, task: ForecastTask) -> RenewalRun:
        """Fit the zone model to a zone task and draw trajectories.

        As `RenewalModel.run`, with `zone_vintage_inputs` and
        `build_zone_model`, and the delay summaries by province.
        """
        from bdbvhubrenewal.model.zones import build_zone_model

        pr = engine()
        config = model_config(self.variant)
        rows, used = None, None
        if config.bvd:
            from bdbvhubrenewal.data import bvd

            release = bvd_release(task)
            used = bvd_record(release)
            if release is not None:
                rows = bvd.model_rows(release)
            self._log(f"variant={self.name} bvd={used['tag'] or 'none'}")
        data = zone_vintage_inputs(task, config, bvd=rows)
        model = build_zone_model(data, config)
        self._log(
            f"variant={self.name} zones={data.n_provinces} sampling start"
        )
        start, cpu_start = time.perf_counter(), time.process_time()
        run_nuts(model, data, {"seed": task.seed, **self.sampler})
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
        # Each province's delay, which its zones share, and the shared one.
        shifts = {
            name: np.asarray(posterior["delay_shift"])[..., p]
            for p, name in enumerate(data.parent_names)
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

    def forecast_fit(self, task: ForecastTask) -> ZoneRenewalFit:
        """Fit the zones together and sum them to the provinces.

        The zones with a case in the random walks' walk window are
        fitted in one model. The rest take `randomwalknowcast`'s
        introduction layer (`_silent_walk`), so a zone with no recent
        case is exactly 0 until an introduction. Each province sample is
        the sum of its zones' samples of the same id. National is the
        sum of the provinces, with no unassigned part: the hub's
        national target has no case without a province in any release
        (`data/national-coherence.csv`).

        Parameters
        ----------
        task : ForecastTask
            A province task.

        Returns
        -------
        ZoneRenewalFit
            The run, province samples and zone samples.
        """
        walk, silent = _silent_walk(_zone_walk_task(task))
        fitted = tuple(z for z in task_zones(task) if z not in silent)
        zones = [walk]
        later = [
            walk.loc[walk["target_end_date"] <= d].assign(as_of=d)
            for d in task.report_by
        ]
        runs: dict[str, RenewalRun] = {}
        if fitted:
            zt = replace(task, target=zone_target(task), locations=fitted)
            runs = self._run_all({"all": zt})
            zones.append(samples_table(zt, runs["all"].data, runs["all"].draws))
            later.extend(_as_reported(zt, runs["all"]))
        zone_samples = pd.concat(zones, ignore_index=True)
        return ZoneRenewalFit(
            runs,
            sum_zones(zone_samples),
            None,
            as_reported=sum_zones(pd.concat(later, ignore_index=True))
            if later
            else None,
            zones=zone_samples,
        )

    def forecast(self, task: ForecastTask) -> pd.DataFrame:
        """Return province samples, keeping the zone samples.

        As `RenewalModel.forecast`, and sets `zone_samples`.
        """
        out = super().forecast(task)
        fit = self.last_fit
        self.zone_samples = getattr(fit, "zones", None)
        return out

    def write_fit_dir(
        self, task: ForecastTask, fit_dir: Path | str, config_name: str
    ) -> dict[str, Any]:
        """Write the last fit's directory, with `ZONES_FILE`.

        As `RenewalModel.write_fit_dir`, plus the zone samples, which the
        submission takes in place of a split of the provinces.
        """
        record = super().write_fit_dir(task, fit_dir, config_name)
        if self.zone_samples is not None:
            write_csv(
                self.zone_samples.loc[:, list(SAMPLE_COLUMNS)],
                Path(fit_dir) / ZONES_FILE,
            )
        return record


def zone_variant(variant: Variant) -> Variant:
    """Return a variant's description for its zone fit."""
    return replace(
        variant,
        description=variant.description.replace("Province", "Zone", 1)
        + " Health zones nested in provinces; provinces are zone sums.",
    )


def _factory(variant: Variant) -> ZoneRenewalModel:
    return ZoneRenewalModel(variant)


def enable_zones() -> None:
    """Register the zone versions of `ZONE_VARIANTS` in place of theirs."""
    for name in ZONE_VARIANTS:
        variant = zone_variant(VARIANTS[name])
        register_model(
            name,
            partial(_factory, variant),
            variant.description,
            fitted=True,
            replace=True,
        )
