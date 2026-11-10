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
    "zone_mode",
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


ZONE_MODES: Final = ("province", "conditional", "joint")
"""How `renewal` and `renewalbvd` forecast health zones:

- `province`: province fits, their samples split by `ZoneAllocator`
  (the default);
- `conditional`: the same province fits, their samples split by a share
  model fitted to the zone counts (`zonesplit.ZoneShares`), a second
  stage as in BVDOutbreakSize's zone model;
- `joint`: zone fits (`ZoneRenewalModel`), provinces the zone sums.
"""


def zone_mode(path: Path | str = HUB_CONFIG) -> str:
    """Return how `renewal` and `renewalbvd` forecast zones (`ZONE_MODES`).

    `ZONES_ENV` if set, else `[renewal] zones` in `config/hub.toml`, else
    `province`. False is `province` and true is `joint`.

    Parameters
    ----------
    path : pathlib.Path or str
        The hub config.

    Returns
    -------
    str
        One of `ZONE_MODES`.

    Raises
    ------
    ValueError
        If the setting is not a mode or a boolean.
    """
    words = {
        "1": "joint",
        "true": "joint",
        "0": "province",
        "false": "province",
    }
    env = os.environ.get(ZONES_ENV)
    if env:
        value: object = env.lower()
    else:
        config = Path(path)
        raw = tomllib.loads(config.read_text()) if config.exists() else {}
        value = raw.get("renewal", {}).get("zones", False)
        if isinstance(value, bool):
            value = "joint" if value else "province"
    mode = words.get(str(value), str(value))
    if mode not in ZONE_MODES:
        raise ValueError(
            f"{ZONES_ENV} or [renewal] zones must be one of {ZONE_MODES} "
            f"or a boolean, got {value!r}."
        )
    return mode


def zone_fits(path: Path | str = HUB_CONFIG) -> bool:
    """Return whether `renewal` and `renewalbvd` are fitted on zones.

    True for the `joint` mode of `zone_mode`.

    Parameters
    ----------
    path : pathlib.Path or str
        The hub config.

    Returns
    -------
    bool
        True for zone fits.
    """
    return zone_mode(path) == "joint"


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


ZONE_ESTIMATES: Final = "zone_estimates.csv"
"""BVDOutbreakSize's zone Rt summaries in a release (its zone model)."""

ZONE_PREFIX: Final = "zone:"
"""Prefix of the BVD term groups of single zones (`zone_bvd_rows`)."""


def _norm(name: object) -> str:
    import unicodedata

    text = unicodedata.normalize("NFKD", str(name))
    return "".join(c for c in text if c.isalnum()).lower()


def zone_bvd_rows(
    release: Any,
    cache_dir: Path | str | None = None,
    names: pd.DataFrame | None = None,
    download: Any = None,
) -> pd.DataFrame | None:
    """BVDOutbreakSize's zone Rt of a release, as rows of the BVD term.

    Each zone in the release's `zone_estimates.csv` becomes a row like a
    patch row of `data.bvd.model_rows`: its log median and the log-scale
    SD of its 90% interval, relative to the release's national
    reference. Its group is `ZONE_PREFIX` and its hub code, and its member
    is that code. Zones are matched to hub codes by province and name
    (`health_zones.csv`); zones with no hub case yet are not matched and
    are dropped, as they are not fitted.

    Parameters
    ----------
    release : bdbvhubrenewal.data.bvd.BvdRelease
        The release chosen for the vintage.
    cache_dir : pathlib.Path or str, optional
        The release cache. Defaults to `bvd/` in the data directory. The
        file is downloaded into the release's folder if missing.
    names : pandas.DataFrame, optional
        Hub zone codes and names (`load_health_zones`).
    download : callable, optional
        Returns the bytes at a URL.

    Returns
    -------
    pandas.DataFrame or None
        Columns of `data.bvd.MODEL_ROW_COLUMNS`, or None if the release
        has no zone estimates.
    """
    from bdbvhubrenewal.data import bvd

    cache = Path(cache_dir) if cache_dir is not None else _data_dir() / "bvd"
    path = cache / release.tag / ZONE_ESTIMATES
    if not path.exists():
        url = (
            f"https://github.com/{bvd.BVD_REPO}/releases/download/"
            f"{release.tag}/{ZONE_ESTIMATES}"
        )
        try:
            body = (download or bvd._download)(url)
        except Exception:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    est = pd.read_csv(path)
    if names is None:
        from bdbvhubrenewal.data import load_health_zones

        names = load_health_zones(_data_dir())
    hub = names.drop_duplicates("location")
    key = hub["location"].str[:5] + "|" + hub["location_name"].map(_norm)
    code = dict(zip(key, hub["location"], strict=True))
    est = est.assign(
        location=(
            est["province"].map(bvd.BVD_PROVINCES)
            + "|"
            + est["label"].map(_norm)
        ).map(code)
    ).dropna(subset=["location", "R_T_median", "R_T_lower", "R_T_upper"])
    est = est.loc[(est["R_T_lower"] > 0) & (est["R_T_median"] > 0)]
    nat = bvd.model_rows(release).iloc[0]
    ref = (float(nat["ref_log_mean"]), float(nat["ref_log_sd"]))
    rows = [
        (
            ZONE_PREFIX + str(r["location"]),
            str(r["location"]),
            release.as_of.isoformat(),
            float(np.log(r["R_T_median"])),
            float(np.log(r["R_T_upper"] / r["R_T_lower"]) / (2 * bvd.Z90)),
            *ref,
        )
        for _, r in est.iterrows()
    ]
    if not rows:
        return None
    return pd.DataFrame(rows, columns=list(bvd.MODEL_ROW_COLUMNS))


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


def zone_inputs_digest(path: Path | str = HUB_CONFIG) -> str:
    """Hash the data files a zone fit reads that its code does not hold.

    The zone centres and populations (`model.zones.ZONE_CENTRES`) and the
    `[renewal]` table of the hub config. The fit key hashes code, not
    data files, so this digest sits in `ZoneRenewalModel`'s
    representation, which the key does hash: editing either refits.

    Parameters
    ----------
    path : pathlib.Path or str
        The hub config.

    Returns
    -------
    str
        The first 16 hex digits of a SHA-256.
    """
    import hashlib
    import json

    from bdbvhubrenewal.model.zones import ZONE_CENTRES

    digest = hashlib.sha256(Path(ZONE_CENTRES).read_bytes())
    config = Path(path)
    table = (
        tomllib.loads(config.read_text()).get("renewal", {})
        if config.exists()
        else {}
    )
    digest.update(json.dumps(table, sort_keys=True).encode())
    return digest.hexdigest()[:16]


@dataclass
class ZoneRenewalModel(RenewalModel):
    """A renewal variant fitted on health zones, as a `ForecastModel`.

    Attributes
    ----------
    zones : bool
        Always True: in the representation, so it keys a stored fit
        apart from the province model.
    inputs : str
        `zone_inputs_digest`, in the representation so the zone data
        files key a stored fit.
    zone_samples : pandas.DataFrame or None
        Zone samples of the last `forecast`, which sum to its province
        samples.
    """

    zones: bool = field(init=False, default=True)
    inputs: str = field(init=False, default_factory=zone_inputs_digest)
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
                zone_rows = zone_bvd_rows(release)
                if zone_rows is not None:
                    rows = pd.concat([rows, zone_rows], ignore_index=True)
                    used = {**used, "zones": len(zone_rows)}
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
