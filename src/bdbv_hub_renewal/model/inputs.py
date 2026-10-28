# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Model inputs from the tidy hub tables.

- `vintage_from_tables` builds the model's `Vintage` for one data vintage.
- `province_centres`, `province_distances` and `gravity_kernel` give the
  importation kernel.
- `bvd_rows` and `bvd_inputs` turn BVDOutbreakSize estimates into the
  inputs of the BVD term.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from bdbv_hub_renewal.data.delays import hub_delay_pmf

from .model import MAIN, ModelConfig, Vintage
from .renewal import adjacency_kernel

NATIONAL = "CD"
START_COUNT = 5
"""The week grid starts one week before national weekly onsets first
reach this count. Earlier weeks are returned as observed."""


CENTRES = Path(__file__).parent.parent / "data" / "province_centres.csv"
"""Population-weighted centres of all 26 provinces (see the file header)."""


def province_centres() -> pd.DataFrame:
    """Read the province centres shipped with the package.

    Returns
    -------
    pandas.DataFrame
        Columns `location`, `latitude`, `longitude`, `worldpop`, `zones`.
    """
    return pd.read_csv(CENTRES, comment="#")


def province_distances(locations: tuple[str, ...]) -> np.ndarray:
    """Great-circle distances in km between province centres.

    Returns
    -------
    numpy.ndarray
        Symmetric, zero diagonal.

    Raises
    ------
    ValueError
        If a province has no centre.
    """
    centres = province_centres().set_index("location").reindex(list(locations))
    if centres[["latitude", "longitude"]].isna().any().any():
        raise ValueError("Missing province centre.")
    lat = np.radians(centres["latitude"].to_numpy(float))
    lon = np.radians(centres["longitude"].to_numpy(float))
    a = (
        np.sin((lat[:, None] - lat[None, :]) / 2) ** 2
        + np.cos(lat[:, None])
        * np.cos(lat[None, :])
        * np.sin((lon[:, None] - lon[None, :]) / 2) ** 2
    )
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def gravity_kernel(
    locations: tuple[str, ...], table: pd.DataFrame, decay_km: float = 300.0
) -> np.ndarray:
    r"""Row-normalised gravity kernel between provinces.

    $K_{qp} \propto N_p \exp(-D_{qp} / \text{decay})$ for $q \ne p$, as
    `bdbv_hub_renewal.data.simulate.gravity_kernel`, with populations $N$ from
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
    counts_by_week = national.to_numpy()
    threshold = min(START_COUNT, counts_by_week.max())
    big = int(np.argmax(counts_by_week >= threshold))
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
            & (
                delays["location"].isin(active)
                if active != (NATIONAL,)
                else delays["location"].ne("")
            )
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
        names = [active[0] if len(active) == 1 else NATIONAL]
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
        distances=None
        if config.national
        else province_distances(active + idle),
        all_populations=None
        if config.national
        else table.set_index("location")
        .loc[list(active + idle), "population"]
        .to_numpy(float),
    )


RENEWALSP = ModelConfig(
    national=True, coupling="none", inactive="none", report_inactive=True
)
"""Single patch: the renewal model fitted to one location's series."""


def idle_draws(
    counts: np.ndarray, n_weeks: int, n: int, rng: np.random.Generator
) -> np.ndarray:
    r"""Standalone importation-only draws for a province with no cases.

    Weekly counts are Poisson with rate $\lambda \sim
    \text{Gamma}(1/2 + \sum y, w)$ over the $w$ weeks seen (Jeffreys
    prior), so each trajectory has one rate.

    Returns
    -------
    numpy.ndarray
        Draws, `(n, n_weeks)`.
    """
    shape = 0.5 + np.nansum(counts)
    rate = rng.gamma(shape, 1.0 / max(len(counts), 1), size=n)
    return rng.poisson(rate[:, None] * np.ones((1, n_weeks))).astype(float)
