# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""The renewal model with health zones as its places.

Zones are nested in provinces. Rt is
$\log R_z(t) = \log R(t) + \delta_{p(z)}(t) + \epsilon_z(t)$, with
sum-to-zero AR(1) province deviations and sum-to-zero AR(1) zone
deviations within each province (`nested_basis`). The delay and the
missing-onset share are sampled by province and given to the province's
zones (`ByParent`), and the zones' expected delay cells are summed to
provinces for the delay-file likelihood (`ZoneOnsetStreams`).

This module only extends the province model (`model.model`), and nothing
the province model runs imports it, so a province fit's code, and its
stored fit key, do not depend on it.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from jax import Array
from jax.typing import ArrayLike
from numpyro.distributions.transforms import AffineTransform
from pyrenew.ascertainment import IndependentAscertainment
from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent.temporal_processes import TemporalProcess
from pyrenew.metaclass import RandomVariable
from pyrenew.model import MultiSignalModel, PyrenewBuilder
from pyrenew.randomvariable import DistributionalVariable, TransformedVariable

from bdbvhubrenewal.data.delays import hub_delay_pmf
from bdbvhubrenewal.data.releases import release_cutoff

from .censored import CensoredPMF
from .composition_noise import MultinomialNoise
from .delays import FixedTimeVarying
from .hazard_delay import HazardDelay
from .inputs import (
    NATIONAL,
    START_COUNT,
    _saturday,
    _seed_days,
    bvd_inputs,
    province_centres,
)
from .model import (
    ALPHA0,
    FORECAST_RT_HALF_LIFE,
    GEN_INT_LENGTH,
    INCUBATION_LENGTH,
    INCUBATION_MEAN,
    INCUBATION_MEAN_SD,
    INCUBATION_SD,
    INCUBATION_SD_SD,
    MAIN,
    MAX_PUBLICATION_LAG,
    SUNDAY,
    ModelConfig,
    Vintage,
    _dm_noise,
    _named_walk,
    _nb_noise,
    _pooled,
    capped_concentration,
    fit_weeks,
    generation_interval,
)
from .observation import MissingOnsetFraction, OnsetStreams
from .pooled import (
    ActiveValues,
    PooledInitialValue,
    split_initial_value,
    sum_to_zero_basis,
)
from .publication_lag import ReleasePairLikelihood
from .renewal import Importation, MetapopulationInfections
from .rt_observation import RtPseudoObservation
from .temporal import NonCentredRandomWalk

ZONE_CENTRES = Path(__file__).parent.parent / "data" / "zone_centres.csv"
"""Centroids and WorldPop counts of the hub's health zones (see the file
header)."""

DECAY_KM = 300.0
"""Distance scale of the zone gravity kernel."""

MIN_OWN_CASES = 30
"""Cases a zone needs by the cut-off for an Rt deviation of its own, as
BVDOutbreakSize's zone model gives week-to-week changes only to zones
with at least 30 cases. Fewer cases cannot inform a zone's own Rt path."""

ANCHORED = ("rt_sd", "rt_dev_sd", "rt_zone_dev_sd", "pi_sd", "pi_dev_sd")
"""Walks anchored at their mean level over the fit axis
(`MeanAnchoredWalk`)."""


@dataclass
class ZoneVintage(Vintage):
    """A `Vintage` whose places are health zones nested in provinces.

    Attributes
    ----------
    parents : numpy.ndarray
        Index of each zone's province in `parent_names`, `(P,)`.
    parent_names : list of str
        The provinces. `delay_counts` are by province,
        `(n_weeks, len(parent_names), dmax + 1)`.
    own : numpy.ndarray or None
        Boolean `(P,)`: zones with at least `MIN_OWN_CASES` cases by the
        cut-off, which have an Rt deviation of their own within their
        province. The rest follow their province's Rt.
    """

    parents: np.ndarray | None = None
    parent_names: list[str] | None = None
    own: np.ndarray | None = None


def parent_active(parents: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Return the parents with at least one active child.

    Parameters
    ----------
    parents
        Index of each place's parent, `(n,)`.
    active
        Boolean `(n,)`.

    Returns
    -------
    numpy.ndarray
        Boolean, one per parent.
    """
    parents = np.asarray(parents, dtype=int)
    out = np.zeros(int(parents.max()) + 1, dtype=bool)
    np.logical_or.at(out, parents, np.asarray(active, dtype=bool))
    return out


def nested_basis(
    parents: np.ndarray,
    active: np.ndarray | None = None,
    own: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Bases of parent and within-parent deviations of nested places.

    For places $z$ in parents $p(z)$ (zones in provinces), a deviation is
    $\delta_z = \sum_j B_{zj} \xi_j + \sum_k Q_{zk} \eta_k$.
    `between` $B$ lifts `sum_to_zero_basis` over the parents with an
    active place to their places, so a parent's places share its
    deviation and the parent deviations sum to zero. `within` $Q$ is
    block diagonal: each parent's block is `sum_to_zero_basis` over its
    active places, so the place deviations sum to zero within each
    parent. A parent with one active place has no block. An inactive
    place takes its parent's deviation and has no own. The columns of
    $B$ and $Q$ are orthogonal, and with one place per parent $B$ is
    `sum_to_zero_basis` and $Q$ is empty.

    Parameters
    ----------
    parents
        Index of each place's parent, `(n,)`.
    active
        Boolean `(n,)`, or `None` for every place.
    own
        Boolean `(n,)`: the active places with a deviation of their own
        within their parent. `None` is every active place. The others
        take their parent's deviation.

    Returns
    -------
    tuple of numpy.ndarray
        `between`, `(n, k_p - 1)` with `k_p` the parents with an active
        place, and `within`, `(n, sum_p (k_z(p) - 1))` with `k_z(p)` the
        active places of parent `p`.
    """
    parents = np.asarray(parents, dtype=int)
    n = parents.size
    keep = np.ones(n, dtype=bool) if active is None else np.asarray(active)
    par = parent_active(parents, keep)
    between = sum_to_zero_basis(par.size, par)[parents]
    if own is not None:
        keep = keep & np.asarray(own, dtype=bool)
    blocks = []
    for p in range(par.size):
        mine = keep & (parents == p)
        if mine.sum() < 2:
            continue
        block = np.zeros((n, int(mine.sum()) - 1))
        block[mine] = sum_to_zero_basis(int(mine.sum()))
        blocks.append(block)
    within = np.hstack(blocks) if blocks else np.zeros((n, 0))
    return between, within


def _nested_walk(
    baseline_process: TemporalProcess,
    parent_process: TemporalProcess,
    place_process: TemporalProcess | None,
    base_start: ArrayLike,
    dev_start: ArrayLike,
    n_timepoints: int,
    name_prefix: str,
    first_day_dow: int | None,
    bases: tuple[np.ndarray, np.ndarray],
) -> tuple[ArrayLike, ArrayLike]:
    """`pooled_walk` with parent and within-parent deviations.

    The starting deviations are projected onto each basis by least
    squares, which for an orthonormal basis is `basis.T @ start`.
    """
    baseline = baseline_process.sample(
        n_timepoints=n_timepoints,
        initial_value=base_start,
        name_prefix=f"{name_prefix}_baseline",
        first_day_dow=first_day_dow,
    )
    out = baseline
    for basis, process, label in (
        (bases[0], parent_process, "deviations"),
        (bases[1], place_process, "zone_deviations"),
    ):
        if basis.shape[1] == 0:
            continue
        if process is None:
            raise ValueError("Nested deviations need a place process.")
        coords = process.sample(
            n_timepoints=n_timepoints,
            n_processes=basis.shape[1],
            initial_value=jnp.asarray(np.linalg.pinv(basis)) @ dev_start,
            name_prefix=f"{name_prefix}_{label}",
            first_day_dow=first_day_dow,
        )
        out = out + coords @ jnp.asarray(basis).T
    return baseline, jnp.broadcast_to(out, (n_timepoints, bases[0].shape[0]))


class MeanAnchoredWalk(NonCentredRandomWalk):
    """`NonCentredRandomWalk` whose start is its mean over the fit axis.

    The walk's fitted part is shifted so its mean is the starting value.
    The data fix a walk's level, so with the start at the first knot the
    start and the steps trade off along a ridge that NUTS crosses only
    with very long trajectories (renewal-speed's Laplace analysis of
    `renewal` at 2026-11-01: condition number 1.9e7, about 4300
    leapfrog steps). Anchoring at the mean removes that ridge.
    """

    def sample(
        self,
        n_timepoints: int,
        initial_value: float | ArrayLike | None = None,
        n_processes: int = 1,
        name_prefix: str = "rw",
        *,
        first_day_dow: int | None = None,
    ) -> Array:
        """Sample trajectories, as `NonCentredRandomWalk.sample`."""
        from .temporal import _damped_sum

        n_past = (
            n_timepoints
            if self.n_fit is None
            else min(self.n_fit, n_timepoints)
        )
        x0 = jnp.broadcast_to(
            jnp.asarray(0.0 if initial_value is None else initial_value),
            (n_processes,),
        )
        frozen = min(self.frozen, n_past - 1)
        n_free = n_past - 1 - frozen
        sd = self.innovation_sd_rv()
        phi = (
            1.0
            if self.half_life_rv is None
            else 0.5 ** (1.0 / self.half_life_rv())
        )
        z = numpyro.sample(
            f"{name_prefix}_z",
            dist.Normal(0.0, 1.0).expand([n_free, n_processes]),
        )
        free = _damped_sum(phi, sd * z)
        decay = phi ** jnp.arange(1, n_free + 1)[:, None]
        past = jnp.vstack(
            [jnp.repeat(x0[None, :], frozen + 1, axis=0), decay * x0 + free]
        )
        past = past - past.mean(axis=0) + x0
        n_future = n_timepoints - n_past
        if n_future == 0:
            return past
        z_future = numpyro.sample(
            f"{name_prefix}_z_future",
            dist.Normal(0.0, 1.0).expand([n_future, n_processes]),
        )
        if self.forecast_half_life is not None:
            phi = 0.5 ** (1.0 / self.forecast_half_life)
        decay = phi ** jnp.arange(1, n_future + 1)[:, None]
        future = decay * past[-1] + _damped_sum(phi, sd * z_future)
        return jnp.vstack([past, future])


def _zone_walk(
    name: str,
    sd: float,
    n_weeks: int,
    ar: bool = False,
    forecast_half_life: float | None = None,
    anchor: bool = True,
) -> TemporalProcess:
    """`model._named_walk`, anchored at the mean for `ANCHORED` walks."""
    from pyrenew.latent.temporal_processes import WeeklyTemporalProcess

    process = _named_walk(name, sd, n_weeks, ar, forecast_half_life)
    if not anchor or name not in ANCHORED:
        return process
    inner = process.inner
    anchored = MeanAnchoredWalk(
        inner.innovation_sd_rv,
        frozen=inner.frozen,
        n_fit=inner.n_fit,
        half_life_rv=inner.half_life_rv,
        forecast_half_life=inner.forecast_half_life,
    )
    return WeeklyTemporalProcess(anchored, start_dow=SUNDAY)


def zone_pooled_walk(
    baseline_process: TemporalProcess,
    deviation_process: TemporalProcess,
    place_process: TemporalProcess,
    initial: ArrayLike,
    n_timepoints: int,
    parents: np.ndarray,
    name_prefix: str,
    first_day_dow: int | None = None,
    active: np.ndarray | None = None,
    own: np.ndarray | None = None,
) -> tuple[ArrayLike, ArrayLike]:
    """`pooled.pooled_walk` with province and zone deviations.

    Zones outside `own` have no deviation of their own (`nested_basis`).

    Returns
    -------
    tuple of ArrayLike
        The pooled trend `(n_timepoints, 1)` and the zone trajectories
        `(n_timepoints, n_zones)`.
    """
    base_start, dev_start = split_initial_value(initial, parents.size)
    return _nested_walk(
        baseline_process,
        deviation_process,
        place_process,
        base_start,
        dev_start,
        n_timepoints,
        name_prefix,
        first_day_dow,
        nested_basis(parents, active, own),
    )


class ByParent(RandomVariable):
    """A per-group random variable sampled for parents, given to places.

    Wraps a random variable that takes `n_groups` (such as `HazardDelay`
    or `MissingOnsetFraction`): it is sampled once per parent (province)
    and each place (zone) takes its parent's value along `axis`.
    """

    def __init__(
        self, name: str, inner: RandomVariable, parents: np.ndarray, axis: int
    ) -> None:
        """Initialize the wrapper.

        Parameters
        ----------
        name
            Name of the variable.
        inner
            Takes `n_groups` and returns one value per group on `axis`.
        parents
            Index of each place's parent, `(n_places,)`.
        axis
            Group axis of `inner`'s value.

        Raises
        ------
        TypeError
            If `inner` is not a `RandomVariable`.
        """
        if not isinstance(inner, RandomVariable):
            raise TypeError("inner must be a RandomVariable.")
        super().__init__(name=name)
        self.inner = inner
        self.parents = np.asarray(parents, dtype=int)
        self.n_parents = int(self.parents.max()) + 1
        self.axis = axis

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"ByParent(name={self.name!r}, inner={self.inner!r}, "
            f"parents={self.parents.tolist()!r}, axis={self.axis})"
        )

    def shared_groups(self, n_groups: int) -> np.ndarray:
        """Return the places whose parent takes the shared value.

        Parameters
        ----------
        n_groups
            Number of places.

        Returns
        -------
        numpy.ndarray
            Boolean `(n_groups,)`.
        """
        inner = getattr(self.inner, "shared_groups", None)
        if inner is None:
            return np.zeros(n_groups, dtype=bool)
        shared = np.asarray(inner(self.n_parents), dtype=bool)
        out: np.ndarray = shared[self.parents]
        return out

    def sample(self, n_groups: int | None = None, **kwargs: object) -> Array:
        """Sample the parents' values and give each place its parent's.

        Parameters
        ----------
        n_groups
            Number of places. Ignored beyond a check.
        **kwargs
            Passed to `inner`.

        Returns
        -------
        Array
            `inner`'s value with the group axis indexed by place.

        Raises
        ------
        ValueError
            If `n_groups` is not the number of places.
        """
        if n_groups is not None and n_groups != self.parents.size:
            raise ValueError(
                f"Expected {self.parents.size} places, got {n_groups}."
            )
        value = self.inner(n_groups=self.n_parents, **kwargs)
        return jnp.take(jnp.asarray(value), self.parents, axis=self.axis)


class ZoneInfections(MetapopulationInfections):
    """`MetapopulationInfections` with zone deviations within provinces.

    Rt is the pooled trend plus a province deviation
    (`subpop_rt_deviation_process`) plus a zone deviation within the
    province (`place_deviation_process`), each sum-to-zero
    (`zone_pooled_walk`).
    """

    def __init__(
        self,
        *,
        parents: ArrayLike,
        place_deviation_process: TemporalProcess,
        own: ArrayLike | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the zone infections process.

        Parameters
        ----------
        parents
            Index of each zone's province.
        place_deviation_process
            Temporal process of the zone deviations.
        own
            Boolean, one per zone: zones with an Rt deviation of their own
            within their province. `None` is every seeded zone.
        **kwargs
            Passed to `MetapopulationInfections`.
        """
        super().__init__(**kwargs)
        self.parents = np.asarray(parents, dtype=int)
        self.place_deviation_process = place_deviation_process
        self.own = None if own is None else np.asarray(own, dtype=bool)

    def sample(
        self,
        n_days_post_init: int,
        subpop_fractions: Array | None = None,
        first_day_dow: int | None = None,
        **kwargs: object,
    ) -> Any:
        """Sample per-capita infections for each zone.

        As `MetapopulationInfections.sample`, with `zone_pooled_walk` in
        place of `pooled_walk`.
        """
        from pyrenew.latent.base import LatentSample

        pop = self._parse_and_validate_fractions(subpop_fractions)
        n_init = self.n_initialization_points
        log_rt_baseline, log_rt = zone_pooled_walk(
            self.baseline_rt_process,
            self.subpop_rt_deviation_process,
            self.place_deviation_process,
            self.log_rt_time_0_rv(),
            n_init + n_days_post_init,
            self.parents,
            "log_rt",
            first_day_dow,
            active=self.seeded,
            own=self.own,
        )
        rt = jnp.exp(log_rt)
        gen_int = self.gen_int_rv()
        I0 = self._validate_and_prepare_I0(jnp.asarray(self.I0_rv()), pop)
        I0_all, seeding = self._initial_infections(
            I0, rt[0], gen_int, n_days_post_init
        )
        post, r_eff = self._infections_from_rt(
            rt[n_init:], I0_all, gen_int, pop, seeding
        )
        infections = jnp.vstack([I0_all, post])
        aggregate = infections @ pop.fractions
        with numpyro.handlers.scope(prefix=self.name, divider="::"):
            numpyro.deterministic("rt_baseline", jnp.exp(log_rt_baseline))
            numpyro.deterministic("rt_subpop", rt)
            numpyro.deterministic(
                "rt_subpop_adjusted", jnp.vstack([rt[:n_init], r_eff])
            )
        if self.rt_observation is not None:
            self.rt_observation(
                rt=rt,
                infections=infections,
                n_init=n_init,
                fractions=pop.fractions,
            )
        return LatentSample(aggregate=aggregate, all_subpops=infections)


class ZoneOnsetStreams(OnsetStreams):
    """`OnsetStreams` with the delay counts by province.

    Each province's expected delay cells are the sum of its zones'.
    """

    def __init__(
        self, *args: Any, delay_groups: np.ndarray, **kwargs: Any
    ) -> None:
        """Initialize the streams.

        Parameters
        ----------
        *args, **kwargs
            Passed to `OnsetStreams`.
        delay_groups
            The province of each zone, `(P,)`.
        """
        super().__init__(*args, **kwargs)
        self.delay_groups = np.asarray(delay_groups, dtype=int)

    def _score_delays(self, triangle: Array, counts: Array) -> None:
        n_groups = int(self.delay_groups.max()) + 1
        by_province = jax.ops.segment_sum(
            jnp.swapaxes(triangle, 0, 1), self.delay_groups, n_groups
        ).swapaxes(0, 1)
        super()._score_delays(by_province, counts)


def zone_table(locations: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Health zone populations and centres, as a locations table.

    The hub zones the map lacks (19 of 519, none with a case so far) take
    their province's centre and the mean WorldPop count of its mapped
    zones, or the median zone's where none of its zones are mapped.

    Parameters
    ----------
    locations : tuple of str, optional
        Zones to return, in order. Defaults to every zone in the file.

    Returns
    -------
    pandas.DataFrame
        Columns `location`, `population`, `latitude` and `longitude`.
    """
    zones = pd.read_csv(ZONE_CENTRES, comment="#").rename(
        columns={"worldpop": "population"}
    )
    zones = zones.loc[:, ["location", "population", "latitude", "longitude"]]
    if locations is None:
        return zones
    return _zone_rows(tuple(locations), zones.set_index("location"))


def _zone_rows(
    locations: tuple[str, ...], zones: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Zone table rows for some zones, with the province fallback."""
    if zones is None:
        zones = zone_table().set_index("location")
    missing = [z for z in locations if z not in zones.index]
    if missing:
        provinces = province_centres().set_index("location")
        by_province = zones.groupby(zones.index.str[:5])["population"].mean()
        extra = pd.DataFrame(
            {
                "population": [
                    by_province.get(z[:5], zones["population"].median())
                    for z in missing
                ],
                "latitude": [provinces.loc[z[:5], "latitude"] for z in missing],
                "longitude": [
                    provinces.loc[z[:5], "longitude"] for z in missing
                ],
            },
            index=pd.Index(missing, name="location"),
        )
        zones = pd.concat([zones, extra])
    return zones.reindex(list(locations)).reset_index()


def place_distances(centres: pd.DataFrame) -> np.ndarray:
    """Great-circle distances in km between place centres.

    Parameters
    ----------
    centres : pandas.DataFrame
        Columns `latitude` and `longitude`, one row per place.

    Returns
    -------
    numpy.ndarray
        Symmetric, zero diagonal.

    Raises
    ------
    ValueError
        If a place has no centre.
    """
    if centres[["latitude", "longitude"]].isna().any().any():
        raise ValueError("Missing place centre.")
    lat = np.radians(centres["latitude"].to_numpy(float))
    lon = np.radians(centres["longitude"].to_numpy(float))
    a = (
        np.sin((lat[:, None] - lat[None, :]) / 2) ** 2
        + np.cos(lat[:, None])
        * np.cos(lat[None, :])
        * np.sin((lon[:, None] - lon[None, :]) / 2) ** 2
    )
    return np.asarray(2 * 6371 * np.arcsin(np.sqrt(a)), dtype=float)


def zone_bvd_inputs(
    rows: pd.DataFrame,
    names: list[str],
    parent_of: Mapping[str, str],
    start: dt.date,
    masked: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    """Map BVD rows onto the zones, as `inputs.bvd_inputs`.

    The national row and the patch rows hold every zone of their
    provinces. Rows whose group starts with `zone:` (BVDOutbreakSize's
    zone Rt) hold the one zone they name, as a deviation from national.

    Returns
    -------
    dict or None
        As `inputs.bvd_inputs`.
    """
    single = rows["group"].astype(str).str.startswith("zone:")
    parts = [
        bvd_inputs(
            rows.loc[~single], [parent_of[p] for p in names], start, masked
        )
    ]
    if single.any():
        # A masked province's zones carry its data, so they are left out.
        keep = rows.loc[single & ~rows["members"].str[:5].isin(masked)]
        parts.append(bvd_inputs(keep, names, start, masked))
    found = [p for p in parts if p is not None]
    if not found:
        return None
    out = dict(found[0])
    for part in found[1:]:
        for key in ("groups", "values", "scales", "kinds"):
            out[key] = np.concatenate([out[key], part[key]])
    return out


def zone_kernel(
    zones: tuple[str, ...], table: pd.DataFrame, decay_km: float = DECAY_KM
) -> np.ndarray:
    r"""Row-normalised gravity kernel between zones.

    $K_{qz} \propto N_z \exp(-D_{qz} / \text{decay})$ for $q \ne z$, with
    WorldPop counts $N$ from `table` and great-circle distances $D$
    between zone centroids (`zone_table`).

    Returns
    -------
    numpy.ndarray
        Rows sum to one, zero diagonal.
    """
    pops = table.set_index("location")["population"].reindex(list(zones))
    if pops.isna().any():
        raise ValueError("Missing zone population.")
    dist_km = place_distances(_zone_rows(zones))
    k = pops.to_numpy(float)[None, :] * np.exp(-dist_km / decay_km)
    np.fill_diagonal(k, 0.0)
    k = k / k.sum(axis=1, keepdims=True)
    if not np.all(np.isfinite(k)):
        raise ValueError("Importation kernel is not finite.")
    return np.asarray(k, dtype=float)


def zone_vintage_from_tables(
    target: pd.DataFrame,
    delays: pd.DataFrame | None,
    vintage: str,
    locations: tuple[str, ...],
    table: pd.DataFrame,
    config: ModelConfig = MAIN,
    horizon_weeks: int = 3,
    masked: tuple[str, ...] = (),
    bvd: pd.DataFrame | None = None,
    release_pairs: dict[str, np.ndarray] | None = None,
    parents: Mapping[str, str] | None = None,
) -> ZoneVintage:
    """Build a `ZoneVintage` from zone rows and province delay rows.

    Every zone in `locations` is modelled. The week grid
    starts one week before national weekly onsets first reach
    `START_COUNT` and ends with the week holding the vintage's cut-off,
    its latest onset day (`data.releases.release_cutoff`).

    Parameters
    ----------
    target : pandas.DataFrame
        Rows as in `bdbvhubrenewal.data.load_target`, with `cutoff`.
    delays : pandas.DataFrame or None
        Delay rows (`location`, `onset_week`, `delay`) at this vintage.
    vintage : str
        The `as_of` date.
    locations : tuple of str
        Zones.
    table : pandas.DataFrame
        Zone populations (`zone_table`).
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
    release_pairs : dict, optional
        Rows of each earlier delay file and how many the file before it
        had (`data.releases.release_pairs`).
    parents : mapping, optional
        The province of each zone. Defaults to the code's first five
        characters. The delay counts are by province, `masked` names
        provinces, and a BVD patch holds the zones of its provinces.

    Returns
    -------
    ZoneVintage
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
    totals_wide = cast(pd.DataFrame, wide["total"])
    active = tuple(locations)
    national = totals_wide.reindex(columns=list(active)).fillna(0).sum(axis=1)
    counts_by_week = national.to_numpy()
    threshold = min(START_COUNT, counts_by_week.max())
    big = int(np.argmax(counts_by_week >= threshold))
    first = _saturday(str(national.index[max(0, big - 1)]))
    start = first - dt.timedelta(days=6)
    cutoff = (release_cutoff(target, vintage) - start).days
    n_weeks = cutoff // 7 + 1
    weeks = [
        (first + dt.timedelta(weeks=i)).isoformat() for i in range(n_weeks)
    ]

    def grid(column: str, locs: tuple[str, ...]) -> np.ndarray:
        out = cast(pd.DataFrame, wide[column]).reindex(
            index=weeks, columns=list(locs)
        )
        return out.fillna(0).to_numpy(dtype=float)

    # The province of each place, and the delay tables: one per modelled
    # province, one per province of the zones, or one for a national fit.
    parent_of = {z: z[:5] for z in active} if parents is None else dict(parents)
    parent_names = list(dict.fromkeys(parent_of[p] for p in active))
    delay_places = parent_names
    n_places = 1 if config.national else len(delay_places)
    counts = np.zeros((n_weeks, n_places, config.dmax + 1))
    if delays is not None and len(delays):
        # Known onsets count every case, so negative delays go to 0 and
        # delays past dmax to dmax, the last cell of the hazard pmf,
        # rather than leaving the delay counts short of the known stream.
        keep = delays.loc[
            (
                delays["location"].isin(delay_places)
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
        delay = np.clip(keep["delay"].to_numpy(), 0, config.dmax)
        place = (
            np.zeros(len(keep), dtype=int)
            if config.national
            else keep["location"].map(delay_places.index).to_numpy(dtype=int)
        )
        np.add.at(counts, (week[ok], place[ok], delay[ok]), 1)

    known, imputed = grid("observed", active), grid("imputed", active)
    totals = known + imputed
    pops = table.set_index("location").loc[list(active), "population"]
    names = list(active)
    if config.national:
        known = known.sum(axis=1, keepdims=True)
        imputed = imputed.sum(axis=1, keepdims=True)
        pops = pd.Series([pops.sum()])
        names = [active[0] if len(active) == 1 else NATIONAL]
    kernel = None if len(names) == 1 else zone_kernel(active, table)
    off = [p in masked or parent_of.get(p, p) in masked for p in names]
    return ZoneVintage(
        start_date=start,
        cutoff=cutoff,
        known=known,
        imputed=imputed,
        delay_counts=counts,
        hub_delay_pmf=hub_delay_pmf(vintage, config.dmax),
        populations=pops.to_numpy(dtype=float),
        kernel=kernel,
        horizon_weeks=horizon_weeks,
        names=names,
        fit_mask=~np.array(off, dtype=bool),
        bvd=None
        if bvd is None or config.national
        else zone_bvd_inputs(bvd, names, parent_of, start, masked),
        seed_days=None if config.national else _seed_days(totals),
        release_pairs=release_pairs,
        parents=np.array([parent_names.index(parent_of[p]) for p in names]),
        parent_names=parent_names,
        own=totals.sum(axis=0) >= MIN_OWN_CASES,
    )


def build_zone_model(
    data: ZoneVintage, config: ModelConfig = MAIN, anchor: bool = True
) -> MultiSignalModel:
    """Build the zone model for one vintage, as `model.build_model`.

    Parameters
    ----------
    data : Vintage
        The vintage.
    config : ModelConfig
        Structural choices.
    anchor : bool
        Anchor the `ANCHORED` walks at their mean (`MeanAnchoredWalk`).
        Without it, and with one zone per province, this is the province
        model.

    Returns
    -------
    MultiSignalModel
        Model with one observation, `onsets`.
    """
    n_prov = data.n_provinces
    # Provinces with no case are never seeded. Their seed and deviations
    # are barely informed (only their zero counts, through importation),
    # so they are not sampled.
    active = data.seeded

    # With zones as places, the delay and the missing-onset share are by
    # province (the delay files list provinces only) and each zone takes
    # its province's; with provinces as places they are by place.
    def _zone_walk_a(
        name: str,
        sd: float,
        n_weeks: int,
        ar: bool = False,
        forecast_half_life: float | None = None,
    ) -> TemporalProcess:
        return _zone_walk(name, sd, n_weeks, ar, forecast_half_life, anchor)

    if data.parents is None:
        raise ValueError("A zone model needs each zone's province.")
    parents = data.parents
    n_groups = int(parents.max()) + 1
    group_active = parent_active(parents, active)

    def by_place(rv: Any, axis: int) -> Any:
        return ByParent(f"{rv.name}_by_zone", rv, parents, axis)

    mixing = None
    if config.coupling == "gravity" and data.kernel is not None:
        mixing = Importation(
            "mixing",
            DeterministicVariable("kernel", data.kernel),
            DistributionalVariable("importation", dist.Beta(1, 100)),
        )

    effective_population = None
    if config.depletion:
        effective_population = (
            DistributionalVariable(
                "effective_population", dist.LogNormal(np.log(0.05), 0.75)
            )
            if config.effective_population is None
            else DeterministicVariable(
                "effective_population", config.effective_population
            )
        )

    builder = PyrenewBuilder()
    builder.configure_latent(
        ZoneInfections,
        name="latent",
        gen_int_rv=DeterministicPMF("gen_int", generation_interval()),
        # The seed is the infections per day that will be confirmed, on
        # the first data day, turned into infections per resident.
        # Never-seeded provinces take the prior median, 1, as only their
        # near-zero start uses it.
        I0_rv=TransformedVariable(
            "I0",
            ActiveValues(
                "initial_infections_all",
                DistributionalVariable(
                    "initial_infections",
                    dist.LogNormal(0.0, 1.5).expand([int(active.sum())]),
                ),
                active,
                1.0,
            ),
            AffineTransform(0.0, 1 / (ALPHA0 * data.populations)),
        ),
        log_rt_time_0_rv=_pooled(
            "log_rt0", n_prov, 0.0, 0.5, 0.3, active=active
        ),
        baseline_rt_process=_zone_walk_a(
            "rt_sd",
            0.2,
            fit_weeks(data, config),
            forecast_half_life=FORECAST_RT_HALF_LIFE,
        ),
        subpop_rt_deviation_process=_zone_walk_a(
            "rt_dev_sd", 0.1, fit_weeks(data, config), ar=True
        ),
        mixing_rv=mixing,
        effective_population_rv=effective_population,
        seed_days=data.seed_days,
        seeded=active,
        parents=parents,
        place_deviation_process=_zone_walk_a(
            "rt_zone_dev_sd", 0.1, fit_weeks(data, config), ar=True
        ),
        own=data.own,
        rt_observation=RtPseudoObservation(
            "bvd", alpha=config.bvd_alpha, **data.bvd
        )
        if config.bvd and data.bvd is not None
        else None,
    )

    # Only the product of ascertainment and the effective population is
    # identified (see NOTES.md), so the starting ascertainment is fixed and
    # the effective population carries the level.
    ascertainment = IndependentAscertainment(
        name="ascertainment",
        rate_rvs={"onsets": DeterministicVariable("alpha0", ALPHA0)},
        temporal_processes={
            "onsets": _zone_walk_a("alpha_sd", 0.03, fit_weeks(data, config))
        },
    )
    builder.add_ascertainment(ascertainment)

    pad = GEN_INT_LENGTH  # The model axis starts GEN_INT_LENGTH days early.

    def fixed(name: str, daily: np.ndarray) -> FixedTimeVarying:
        return FixedTimeVarying(
            name, np.vstack([np.repeat(daily[:1], pad, 0), daily])
        )

    if config.streams == "total":
        if config.truncation:
            rows = data.hub_delay_pmf
        else:
            rows = np.eye(1, config.dmax + 1)[0]
        delay = fixed("delay", np.tile(rows, (7 * data.n_weeks, 1)))
        missing = None
    elif not config.truncation:
        raise ValueError("Split streams always use right truncation.")
    else:
        missing = MissingOnsetFraction(
            "missing_fraction",
            # Non-centred: the imputed share pins these offsets only
            # weakly, and centred they form a funnel (R-hat 1.68).
            _pooled(
                "logit_pi0",
                n_groups,
                -2.0,
                1.0,
                0.5,
                centred=False,
                active=group_active,
            ),
            _zone_walk_a("pi_sd", 0.3, fit_weeks(data, config)),
            _zone_walk_a("pi_dev_sd", 0.1, fit_weeks(data, config)),
            active=group_active,
        )

    if missing is not None:
        delay = HazardDelay(
            "delay",
            config.dmax,
            DistributionalVariable("delay_hazard0", dist.Normal(-2.0, 1.0)),
            # Real fits put the step SD at about 0.2 (2026-10-24: 90% CrI
            # 0.16 to 0.25), in the tail of a scale of 0.1, hence 0.4.
            NonCentredRandomWalk(
                DistributionalVariable("delay_smooth_sd", dist.HalfNormal(0.4))
            ),
            _zone_walk_a("delay_sd", 0.15, fit_weeks(data, config)),
            # Partial pooling of each seeded province's delay towards the
            # shared delay, in level and in drift over time. Never-seeded
            # provinces have no delay data, so they take the shared delay.
            PooledInitialValue(
                "delay_offset",
                n_groups,
                DeterministicVariable("delay_offset_mean", 0.0),
                DistributionalVariable("delay_offset_sd", dist.HalfNormal(0.5)),
                centred=False,
                active=group_active,
            ),
            _zone_walk_a("delay_dev_sd", 0.1, fit_weeks(data, config)),
            active=group_active,
        )
    builder.add_observation(
        ZoneOnsetStreams(
            name="onsets",
            temporal_pmf_rv=CensoredPMF(
                "infection_to_onset",
                "lognormal",
                # BVDOutbreakSize's priors (onset_incidence_model).
                DistributionalVariable(
                    "incubation_mean",
                    dist.TruncatedNormal(
                        INCUBATION_MEAN, INCUBATION_MEAN_SD, low=1.0
                    ),
                ),
                DistributionalVariable(
                    "incubation_sd",
                    dist.TruncatedNormal(
                        INCUBATION_SD, INCUBATION_SD_SD, low=1.0
                    ),
                ),
                INCUBATION_LENGTH,
            ),
            ascertainment_rate_rv=ascertainment.for_signal("onsets"),
            missing_fraction_rv=None
            if missing is None
            else by_place(missing, 1),
            delay_rv=by_place(delay, 0),
            delay_groups=parents,
            known_noise=_nb_noise("known"),
            imputed_noise=None if missing is None else _nb_noise("imputed"),
            start_dow=SUNDAY,
            publication_lag=DistributionalVariable(
                "publication_lag",
                dist.Dirichlet(np.ones(MAX_PUBLICATION_LAG + 1)),
            ),
            delay_noise=_dm_noise()
            if config.delay_counts == "dirichlet-multinomial"
            else MultinomialNoise(),
            # The axis starts on a Sunday with whole weeks.
            weekly_axis=True,
            pair_likelihood=ReleasePairLikelihood(
                "release_pairs",
                dist.Beta(1.0, 19.0),
                # At most 1000, the binomial limit for cells of 1 to 30
                # rows (`capped_concentration`).
                capped_concentration(
                    "release_pairs_concentration",
                    DistributionalVariable(
                        "release_pairs_overdispersion", dist.HalfNormal(0.5)
                    ),
                    1e-3,
                ),
            ),
        )
    )
    model = builder.build()
    model.config = config
    if model.latent.n_initialization_points != pad:
        raise ValueError(
            "The model axis must start a whole number of weeks early."
        )
    return model
