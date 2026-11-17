# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""A Gaussian copula that pairs a model's marginal draws into trajectories.

A model that gives each place and step only its marginal, such as
`randomwalknowcast`, draws each cell independently.
`GaussianCopula.couple` pairs those draws into trajectories: each
trajectory draws one correlated normal vector $z$ over places and steps,
and every cell's draws are reordered across trajectories so they rank as
$z$ does.
This is the Gaussian copula with $u = \Phi(z)$ read off each cell's own
draws, so each cell keeps exactly the draws it had.

The correlation of $z$ is a place correlation times a step correlation.
Two steps $\delta$ days apart have correlation $\rho_t^{\delta / 7}$.
Between places it is $\rho_p$ for two provinces, $\rho_n$ for the
country and a province, and $\rho_z$ for two health zones of one
province or a zone and its province. Zones of different provinces take
$\rho_p$, and a zone and the country $\rho_n$.
`estimate_copula` estimates the four from the normal scores of matured
weeks at past origins (`normal_scores`), shrunk towards a prior.

`pair_samples` pairs one forecast table, and `RollingCopula` pairs a
model's forecasts at every origin, each with the copula estimated from
the model's own recent matured errors.
`pair_registered` pairs the models registered as `paired`, those that
draw each place and week on their own.
Zones are paired in their own right. A province that is the sum of its
zones is summed again from the paired zones, and a country that is the
sum of its provinces from the paired provinces.
"""

from __future__ import annotations

import dataclasses
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from scipy.special import ndtri
from statsmodels.stats.correlation_tools import corr_nearest

from bdbvhubrenewal.forecasters.base import NATIONAL, list_models

__all__ = [
    "MAX_RHO",
    "PAIRINGS",
    "PRIOR",
    "PRIOR_WEIGHT",
    "WINDOW_WEEKS",
    "GaussianCopula",
    "RollingCopula",
    "error_scores",
    "estimate_copula",
    "estimate_record",
    "normal_scores",
    "pair_registered",
    "pair_samples",
    "pair_walks",
]

MAX_RHO: Final = 0.95
"""Largest correlation `estimate_copula` gives."""

WINDOW_WEEKS: Final = 8
"""Matured weeks of errors `RollingCopula` estimates from."""

PAIRINGS: Final = ("off", "prior", "rolling")
"""How a walk model pairs its draws: not at all, by the prior, or by the
copula estimated from its own recent errors (the default)."""

PRIOR_WEIGHT: Final = 100.0
"""Pairs of normal scores the prior counts as, in `estimate_copula`'s
shrinkage."""

_CELL = ["model", "origin", "location", "target_end_date"]
_FIT_COLUMNS = [
    "model",
    "origin",
    "n_scores",
    "rho_time",
    "rho_provinces",
    "rho_national",
    "rho_zones",
]


def _is_zone(location: pd.Series) -> pd.Series:
    """Return whether each location is a health zone (`CD-XX-YY`)."""
    return location.astype(str).str.count("-") >= 2


def _province(code: str) -> str:
    """Return a zone's province, or the code itself."""
    return code.rsplit("-", 1)[0] if code.count("-") >= 2 else code


def _factor(corr: np.ndarray) -> np.ndarray:
    """Return the Cholesky factor $L$ of the correlation matrix nearest `corr`.

    A valid correlation matrix is used as it is. Otherwise it is replaced
    by the nearest correlation matrix (Higham 2002), as statsmodels'
    `corr_nearest` gives it.
    """
    if np.linalg.eigvalsh(corr).min() <= 0:
        corr = corr_nearest(corr, threshold=1e-10)
    return np.asarray(np.linalg.cholesky(corr))


@dataclass(frozen=True)
class GaussianCopula:
    r"""Correlations of a Gaussian copula over places and weeks.

    Attributes
    ----------
    rho_time : float
        $\rho_t$, between consecutive weeks of a place.
    rho_provinces : float
        $\rho_p$, between two provinces in a week.
    rho_national : float
        $\rho_n$, between the country and a province or zone in a week.
    rho_zones : float
        $\rho_z$, between two zones of one province, or a zone and its
        province, in a week.
    """

    rho_time: float = 0.0
    rho_provinces: float = 0.0
    rho_national: float = 0.0
    rho_zones: float = 0.0

    def __post_init__(self) -> None:
        """Check each correlation is in [0, 1)."""
        for name in ("rho_time", "rho_provinces", "rho_national", "rho_zones"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be between 0 and 1 (not 1).")

    def normals(
        self,
        rng: np.random.Generator,
        n: int,
        n_places: int,
        days: np.ndarray,
        national: bool = True,
        locations: Sequence[str] | None = None,
        country: str = NATIONAL,
    ) -> np.ndarray:
        r"""Draw correlated standard normals over places and steps.

        Where the correlations give no correlation matrix, a nearby one is
        used (`_factor`).

        Parameters
        ----------
        rng : numpy.random.Generator
            Random generator.
        n : int
            Trajectories.
        n_places : int
            Places.
        days : numpy.ndarray
            Day of each step.
        national : bool
            Whether the first place is the country. Otherwise every place
            is a province. Ignored with `locations`.
        locations : sequence of str, optional
            The places' codes, which say which is the country (`country`),
            a province or a zone (`CD-XX-YY`). By default the first place
            is the country if `national` and the others are provinces.
        country : str
            The country's code in `locations`.

        Returns
        -------
        numpy.ndarray
            `[n, n_places, len(days)]`.
        """
        place = self.places(n_places, national, locations, country)
        days = np.asarray(days, dtype=float)
        step = self.rho_time ** (np.abs(np.subtract.outer(days, days)) / 7)
        e = rng.standard_normal((n, n_places, len(days)))
        return np.asarray(
            np.einsum(
                "pi,nij,tj->npt",
                _factor(place),
                e,
                _factor(step),
                optimize=True,
            )
        )

    def places(
        self,
        n_places: int,
        national: bool = True,
        locations: Sequence[str] | None = None,
        country: str = NATIONAL,
    ) -> np.ndarray:
        """Return the correlation between places, as `normals` uses it."""
        if locations is None:
            place = np.full((n_places, n_places), self.rho_provinces)
            if national:
                place[0, :] = place[:, 0] = self.rho_national
            np.fill_diagonal(place, 1.0)
            return place
        codes = np.asarray(list(locations), dtype=object)
        is_country = codes == country
        zone = np.array([c.count("-") >= 2 for c in codes]) & ~is_country
        parent = np.array([_province(c) for c in codes], dtype=object)
        place = np.full((len(codes), len(codes)), self.rho_provinces)
        same = (parent[:, None] == parent[None, :]) & (
            zone[:, None] | zone[None, :]
        )
        place[same] = self.rho_zones
        place[is_country, :] = self.rho_national
        place[:, is_country] = self.rho_national
        np.fill_diagonal(place, 1.0)
        return place

    def order(
        self,
        draws: np.ndarray,
        rng: np.random.Generator,
        days: np.ndarray | None = None,
        national: bool = True,
        locations: Sequence[str] | None = None,
        country: str = NATIONAL,
    ) -> np.ndarray:
        """Return the trajectory each cell's draws go to.

        Parameters
        ----------
        draws : numpy.ndarray
            Draws of each place and step, `[n, n_places, n_steps]`.
        rng : numpy.random.Generator
            Random generator.
        days : numpy.ndarray, optional
            Day of each step, weekly steps by default.
        national : bool
            Whether the first place is the country.
        locations : sequence of str, optional
            The places' codes (`normals`).
        country : str
            The country's code in `locations`.

        Returns
        -------
        numpy.ndarray
            Indices `[n, n_places, n_steps]`: trajectory $k$ of a cell
            takes the cell's draw at index $k$, so the draws rank as the
            copula's normals do.
        """
        n, n_places, n_steps = draws.shape
        if days is None:
            days = 7 * np.arange(n_steps)
        z = self.normals(rng, n, n_places, days, national, locations, country)
        rank = np.argsort(np.argsort(z, axis=0), axis=0)
        by_value = np.argsort(draws, axis=0, kind="stable")
        return np.take_along_axis(by_value, rank, axis=0)

    def couple(
        self,
        draws: np.ndarray,
        rng: np.random.Generator,
        days: np.ndarray | None = None,
        national: bool = True,
        locations: Sequence[str] | None = None,
        country: str = NATIONAL,
    ) -> np.ndarray:
        """Pair independent draws into trajectories (`order`).

        Parameters
        ----------
        draws : numpy.ndarray
            Draws of each place and step, `[n, n_places, n_steps]`.
        rng : numpy.random.Generator
            Random generator.
        days : numpy.ndarray, optional
            Day of each step, weekly steps by default.
        national : bool
            Whether the first place is the country.
        locations : sequence of str, optional
            The places' codes (`normals`).
        country : str
            The country's code in `locations`.

        Returns
        -------
        numpy.ndarray
            The same draws of each cell, reordered across trajectories.
        """
        index = self.order(draws, rng, days, national, locations, country)
        return np.take_along_axis(draws, index, axis=0)


PRIOR: Final = GaussianCopula(
    rho_time=0.9, rho_provinces=0.7, rho_national=0.7, rho_zones=0.7
)
"""The correlations estimates are shrunk towards. The zones' is set to
the provinces' (no zone errors had matured to choose it)."""


def _final(samples: pd.DataFrame) -> pd.Series:
    """Rows of final counts: those with no `as_of`, if there is one."""
    if "as_of" not in samples:
        return pd.Series(True, index=samples.index)
    return samples["as_of"].isna()


def _summed(final: pd.DataFrame, national: str) -> bool:
    """Whether the country is in `final` as the sum of the provinces."""
    is_nat = final["location"] == national
    provinces = ~is_nat & ~_is_zone(final["location"])
    return _sum_of(final, is_nat, provinces)


def _sum_of(final: pd.DataFrame, parent: pd.Series, parts: pd.Series) -> bool:
    """Whether the rows `parent` are, draw by draw, the sum of `parts`."""
    if not parent.any() or not parts.any():
        return False
    keys = ["target_end_date", "sample"]
    total = final.loc[parts].groupby(keys)["value"].sum()
    own = final.loc[parent].set_index(keys)["value"]
    return bool(np.allclose(own.to_numpy(), total.reindex(own.index)))


def _summed_provinces(final: pd.DataFrame) -> set[str]:
    """Return the provinces in `final` that are the sum of their zones."""
    zone = _is_zone(final["location"])
    parents = final.loc[zone, "location"].map(_province)
    out = set()
    for prov in set(parents):
        mine = zone & (final["location"].map(_province) == prov)
        if _sum_of(final, final["location"] == prov, mine):
            out.add(prov)
    return out


def pair_samples(
    samples: pd.DataFrame,
    copula: GaussianCopula,
    rng: np.random.Generator,
    national: str = NATIONAL,
) -> pd.DataFrame:
    """Pair one forecast's draws into trajectories by a copula.

    Each place and week keeps its draws, and every row of a draw, such as
    its counts as a later release reports them (`as_of`), moves with it.
    Health zones are paired in their own right (`rho_zones`).
    A province that is the sum of its zones is not paired but summed
    again from them, and a country that is the sum of the provinces is
    summed again from those.

    Parameters
    ----------
    samples : pandas.DataFrame
        One forecast: `location`, `target_end_date`, `sample` and
        `value`, a draw of every place and week for each sample, and
        optionally `as_of` and other columns.
    copula : GaussianCopula
        The copula.
    rng : numpy.random.Generator
        Random generator.
    national : str
        The country's location code.

    Returns
    -------
    pandas.DataFrame
        `samples` with the sample ids of each cell reordered.
    """
    final = samples.loc[_final(samples)]
    summed = _summed(final, national)
    sums = _summed_provinces(final)
    places = sorted(set(final["location"]) - {national} - sums)
    own = not summed and bool((final["location"] == national).any())
    if own:
        places = [national, *places]
    final = final.loc[final["location"].isin(places)]
    weeks = sorted(set(final["target_end_date"]))
    wide = final.pivot(
        index="sample", columns=["location", "target_end_date"], values="value"
    ).reindex(columns=pd.MultiIndex.from_product([places, weeks]))
    draws = wide.to_numpy().reshape(len(wide), len(places), len(weeks))
    days = (pd.to_datetime(weeks) - pd.Timestamp(weeks[0])).days.to_numpy()
    index = copula.order(draws, rng, days, locations=places, country=national)
    ids = wide.index.to_numpy()
    moves = pd.DataFrame(
        {
            "location": np.repeat(places, len(weeks) * len(ids)),
            "target_end_date": np.tile(np.repeat(weeks, len(ids)), len(places)),
            "sample": ids[index.transpose(1, 2, 0).reshape(-1)],
            "paired": np.tile(ids, len(places) * len(weeks)),
        }
    )
    keys = ["location", "target_end_date", "sample"]
    out = samples.merge(moves, on=keys, how="left")
    out["sample"] = (
        out["paired"].fillna(out["sample"]).astype(samples["sample"].dtype)
    )
    out = out.drop(columns="paired")
    for prov in sorted(sums):
        mine = _is_zone(out["location"]) & (
            out["location"].map(_province) == prov
        )
        out = _sum_into(out, out["location"] == prov, mine)
    if summed:
        out = _sum_country(out, national)
    return out[list(samples.columns)].reset_index(drop=True)


def _sum_country(samples: pd.DataFrame, national: str) -> pd.DataFrame:
    """Set the country's rows to the sum of its provinces' rows."""
    province = samples["location"].str.count("-") == 1
    return _sum_into(samples, samples["location"] == national, province)


def _sum_into(
    samples: pd.DataFrame, parent: pd.Series, parts: pd.Series
) -> pd.DataFrame:
    """Set the rows `parent` to the sum of the rows `parts`, by draw."""
    keys = ["target_end_date", "sample"]
    if "as_of" in samples:
        keys.append("as_of")
    total = (
        samples.loc[parts]
        .groupby(keys, dropna=False)["value"]
        .sum()
        .rename("total")
    )
    rows = samples.loc[parent].join(total, on=keys)
    have = rows["total"].notna().to_numpy()
    idx = rows.index[have]
    samples.loc[idx, "value"] = rows.loc[idx, "total"].to_numpy()
    return samples


def normal_scores(
    forecasts: pd.DataFrame, truth: pd.DataFrame, seed: int
) -> pd.DataFrame:
    r"""Return the normal score of each truth in its forecast's draws.

    The score is $\Phi^{-1}$ of the randomised PIT: with $m$ draws, $b$
    below the truth and $e$ equal to it, $(b + U (e + 1)) / (m + 1)$ for
    $U$ uniform, drawn for each forecast from a hash of the seed, model,
    origin, place and week, so a score does not depend on which other
    forecasts are scored with it.
    A forecast whose draws are all equal gives no score, as its PIT is
    $U$ alone.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Columns `model`, `origin`, `location`, `target_end_date`, `sample`
        and `value`.
    truth : pandas.DataFrame
        Columns `location`, `target_end_date` and `truth`, plus any
        columns it should be matched on.
    seed : int
        Seed of the uniform draws.

    Returns
    -------
    pandas.DataFrame
        One row per forecast with a truth and spread: `model`, `origin`,
        `location`, `target_end_date` and `z`.
    """
    on = [c for c in truth.columns if c != "truth"]
    rows = forecasts.merge(truth.dropna(subset=["truth"]), on=on)
    rows = rows.assign(
        below=rows["value"] < rows["truth"],
        equal=rows["value"] == rows["truth"],
    )
    out = rows.groupby(_CELL, as_index=False).agg(
        below=("below", "sum"),
        equal=("equal", "sum"),
        m=("value", "size"),
        spread=("value", "nunique"),
    )
    out = out.loc[out["spread"] > 1].reset_index(drop=True)
    if out.empty:
        return out[_CELL].assign(z=pd.Series(dtype=float))
    keys = out[_CELL].astype(str).agg("|".join, axis=1)
    # A hash in (0, 1): never 0 or 1, so the score is finite.
    u = np.array([(zlib.crc32(f"{seed}|{k}".encode()) + 0.5) for k in keys])
    u = u / 2.0**32
    pit = (out["below"] + u * (out["equal"] + 1)) / (out["m"] + 1)
    return out[_CELL].assign(z=ndtri(pit.to_numpy()))


def _corr(pairs: pd.DataFrame) -> float | None:
    """Pearson correlation of `z_a` and `z_b`, or None if it has none."""
    if len(pairs) < 2:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.corrcoef(pairs["z_a"], pairs["z_b"])[0, 1]
    return float(r) if np.isfinite(r) else None


def estimate_copula(
    scores: pd.DataFrame,
    prior: GaussianCopula = GaussianCopula(),  # noqa: B008
    weight: float = 0.0,
    national: str = NATIONAL,
) -> GaussianCopula:
    r"""Estimate the copula's correlations from normal scores.

    Each is the correlation $r$ of the scores of pairs from one forecast:
    $\rho_t$ of a place's consecutive weeks, $\rho_p$ of two provinces in a
    week, $\rho_n$ of the country and a province in a week, and $\rho_z$
    of two zones of one province, or a zone and its province, in a week.
    With $n$ such pairs and the prior's $\rho_0$ each is
    $(n r + w \rho_0) / (n + w)$, the prior's where the pairs give no
    $r$, then floored at zero and capped at `MAX_RHO`.

    Parameters
    ----------
    scores : pandas.DataFrame
        `normal_scores`: `model`, `origin`, `location`,
        `target_end_date` and `z`.
    prior : GaussianCopula
        The correlations to shrink towards.
    weight : float
        $w$, the pairs the prior counts as.
    national : str
        The country's location code.

    Returns
    -------
    GaussianCopula
    """
    pairs = _pairs(scores, national)
    return _estimate(pairs, prior, weight)[0]


def _estimate(
    pairs: dict[str, pd.DataFrame], prior: GaussianCopula, weight: float
) -> tuple[GaussianCopula, dict[str, object]]:
    """Shrink each correlation of `pairs` to the prior, with its record."""
    out: dict[str, float] = {}
    record: dict[str, object] = {}
    for name, rows in pairs.items():
        r, rho0, n = _corr(rows), getattr(prior, name), len(rows)
        rho = rho0 if r is None else (n * r + weight * rho0) / (n + weight)
        out[name] = float(np.clip(rho, 0.0, MAX_RHO))
        short = name.removeprefix("rho_")
        record[f"n_{short}"] = int(n) if r is not None else 0
        record[f"prior_{short}"] = r is None
    return GaussianCopula(**out), record


def _pairs(scores: pd.DataFrame, national: str) -> dict[str, pd.DataFrame]:
    """Return the pairs of normal scores behind each correlation."""
    week = pd.to_datetime(scores["target_end_date"])
    scores = scores.drop(columns="target_end_date").assign(week=week)
    forecast = ["model", "origin"]
    later = scores.assign(week=scores["week"] - pd.Timedelta(days=7))
    time = scores.merge(
        later, on=[*forecast, "location", "week"], suffixes=("_a", "_b")
    )
    same = scores.merge(scores, on=[*forecast, "week"], suffixes=("_a", "_b"))
    a_national = same["location_a"] == national
    b_national = same["location_b"] == national
    a_zone = _is_zone(same["location_a"]) & ~a_national
    b_zone = _is_zone(same["location_b"]) & ~b_national
    ordered = same["location_a"] < same["location_b"]
    provinces = same.loc[
        ~a_national & ~b_national & ~a_zone & ~b_zone & ordered
    ]
    kin = same["location_a"].map(_province) == same["location_b"].map(_province)
    zones = same.loc[(a_zone | b_zone) & kin & ordered]
    return {
        "rho_time": time,
        "rho_provinces": provinces,
        "rho_national": same.loc[a_national & ~b_national & ~b_zone],
        "rho_zones": zones,
    }


def estimate_record(
    scores: pd.DataFrame,
    prior: GaussianCopula = PRIOR,
    weight: float = PRIOR_WEIGHT,
    national: str = NATIONAL,
) -> tuple[GaussianCopula, dict[str, object]]:
    """Estimate the copula as `estimate_copula`, with its record.

    The record has, for each correlation (`time`, `provinces`, `national`,
    `zones`), `n_<name>`, the pairs behind it, and `prior_<name>`, whether
    it fell back to the prior for want of pairs that define it.
    """
    return _estimate(_pairs(scores, national), prior, weight)


@dataclass(frozen=True)
class RollingCopula:
    """A copula estimated at each origin from a model's recent errors.

    Attributes
    ----------
    prior : GaussianCopula
        The correlations estimates are shrunk towards.
    window_weeks : int
        The latest matured weeks whose errors are used.
    weight : float
        The pairs the prior counts as (`estimate_copula`).
    national : str
        The country's location code.
    """

    prior: GaussianCopula = PRIOR
    window_weeks: int = WINDOW_WEEKS
    weight: float = PRIOR_WEIGHT
    national: str = NATIONAL

    def fit(
        self, scores: pd.DataFrame, origin: str
    ) -> tuple[GaussianCopula, int]:
        """Estimate the copula at an origin.

        The scores used are those of forecasts made before the origin
        whose truth was released by it, of the `window_weeks` latest
        such weeks.

        Parameters
        ----------
        scores : pandas.DataFrame
            One model's `normal_scores`, with `truth_vintage`, the
            release of each truth.
        origin : str
            The origin, ISO date.

        Returns
        -------
        tuple
            The copula, and the number of scores it is estimated from.
        """
        known = scores.loc[
            (scores["origin"] < origin) & (scores["truth_vintage"] <= origin)
        ]
        if known.empty:
            return self.prior, 0
        last = pd.Timestamp(known["target_end_date"].max())
        first = last - pd.Timedelta(weeks=self.window_weeks)
        used = known.loc[pd.to_datetime(known["target_end_date"]) > first]
        copula = estimate_copula(used, self.prior, self.weight, self.national)
        return copula, len(used)

    def pair(
        self, forecasts: pd.DataFrame, truth: pd.DataFrame, seed: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Pair every model's forecast at every origin.

        Parameters
        ----------
        forecasts : pandas.DataFrame
            `model`, `origin`, `location`, `target_end_date`, `sample`
            and `value`, and optionally `as_of` and other columns.
        truth : pandas.DataFrame
            `rolling_truth`: `location`, `target_end_date`, `truth` and
            `truth_vintage`.
        seed : int
            Seed. Each forecast's score, and each model and origin's
            pairing, draw from their own streams.

        Returns
        -------
        tuple of pandas.DataFrame
            The paired forecasts, and each model and origin's fit:
            `model`, `origin`, `n_scores` and the four correlations.
        """
        cells = ["location", "target_end_date"]
        released = truth[[*cells, "truth_vintage"]]
        paired, fits = [], []
        for model, own in forecasts.groupby("model"):
            model = str(model)
            key = [seed, zlib.crc32(model.encode())]
            final = own.loc[_final(own)]
            scores = normal_scores(
                final[["model", "origin", *cells, "sample", "value"]],
                truth[[*cells, "truth"]],
                seed,
            ).merge(released, on=cells)
            for origin, rows in own.groupby("origin"):
                origin = str(origin)
                copula, n = self.fit(scores, origin)
                rng = np.random.default_rng([*key, zlib.crc32(origin.encode())])
                paired.append(pair_samples(rows, copula, rng, self.national))
                fits.append(
                    {
                        "model": model,
                        "origin": origin,
                        "n_scores": n,
                        **dataclasses.asdict(copula),
                    }
                )
        out = pd.concat(paired, ignore_index=True)[list(forecasts.columns)]
        return out, pd.DataFrame(fits, columns=_FIT_COLUMNS)


def pair_registered(
    forecasts: pd.DataFrame,
    truth: pd.DataFrame,
    seed: int,
    national: str = NATIONAL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair the forecasts of the models registered as `paired`.

    Each is paired by `RollingCopula` with its defaults. Other models'
    rows are left as they are.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        As for `RollingCopula.pair`.
    truth : pandas.DataFrame
        As for `RollingCopula.pair`.
    seed : int
        Seed.
    national : str
        The country's location code.

    Returns
    -------
    tuple of pandas.DataFrame
        The forecasts, and each paired model and origin's fit.
    """
    names = {s.name for s in list_models() if s.paired}
    mine = forecasts["model"].isin(names)
    if not mine.any():
        return forecasts, pd.DataFrame(columns=_FIT_COLUMNS)
    paired, fits = RollingCopula(national=national).pair(
        forecasts.loc[mine], truth, seed
    )
    out = pd.concat([forecasts.loc[~mine], paired], ignore_index=True)
    return out, fits


def error_scores(
    added: np.ndarray,
    remaining: np.ndarray,
    valid: np.ndarray,
    size: np.ndarray,
    releases: list[str],
    ages: np.ndarray,
    locations: tuple[str, ...],
    model: str,
    seed: int,
    window_weeks: int = WINDOW_WEEKS,
) -> pd.DataFrame:
    r"""Return normal scores of a model's own nowcasts at earlier releases.

    At each earlier release $r$ the model nowcast the count still to come
    of each place and week back, as a negative binomial with mean
    `remaining` and size $\phi_k$; the origin release reports what came
    (`added`). The score is $\Phi^{-1}$ of the randomised PIT of `added`
    in that distribution, with the uniform drawn from a hash, as
    `normal_scores`. Releases more than `window_weeks` weeks before the
    origin are left out, as are cells with no nowcast spread.

    Parameters
    ----------
    added, remaining, valid : numpy.ndarray
        `chain.Errors` fields, `[R, L, K]`.
    size : numpy.ndarray
        Negative binomial size of each place and week back, `[L, K]`.
    releases : list of str
        The releases, `[R]`.
    ages : numpy.ndarray
        Days from each release's cut-off to the origin's, `[R]`.
    locations : tuple of str
        The places, `[L]`.
    model : str
        The model, as the scores' `model`.
    seed : int
        Seed of the uniform draws.
    window_weeks : int
        The latest weeks of releases used.

    Returns
    -------
    pandas.DataFrame
        `model`, `origin` (the release), `location`, `target_end_date`
        (a date 7 days apart per week back, so consecutive weeks pair) and
        `z`, as `estimate_copula` takes them.
    """
    from scipy.stats import nbinom

    cols = ["model", "origin", "location", "target_end_date", "z"]
    keep = np.asarray(ages) <= 7 * window_weeks
    if not keep.any():
        return pd.DataFrame(columns=cols)
    mu = np.asarray(remaining, float)[keep]
    y = np.asarray(added, float)[keep]
    ok = np.asarray(valid, bool)[keep] & (mu > 0)
    phi = np.broadcast_to(np.asarray(size, float)[None], mu.shape)
    ok &= np.isfinite(phi) & (phi > 0)
    r, loc, k = np.nonzero(ok)
    if r.size == 0:
        return pd.DataFrame(columns=cols)
    n, m, yy = phi[r, loc, k], mu[r, loc, k], np.maximum(y[r, loc, k], 0)
    p = n / (n + m)
    below = nbinom.cdf(yy - 1, n, p)
    at = nbinom.pmf(yy, n, p)
    rel = np.asarray(releases, dtype=object)[keep][r]
    codes = np.asarray(locations, dtype=object)[loc]
    keys = [
        f"{seed}|{model}|{a}|{b}|{c}"
        for a, b, c in zip(rel, codes, k, strict=True)
    ]
    u = (np.array([zlib.crc32(x.encode()) for x in keys]) + 0.5) / 2.0**32
    pit = np.clip(below + u * at, 1e-12, 1 - 1e-12)
    week = pd.Timestamp("2000-01-01") - pd.to_timedelta(7 * k, unit="D")
    return pd.DataFrame(
        {
            "model": model,
            "origin": rel,
            "location": codes,
            "target_end_date": week.strftime("%Y-%m-%d"),
            "z": ndtri(pit),
        }
    )[cols]


def pair_walks(
    values: np.ndarray,
    locations: tuple[str, ...],
    anchors: np.ndarray,
    copula: GaussianCopula,
    rng: np.random.Generator,
    days: np.ndarray | None = None,
) -> np.ndarray:
    """Pair a walk model's draws, keeping each place's walk whole.

    Weeks of a place before its walk starts (`anchors`) are drawn on their
    own, so each is reordered as the copula's normals rank (the Schaake
    shuffle; Clark et al. 2004). From its walk start on, a place's draws
    are joint trajectories, so each moves as a whole, ranked by its total
    over the walk weeks against the start week's normal, keeping the walk's
    own dependence.

    Parameters
    ----------
    values : numpy.ndarray
        Draws `[S, L, H]`.
    locations : tuple of str
        The places' codes (`GaussianCopula.normals`).
    anchors : numpy.ndarray
        Each place's first walk week, an index into the weeks, `[L]`; at
        or beyond `H` the place has no walk.
    copula : GaussianCopula
        The copula.
    rng : numpy.random.Generator
        Random generator.
    days : numpy.ndarray, optional
        Day of each week, weekly by default.

    Returns
    -------
    numpy.ndarray
        `values` with each cell's draws reordered across trajectories.
    """
    n, _, n_steps = values.shape
    if days is None:
        days = 7 * np.arange(n_steps)
    z = copula.normals(rng, n, len(locations), days, locations=list(locations))
    rank = np.argsort(np.argsort(z, axis=0), axis=0)
    index = np.take_along_axis(
        np.argsort(values, axis=0, kind="stable"), rank, axis=0
    )
    for loc, a in enumerate(np.asarray(anchors, int)):
        if a < n_steps:
            # A walk is ranked by its total, which carries its latent path
            # better than one week's count with its own observation noise.
            total = values[:, loc, a:].sum(axis=1)
            whole = np.argsort(total, kind="stable")[rank[:, loc, a]]
            index[:, loc, a:] = whole[:, None]
    return np.take_along_axis(values, index, axis=0)
