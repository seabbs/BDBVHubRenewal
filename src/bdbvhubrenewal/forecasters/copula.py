# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""A Gaussian copula that pairs a model's marginal draws into trajectories.

A model that gives each place and step only its marginal, such as
Chronos-2's quantiles, draws each cell independently.
`GaussianCopula.couple` pairs those draws into trajectories: each
trajectory draws one correlated normal vector $z$ over places and steps,
and every cell's draws are reordered across trajectories so they rank as
$z$ does.
This is the Gaussian copula with $u = \Phi(z)$ read off each cell's own
draws, so each cell keeps exactly the draws it had.

The correlation of $z$ is a place correlation times a step correlation.
Two steps $\delta$ days apart have correlation $\rho_t^{\delta / 7}$.
Between places it is $\rho_p$ for two provinces and $\rho_n$ for the
country and a province.
`estimate_copula` estimates the three from the normal scores of matured
weeks at past origins (`normal_scores`), shrunk towards a prior.

`pair_samples` pairs one forecast table, and `RollingCopula` pairs a
model's forecasts at every origin, each with the copula estimated from
the model's own recent matured errors.
A country that is the sum of its provinces is summed again from the
paired provinces.
"""

from __future__ import annotations

import dataclasses
import zlib
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from scipy.special import ndtri

from bdbvhubrenewal.forecasters.base import NATIONAL

__all__ = [
    "MAX_RHO",
    "PRIOR",
    "PRIOR_WEIGHT",
    "WINDOW_WEEKS",
    "GaussianCopula",
    "RollingCopula",
    "estimate_copula",
    "normal_scores",
    "pair_samples",
]

MAX_RHO: Final = 0.95
"""Largest correlation `estimate_copula` gives."""

WINDOW_WEEKS: Final = 8
"""Matured weeks of errors `RollingCopula` estimates from."""

PRIOR_WEIGHT: Final = 100.0
"""Normal scores the prior counts as, in `estimate_copula`'s shrinkage."""

_CELL = ["model", "origin", "location", "target_end_date"]


def _factor(corr: np.ndarray) -> np.ndarray:
    """Return $A$ with $A A^T$ a correlation matrix near `corr`.

    Negative eigenvalues are set to zero and the rows rescaled to unit
    length. A row left with no length keeps its own unit normal.
    """
    values, vectors = np.linalg.eigh(corr)
    a = vectors * np.sqrt(np.clip(values, 0.0, None))
    norm = np.linalg.norm(a, axis=1, keepdims=True)
    empty = norm[:, 0] == 0.0
    a[empty] = np.eye(len(a))[empty]
    norm[empty] = 1.0
    return np.asarray(a / norm)


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
        $\rho_n$, between the country and a province in a week.
    """

    rho_time: float = 0.0
    rho_provinces: float = 0.0
    rho_national: float = 0.0

    def __post_init__(self) -> None:
        """Check each correlation is in [0, 1)."""
        for name in ("rho_time", "rho_provinces", "rho_national"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be between 0 and 1 (not 1).")

    def normals(
        self,
        rng: np.random.Generator,
        n: int,
        n_places: int,
        days: np.ndarray,
        national: bool = True,
    ) -> np.ndarray:
        r"""Draw correlated standard normals over places and steps.

        Where $\rho_p$ and $\rho_n$ give no correlation matrix, a nearby
        one is used (`_factor`).

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
            is a province.

        Returns
        -------
        numpy.ndarray
            `[n, n_places, len(days)]`.
        """
        place = np.full((n_places, n_places), self.rho_provinces)
        if national:
            place[0, :] = place[:, 0] = self.rho_national
        np.fill_diagonal(place, 1.0)
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

    def order(
        self,
        draws: np.ndarray,
        rng: np.random.Generator,
        days: np.ndarray | None = None,
        national: bool = True,
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
        z = self.normals(rng, n, n_places, days, national)
        rank = np.argsort(np.argsort(z, axis=0), axis=0)
        by_value = np.argsort(draws, axis=0, kind="stable")
        return np.take_along_axis(by_value, rank, axis=0)

    def couple(
        self,
        draws: np.ndarray,
        rng: np.random.Generator,
        days: np.ndarray | None = None,
        national: bool = True,
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

        Returns
        -------
        numpy.ndarray
            The same draws of each cell, reordered across trajectories.
        """
        index = self.order(draws, rng, days, national)
        return np.take_along_axis(draws, index, axis=0)


PRIOR: Final = GaussianCopula(rho_time=0.9, rho_provinces=0.7, rho_national=0.7)
"""The correlations estimates are shrunk towards, chosen on the Chronos-2
models' release backtest energy score."""


def _final(samples: pd.DataFrame) -> pd.Series:
    """Rows of final counts: those with no `as_of`, if there is one."""
    if "as_of" not in samples:
        return pd.Series(True, index=samples.index)
    return samples["as_of"].isna()


def _summed(final: pd.DataFrame, national: str) -> bool:
    """Whether the country is in `final` as the sum of the other places."""
    is_nat = final["location"] == national
    if not is_nat.any():
        return False
    keys = ["target_end_date", "sample"]
    total = final.loc[~is_nat].groupby(keys)["value"].sum()
    own = final.loc[is_nat].set_index(keys)["value"]
    return bool(np.allclose(own.to_numpy(), total.reindex(own.index)))


def pair_samples(
    samples: pd.DataFrame,
    copula: GaussianCopula,
    rng: np.random.Generator,
    national: str = NATIONAL,
) -> pd.DataFrame:
    """Pair one forecast's draws into trajectories by a copula.

    Each place and week keeps its draws, and every row of a draw, such as
    its counts as a later release reports them (`as_of`), moves with it.
    A country that is the sum of the other places is not paired but
    summed again from them.

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
    is_final = _final(samples)
    final = samples.loc[is_final]
    summed = _summed(final, national)
    places = sorted(set(final["location"]) - {national})
    own = not summed and bool((final["location"] == national).any())
    if own:
        places = [national, *places]
    weeks = sorted(set(final["target_end_date"]))
    wide = final.pivot(
        index="sample", columns=["location", "target_end_date"], values="value"
    ).reindex(columns=pd.MultiIndex.from_product([places, weeks]))
    draws = wide.to_numpy().reshape(len(wide), len(places), len(weeks))
    days = (pd.to_datetime(weeks) - pd.Timestamp(weeks[0])).days.to_numpy()
    index = copula.order(draws, rng, days, national=own)
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
    if summed:
        out = _sum_country(out, national)
    return out[list(samples.columns)].reset_index(drop=True)


def _sum_country(samples: pd.DataFrame, national: str) -> pd.DataFrame:
    """Set the country's rows to the sum of the other places' rows."""
    keys = ["target_end_date", "sample"]
    if "as_of" in samples:
        keys.append("as_of")
    is_nat = samples["location"] == national
    total = (
        samples.loc[~is_nat]
        .groupby(keys, dropna=False)["value"]
        .sum()
        .rename("total")
    )
    nat = samples.loc[is_nat].join(total, on=keys)
    samples.loc[is_nat, "value"] = nat["total"].to_numpy()
    return samples


def normal_scores(
    forecasts: pd.DataFrame, truth: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    r"""Return the normal score of each truth in its forecast's draws.

    The score is $\Phi^{-1}$ of the randomised PIT: with $m$ draws, $b$
    below the truth and $e$ equal to it, $(b + U (e + 1)) / (m + 1)$ for
    $U$ uniform.
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
    rng : numpy.random.Generator
        Random generator.

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
    u = 1.0 - rng.uniform(size=len(out))
    pit = (out["below"] + u * (out["equal"] + 1)) / (out["m"] + 1)
    return out[_CELL].assign(z=ndtri(pit.to_numpy()))


def _corr(pairs: pd.DataFrame) -> float | None:
    """Pearson correlation of `z_a` and `z_b`, or None with too few."""
    if len(pairs) < 2:
        return None
    r = np.corrcoef(pairs["z_a"], pairs["z_b"])[0, 1]
    return float(np.nan_to_num(r))


def estimate_copula(
    scores: pd.DataFrame,
    prior: GaussianCopula = GaussianCopula(),  # noqa: B008
    weight: float = 0.0,
    national: str = NATIONAL,
) -> GaussianCopula:
    r"""Estimate the copula's correlations from normal scores.

    Each is the correlation $r$ of the scores of pairs from one forecast:
    $\rho_t$ of a place's consecutive weeks, $\rho_p$ of two provinces in a
    week, and $\rho_n$ of the country and a province in a week.
    With $n$ scores and the prior's $\rho_0$ each is
    $(n r + w \rho_0) / (n + w)$, the prior's where no pair has $r$,
    then floored at zero and capped at `MAX_RHO`.

    Parameters
    ----------
    scores : pandas.DataFrame
        `normal_scores`: `model`, `origin`, `location`,
        `target_end_date` and `z`.
    prior : GaussianCopula
        The correlations to shrink towards.
    weight : float
        $w$, the scores the prior counts as.
    national : str
        The country's location code.

    Returns
    -------
    GaussianCopula
    """
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
    provinces = same.loc[
        ~a_national & ~b_national & (same["location_a"] < same["location_b"])
    ]
    pairs = {
        "rho_time": time,
        "rho_provinces": provinces,
        "rho_national": same.loc[a_national & ~b_national],
    }
    n = len(scores)
    out = {}
    for name, rows in pairs.items():
        r, rho0 = _corr(rows), getattr(prior, name)
        rho = rho0 if r is None else (n * r + weight * rho0) / (n + weight)
        out[name] = float(np.clip(rho, 0.0, MAX_RHO))
    return GaussianCopula(**out)


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
        The scores the prior counts as (`estimate_copula`).
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
            Seed. Each model and origin draws from its own stream.

        Returns
        -------
        tuple of pandas.DataFrame
            The paired forecasts, and each model and origin's fit:
            `model`, `origin`, `n_scores` and the three correlations.
        """
        cells = ["location", "target_end_date"]
        final = forecasts.loc[_final(forecasts)]
        scores = normal_scores(
            final[["model", "origin", *cells, "sample", "value"]],
            truth[[*cells, "truth"]],
            np.random.default_rng(seed),
        ).merge(truth[[*cells, "truth_vintage"]], on=cells)
        paired, fits = [], []
        for key, rows in forecasts.groupby(["model", "origin"]):
            model, origin = (str(k) for k in key)
            copula, n = self.fit(scores.loc[scores["model"] == model], origin)
            rng = np.random.default_rng(
                [
                    seed,
                    zlib.crc32(model.encode()),
                    zlib.crc32(origin.encode()),
                ]
            )
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
        return out, pd.DataFrame(fits)
