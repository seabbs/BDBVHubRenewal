# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Simulate province outbreaks and publish them as hub-style vintages.

The simulator follows the renewal model's structure, with known
parameters, so recovery can be checked.
Daily symptom onsets in province $p$ follow a renewal process with
spatial importation and susceptible depletion,

$$
\lambda_p(t) = R_p(t)\,\frac{S_p(t)}{N_p}\sum_{s \ge 1} g(s)
\Big[(1 - \varepsilon)\,I_p(t - s) + \varepsilon\sum_{q \ne p}
K_{qp}\,I_q(t - s)\Big],
\quad I_p(t) \sim \text{Poisson}(\lambda_p(t)),
$$

with $\log R_p(t) = \log R(t) + \delta_p$, a national $R(t)$ with a
turning point, fixed province offsets $\delta_p$, a gravity kernel $K$
and an effective population $N_p$.
Each onset is confirmed with probability $\alpha(t)$ (ascertainment),
reported after a delay with a Gamma distribution whose mean $\mu(t)$
changes over time, and has its onset recorded with probability
$1 - \pi(t)$.

`SimulatedOutbreak.vintage` publishes what the hub would show at a date:
known onsets by onset week, missing onsets given an onset by subtracting
a delay drawn from the pooled known delays reported so far (the hub's
back-sampling), their total, and the delay line data.
`SimulatedOutbreak.truth` is the final total once every case is reported.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .data import PROVINCE_CENTRES, PROVINCE_POPULATIONS
from .dates import as_date, epiweek_end
from .delays import ParametricDelay
from .latent import renewal
from .models.base import NATIONAL

__all__ = [
    "SimulatedOutbreak",
    "SimulationSettings",
    "gravity_kernel",
    "simulate_outbreak",
]


def _logistic(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(1 / (1 + np.exp(-x)), dtype=np.float64)


def gravity_kernel(
    locations: tuple[str, ...], decay_km: float = 300.0
) -> NDArray[np.float64]:
    r"""Return a row-normalised gravity kernel between provinces.

    $K_{qp} \propto N_p \exp(-D_{qp} / \text{decay})$ for $q \ne p$,
    with great-circle distance $D$ between province centres.

    Parameters
    ----------
    locations : tuple of str
        Province codes with known centres.
    decay_km : float
        Distance scale in kilometres.

    Returns
    -------
    numpy.ndarray
        Matrix with rows summing to one and a zero diagonal.
    """
    lat = np.radians([PROVINCE_CENTRES[p][0] for p in locations])
    lon = np.radians([PROVINCE_CENTRES[p][1] for p in locations])
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    )
    dist = 2 * 6371 * np.arcsin(np.sqrt(a))
    pop = np.array([PROVINCE_POPULATIONS[p] for p in locations], dtype=float)
    k = pop[None, :] * np.exp(-dist / decay_km)
    np.fill_diagonal(k, 0.0)
    return np.asarray(k / k.sum(axis=1, keepdims=True), dtype=np.float64)


@dataclass(frozen=True)
class SimulationSettings:
    r"""Known parameters of a simulated outbreak.

    Times are days from `start`.

    Attributes
    ----------
    name : str
        Scenario name.
    start : datetime.date
        Day 0, a Sunday.
    n_days : int
        Days simulated.
    locations : tuple of str
        Provinces.
    seed_day : tuple of int
        Day each province is seeded (-1 for importation only).
    seed_size : int
        Onsets seeded on the seed day.
    log_r_offset : tuple of float
        Province offsets $\delta_p$ on log R.
    r_start, r_end : float
        National R before and after the turn.
    turn_day : int
        Centre of the turn in R.
    turn_width : float
        Days over which R turns (logistic scale).
    importation : float
        Share $\varepsilon$ of transmission sent to other provinces.
    effective_fraction : float
        Effective population as a share of the resident population.
    ascertainment : tuple of float
        $\alpha$ at day 0 and at the end, changing logistically mid-way.
    missing : tuple of float
        $\pi$ at day 0, at its peak (day `n_days` / 3) and at the end,
        by report day as in the model.
    delay_mean : tuple of float
        Mean delay at day 0 and at the end, changing linearly.
    delay_cv : float
        Coefficient of variation of the delay.
    publication_lag : tuple of float
        Pmf of the days from report to publication in a vintage, from 0.
        The default is uniform on 0..3, as `vintages.PUBLICATION_LAG_DAYS`
        and the data suggest.
    gi_mean, gi_sd : float
        Generation interval mean and SD in days.
    seed : int
        Random seed.
    """

    name: str = "turning_point"
    start: dt.date = dt.date(2026, 4, 5)
    n_days: int = 238
    locations: tuple[str, ...] = ("CD-IT", "CD-NK", "CD-HU", "CD-TO")
    seed_day: tuple[int, ...] = (0, 30, -1, -1)
    seed_size: int = 5
    log_r_offset: tuple[float, ...] = (0.05, 0.0, -0.05, -0.15)
    r_start: float = 1.6
    r_end: float = 0.85
    turn_day: int = 120
    turn_width: float = 7.0
    importation: float = 0.01
    effective_fraction: float = 0.005
    ascertainment: tuple[float, float] = (0.5, 0.75)
    missing: tuple[float, float, float] = (0.05, 0.25, 0.12)
    delay_mean: tuple[float, float] = (12.0, 6.0)
    delay_cv: float = 1.0
    publication_lag: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    gi_mean: float = 15.0
    gi_sd: float = 9.0
    seed: int = 1


@dataclass(frozen=True)
class SimulatedOutbreak:
    """A simulated outbreak: the true parameters and the case line list.

    Attributes
    ----------
    settings : SimulationSettings
        The settings used.
    parameters : pandas.DataFrame
        Daily true values by province: `date`, `location`, `rt`,
        `susceptible_fraction`, `onsets`, `ascertainment`, `missing` and
        `delay_mean`.
    cases : pandas.DataFrame
        One row per confirmed case: `location`, `onset_date`,
        `report_date`, `publish_date` and `onset_known`.
    """

    settings: SimulationSettings
    parameters: pd.DataFrame
    cases: pd.DataFrame = field(repr=False)

    def vintage(
        self, as_of: dt.date | str, rng: np.random.Generator | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Publish the data the hub would show at a date.

        Parameters
        ----------
        as_of : datetime.date or str
            The vintage date.
        rng : numpy.random.Generator, optional
            For the back-sampling draws. Seeded from the date by default.

        Returns
        -------
        tuple of pandas.DataFrame
            The target rows (as `bdbv_hub_renewal.data.tidy_target`, with
            `kind = "simulated"`) and the delay line data.
        """
        day = as_date(as_of)
        rng = rng or np.random.default_rng(
            [self.settings.seed, day.toordinal()]
        )
        seen = self.cases.loc[self.cases["publish_date"] <= pd.Timestamp(day)]
        known = seen.loc[seen["onset_known"]]
        missing = seen.loc[~seen["onset_known"]]
        delays = (known["report_date"] - known["onset_date"]).dt.days.to_numpy()
        if delays.size:
            draws = rng.choice(delays, size=len(missing), replace=True)
        else:
            draws = np.zeros(len(missing), dtype=int)
        imputed_onset = missing["report_date"] - pd.to_timedelta(
            draws, unit="D"
        )

        def weekly(loc: pd.Series, onset: pd.Series) -> pd.Series:
            week = onset.map(lambda x: epiweek_end(x.date()).isoformat())
            return pd.DataFrame({"location": loc, "week": week}).value_counts()

        obs = weekly(known["location"], known["onset_date"]).rename("observed")
        imp = weekly(missing["location"], imputed_onset).rename("imputed")
        weeks = pd.date_range(
            epiweek_end(self.settings.start), epiweek_end(day), freq="7D"
        ).strftime("%Y-%m-%d")
        index = pd.MultiIndex.from_product(
            [list(self.settings.locations), list(weeks)],
            names=["location", "week"],
        )
        tab = pd.concat([obs, imp], axis=1).reindex(index).fillna(0)
        tab = tab.astype("int64").reset_index()
        tab = tab.rename(columns={"week": "target_end_date"})
        nat = tab.groupby("target_end_date", as_index=False)[
            ["observed", "imputed"]
        ].sum()
        nat["location"] = NATIONAL
        tab = pd.concat([tab, nat], ignore_index=True)
        tab["total"] = tab["observed"] + tab["imputed"]
        tab["location_name"] = tab["location"]
        tab["kind"] = "simulated"
        tab["vintage"] = day.isoformat()
        cols = [
            "kind",
            "vintage",
            "location",
            "location_name",
            "target_end_date",
            "observed",
            "imputed",
            "total",
        ]
        delay_rows = pd.DataFrame(
            {
                "vintage": day.isoformat(),
                "location": known["location"].to_numpy(),
                "location_name": known["location"].to_numpy(),
                "onset_week": [
                    epiweek_end(x.date()).isoformat()
                    for x in known["onset_date"]
                ],
                "delay": delays.astype("int64"),
            }
        )
        return tab[cols], delay_rows

    def vintages(
        self, dates: list[dt.date] | list[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Publish several vintages and stack them.

        Parameters
        ----------
        dates : list of datetime.date or str
            Vintage dates.

        Returns
        -------
        tuple of pandas.DataFrame
            Stacked target rows and delay line data.
        """
        parts = [self.vintage(d) for d in dates]
        return (
            pd.concat([p[0] for p in parts], ignore_index=True),
            pd.concat([p[1] for p in parts], ignore_index=True),
        )

    def truth(self) -> pd.DataFrame:
        """Return the final weekly totals by onset week.

        The final total of a week is every confirmed case with onset in
        it, whether or not the onset was recorded. The hub's imputation
        moves missing onsets between weeks, so this is the target only
        approximately.

        Returns
        -------
        pandas.DataFrame
            Columns `location`, `target_end_date` and `truth`.
        """
        week = self.cases["onset_date"].map(
            lambda x: epiweek_end(x.date()).isoformat()
        )
        tab = (
            pd.DataFrame({"location": self.cases["location"], "week": week})
            .value_counts()
            .rename("truth")
            .reset_index()
            .rename(columns={"week": "target_end_date"})
        )
        nat = tab.groupby("target_end_date")["truth"].sum().reset_index()
        nat["location"] = NATIONAL
        out = pd.concat([tab, nat], ignore_index=True)
        return out[["location", "target_end_date", "truth"]]


def simulate_outbreak(
    settings: SimulationSettings | None = None,
) -> SimulatedOutbreak:
    """Simulate an outbreak with known parameters.

    Parameters
    ----------
    settings : SimulationSettings, optional
        Defaults to the `turning_point` scenario.

    Returns
    -------
    SimulatedOutbreak
        The parameters and line list.
    """
    s = settings or SimulationSettings()
    rng = np.random.default_rng(s.seed)
    n_loc, n_days = len(s.locations), s.n_days
    t = np.arange(n_days, dtype=float)
    gi = ParametricDelay("gamma", s.gi_mean, s.gi_sd).pmf(60)
    turn = _logistic((t - s.turn_day) / s.turn_width)
    log_r = np.log(s.r_start) + turn * (np.log(s.r_end) - np.log(s.r_start))
    rt = np.exp(log_r[None, :] + np.array(s.log_r_offset)[:, None])
    pop = np.array(
        [PROVINCE_POPULATIONS[p] * s.effective_fraction for p in s.locations]
    )
    seeding = np.zeros((n_loc, n_days))
    for i, day in enumerate(s.seed_day):
        if day >= 0:
            seeding[i, day] = s.seed_size
    onsets, sus = renewal(
        rt,
        gi,
        seeding,
        pop,
        kernel=gravity_kernel(s.locations),
        importation=s.importation,
        rng=rng,
    )

    frac = _logistic((t - n_days / 2) / 14)
    asc = s.ascertainment[0] + frac * (s.ascertainment[1] - s.ascertainment[0])
    peak = n_days / 3
    pi0, pi_peak, pi_end = s.missing
    miss = np.where(
        t <= peak,
        pi0 + (pi_peak - pi0) * t / peak,
        pi_peak + (pi_end - pi_peak) * (t - peak) / (n_days - peak),
    )
    mu = s.delay_mean[0] + (s.delay_mean[1] - s.delay_mean[0]) * t / n_days

    lag_pmf = np.asarray(s.publication_lag) / np.sum(s.publication_lag)
    rows = []
    for i, loc in enumerate(s.locations):
        confirmed = rng.binomial(onsets[i].astype(int), asc)
        days = np.repeat(np.arange(n_days), confirmed)
        shape = 1 / s.delay_cv**2
        delay = rng.gamma(shape, mu[days] / shape)
        report = days + np.floor(delay + rng.uniform(size=days.size))
        report_day = np.minimum(report.astype(int), n_days - 1)
        known = rng.uniform(size=days.size) >= miss[report_day]
        lag = rng.choice(len(s.publication_lag), days.size, p=lag_pmf)
        rows.append(
            pd.DataFrame(
                {
                    "location": loc,
                    "onset_day": days,
                    "report_day": report.astype(int),
                    "publish_day": report.astype(int) + lag,
                    "onset_known": known,
                }
            )
        )
    cases = pd.concat(rows, ignore_index=True)
    start = pd.Timestamp(s.start)
    cases["onset_date"] = start + pd.to_timedelta(cases["onset_day"], unit="D")
    cases["report_date"] = start + pd.to_timedelta(
        cases["report_day"], unit="D"
    )
    cases["publish_date"] = start + pd.to_timedelta(
        cases["publish_day"], unit="D"
    )
    cases = cases[
        ["location", "onset_date", "report_date", "publish_date", "onset_known"]
    ]

    dates = pd.date_range(start, periods=n_days, freq="D")
    params = pd.DataFrame(
        {
            "date": np.tile(dates, n_loc),
            "location": np.repeat(s.locations, n_days),
            "rt": rt.reshape(-1),
            "susceptible_fraction": sus.reshape(-1),
            "onsets": onsets.reshape(-1),
            "ascertainment": np.tile(asc, n_loc),
            "missing": np.tile(miss, n_loc),
            "delay_mean": np.tile(mu, n_loc),
        }
    )
    return SimulatedOutbreak(s, params, cases)
