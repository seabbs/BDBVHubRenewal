# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Simulate province outbreaks and publish them as hub-style vintages.

The simulator draws from the model's own components with known
parameters, so recovery can be checked and there is one implementation
of the maths. Expected daily onsets come from the model's renewal
(`bdbv_hub_renewal.model.renewal.coupled_renewal`), with its generation
interval, gravity kernel, importation and depletion of the effective
population, for $\log R_p(t) = \log R(t) + \delta_p$ with a national
$R(t)$ that turns and fixed province offsets $\delta_p$.
Confirmed cases are Poisson with mean $\alpha(t)$ times expected onsets.
Each is reported after a double censored Gamma delay
(`bdbv_hub_renewal.model.delays.censored_pmf`) whose mean $\mu(t)$
changes over time, has no onset with probability $\pi$ at its report
day, and is published after a lag of 0 to 3 days.

`SimulatedOutbreak.vintage` publishes what the hub would show at a date:
known onsets by onset week, missing onsets given an onset by subtracting
a delay drawn from the pooled known delays reported so far (the hub's
back-sampling), their total, and the delay line data.
`SimulatedOutbreak.truth` is the final total once every case is reported.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from bdbv_hub_renewal.data import PROVINCE_POPULATIONS
from bdbv_hub_renewal.data.dates import as_date, epiweek_end
from bdbv_hub_renewal.forecasters.base import NATIONAL
from bdbv_hub_renewal.model.delays import censored_pmf
from bdbv_hub_renewal.model.inputs import (
    gravity_kernel as model_gravity_kernel,
)
from bdbv_hub_renewal.model.model import (
    GI_MEAN,
    GI_SD,
    PRIOR_PUBLICATION_LAG,
    generation_interval,
)
from bdbv_hub_renewal.model.renewal import coupled_renewal, mixing_matrix

__all__ = [
    "SimulatedOutbreak",
    "SimulationSettings",
    "simulate_outbreak",
]

MAX_SIM_DELAY = 120
"""Longest simulated onset-to-report delay in days."""


def _logistic(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(1 / (1 + np.exp(-x)), dtype=np.float64)


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
        The default is the model's `PRIOR_PUBLICATION_LAG`: 85% uniform
        on 0..3 days and 15% uniform on 4..60 days, a tail of late
        entries as in the real delay files.
    backfill_share : float
        Share of cases reported with no onset whose onset is recorded
        later.
    backfill_delay : tuple of int
        Range of days from publication to the onset being recorded.
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
    publication_lag: tuple[float, ...] = PRIOR_PUBLICATION_LAG
    backfill_share: float = 0.3
    backfill_delay: tuple[int, int] = (7, 42)
    gi_mean: float = GI_MEAN
    gi_sd: float = GI_SD
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
        `report_date`, `publish_date`, `onset_known` (recorded when first
        published) and `onset_known_date` (when the onset is recorded,
        NaT if never).
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
        cut = pd.Timestamp(day)
        seen = self.cases.loc[self.cases["publish_date"] <= cut]
        # A case counts as known once its onset is recorded, which for a
        # backfilled case is after it was first published.
        has_onset = seen["onset_known_date"] <= cut
        known = seen.loc[has_onset]
        missing = seen.loc[~has_onset]
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
    gi = generation_interval(s.gi_mean, s.gi_sd)
    turn = _logistic((t - s.turn_day) / s.turn_width)
    log_r = np.log(s.r_start) + turn * (np.log(s.r_end) - np.log(s.r_start))
    rt = np.exp(log_r[None, :] + np.array(s.log_r_offset)[:, None])
    # Expected onsets from the model's own renewal, per resident with
    # the effective share at risk as the pool, as in `build_model`.
    pop = np.array([PROVINCE_POPULATIONS[p] for p in s.locations], float)
    seeding = np.zeros((n_days, n_loc))
    for i, day in enumerate(s.seed_day):
        if day >= 0:
            seeding[day, i] = s.seed_size / pop[i]
    table = pd.DataFrame({"location": s.locations, "population": pop})
    mixing = mixing_matrix(
        jnp.asarray(model_gravity_kernel(s.locations, table)),
        s.importation,
        jnp.asarray(pop / pop.sum()),
    )
    nu = s.effective_fraction
    x, r_eff = coupled_renewal(
        jnp.zeros((gi.size, n_loc)),
        jnp.asarray(rt.T),
        jnp.asarray(gi),
        mixing,
        jnp.full(n_loc, nu),
        pool=nu,
        seeding=jnp.asarray(seeding),
    )
    onsets = (np.asarray(x, float) * pop).T
    sus = (np.asarray(r_eff, float) / rt.T).T

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
    # Daily double censored Gamma delays, as the model's `censored_pmf`,
    # long enough that truncation is negligible.
    delay_cdf = np.cumsum(
        np.asarray(
            jax.vmap(
                lambda m: censored_pmf("gamma", m, s.delay_cv, MAX_SIM_DELAY)
            )(jnp.asarray(mu))
        ),
        axis=1,
    )

    lag_pmf = np.asarray(s.publication_lag) / np.sum(s.publication_lag)
    rows = []
    for i, loc in enumerate(s.locations):
        confirmed = rng.poisson(asc * onsets[i])
        days = np.repeat(np.arange(n_days), confirmed)
        u = rng.uniform(size=days.size) * delay_cdf[days, -1]
        delay = (delay_cdf[days] < u[:, None]).sum(axis=1)
        report = days + delay
        report_day = np.minimum(report.astype(int), n_days - 1)
        known = rng.uniform(size=days.size) >= miss[report_day]
        lag = rng.choice(len(s.publication_lag), days.size, p=lag_pmf)
        publish = report.astype(int) + lag
        lo, hi = s.backfill_delay
        filled = ~known & (rng.uniform(size=days.size) < s.backfill_share)
        later = publish + rng.integers(lo, hi + 1, size=days.size)
        onset_known_day = np.where(
            known, publish, np.where(filled, later, -1)
        ).astype(float)
        onset_known_day[onset_known_day < 0] = np.nan
        rows.append(
            pd.DataFrame(
                {
                    "location": loc,
                    "onset_day": days,
                    "report_day": report.astype(int),
                    "publish_day": publish,
                    "onset_known_day": onset_known_day,
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
    cases["onset_known_date"] = start + pd.to_timedelta(
        cases["onset_known_day"], unit="D"
    )
    cases = cases[
        [
            "location",
            "onset_date",
            "report_date",
            "publish_date",
            "onset_known",
            "onset_known_date",
        ]
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
