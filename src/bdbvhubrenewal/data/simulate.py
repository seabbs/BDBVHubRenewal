# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Simulate province outbreaks and publish them as hub-style vintages.

The simulator draws from the model's own components with known
parameters, so recovery can be checked and there is one implementation
of the maths. Expected daily onsets come from the model's renewal
(`bdbvhubrenewal.model.renewal.coupled_renewal`), with its generation
interval, gravity kernel, importation and depletion of the effective
population, for $\log R_p(t) = \log R(t) + \delta_p$ with a national
$R(t)$ that turns and fixed province offsets $\delta_p$.
Confirmed cases are Poisson with mean $\alpha(t)$ times expected onsets.
Each is reported after a double censored Gamma delay
(`bdbvhubrenewal.model.delays.censored_pmf`) whose mean $\mu(t)$
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

from bdbvhubrenewal.data import PROVINCE_POPULATIONS
from bdbvhubrenewal.data.dates import as_date, epiweek_end
from bdbvhubrenewal.forecasters.base import NATIONAL
from bdbvhubrenewal.model.delays import censored_pmf
from bdbvhubrenewal.model.inputs import (
    gravity_kernel as model_gravity_kernel,
)
from bdbvhubrenewal.model.model import (
    GI_MEAN,
    GI_SD,
    PRIOR_PUBLICATION_LAG,
    generation_interval,
)
from bdbvhubrenewal.model.renewal import coupled_renewal, mixing_matrix

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
    rt_paths, seed_curves : tuple of tuple of float, optional
        Daily Rt and seed onsets by province, `(P, n_days)`, in place of
        the turning-point Rt and single seeds (`calibrated_settings`).
    delay_pmfs : tuple of tuple of float, optional
        Delay pmf on 0..60 days by onset day, in place of the Gamma delay.
        Days past the last row take the last row.
    missing_path : tuple of float, optional
        $\pi$ by report day, in place of the piecewise linear share.
    removal_share : float
        Share of cases removed from the data later (downward revisions).
    removal_delay : tuple of int
        Range of days from publication to removal.
    zone_shares : tuple of tuple, optional
        `(province, zone, share)` rows: each case is placed in a zone of
        its province with these shares.
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
    rt_paths: tuple[tuple[float, ...], ...] | None = None
    seed_curves: tuple[tuple[float, ...], ...] | None = None
    delay_pmfs: tuple[tuple[float, ...], ...] | None = None
    missing_path: tuple[float, ...] | None = None
    removal_share: float = 0.0
    removal_delay: tuple[int, int] = (7, 42)
    zone_shares: tuple[tuple[str, str, float], ...] = ()


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
            The target rows (as `bdbvhubrenewal.data.tidy_target`, with
            `kind = "simulated"`) and the delay line data.
        """
        day = as_date(as_of)
        known, missing, imputed_onset = self._published(day, rng)
        delays = (known["report_date"] - known["onset_date"]).dt.days.to_numpy()

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

    def _published(
        self, day: dt.date, rng: np.random.Generator | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Cases in the data at `day`: known, missing and imputed onsets."""
        rng = rng or np.random.default_rng(
            [self.settings.seed, day.toordinal()]
        )
        cut = pd.Timestamp(day)
        cases = self.cases
        seen = cases.loc[
            (cases["publish_date"] <= cut) & ~(cases["removed_date"] <= cut)
        ]
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
        return known, missing, imputed_onset

    def zone_vintage(
        self, as_of: dt.date | str, rng: np.random.Generator | None = None
    ) -> pd.DataFrame:
        """Publish the health zone totals the hub would show at a date.

        Each case keeps its zone; missing onsets take the same imputed
        onset as in `vintage`, so zones add up to provinces.

        Parameters
        ----------
        as_of : datetime.date or str
            The vintage date.
        rng : numpy.random.Generator, optional
            As for `vintage`.

        Returns
        -------
        pandas.DataFrame
            Rows as `bdbvhubrenewal.data.load_health_zones`: `vintage`,
            `location`, `location_name`, `province`, `target_end_date`,
            with `observed` (known onsets) and `imputed` beside `total`.
        """
        day = as_date(as_of)
        known, missing, imputed_onset = self._published(day, rng)
        onset = pd.concat([known["onset_date"], imputed_onset])
        both = pd.concat([known, missing])
        week = onset.map(lambda x: epiweek_end(x.date()).isoformat())
        cases = pd.DataFrame(
            {
                "location": both["zone"].to_numpy(),
                "province": both["location"].to_numpy(),
                "target_end_date": week.to_numpy(),
                "observed": np.r_[
                    np.ones(len(known), int), np.zeros(len(missing), int)
                ],
            }
        )
        cases["imputed"] = 1 - cases["observed"]
        out = cases.groupby(
            ["location", "province", "target_end_date"], as_index=False
        )[["observed", "imputed"]].sum()
        out["total"] = out["observed"] + out["imputed"]
        out = out.loc[out["location"] != ""]
        out["location_name"] = out["location"]
        out["vintage"] = day.isoformat()
        cols = [
            "vintage",
            "location",
            "location_name",
            "province",
            "target_end_date",
            "observed",
            "imputed",
            "total",
        ]
        return out[cols].reset_index(drop=True)

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
        kept = self.cases.loc[self.cases["removed_date"].isna()]
        week = kept["onset_date"].map(
            lambda x: epiweek_end(x.date()).isoformat()
        )
        tab = (
            pd.DataFrame({"location": kept["location"], "week": week})
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
    if s.rt_paths is not None and s.seed_curves is not None:
        rt = np.asarray(s.rt_paths, dtype=float)
        seeding = np.asarray(s.seed_curves, dtype=float).T / pop
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
    # Share of the pool still susceptible: R_eff / R, 1 where R is 0.
    sus = np.divide(
        np.asarray(r_eff, float).T, rt, out=np.ones_like(rt), where=rt > 0
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
    if s.missing_path is not None:
        path = np.asarray(s.missing_path, dtype=float)
        miss = path[np.minimum(np.arange(n_days), path.size - 1)]
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
    if s.delay_pmfs is not None:
        rows_pmf = np.asarray(s.delay_pmfs, dtype=float)
        rows_pmf = rows_pmf[np.minimum(np.arange(n_days), len(rows_pmf) - 1)]
        delay_cdf = np.cumsum(rows_pmf, axis=1)
        mu = rows_pmf @ np.arange(rows_pmf.shape[1])

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
        removed = rng.uniform(size=days.size) < s.removal_share
        lo, hi = s.removal_delay
        removed_day = np.where(
            removed, publish + rng.integers(lo, hi + 1, size=days.size), -1
        ).astype(float)
        removed_day[removed_day < 0] = np.nan
        zone = np.full(days.size, "", dtype=object)
        mine = [(z, w) for p, z, w in s.zone_shares if p == loc]
        if mine:
            names, weights = zip(*mine, strict=True)
            share = np.asarray(weights, dtype=float)
            zone = rng.choice(
                np.asarray(names), days.size, p=share / share.sum()
            )
        rows.append(
            pd.DataFrame(
                {
                    "location": loc,
                    "onset_day": days,
                    "report_day": report.astype(int),
                    "publish_day": publish,
                    "onset_known_day": onset_known_day,
                    "removed_day": removed_day,
                    "zone": zone,
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
    cases["removed_date"] = start + pd.to_timedelta(
        cases["removed_day"], unit="D"
    )
    cases = cases[
        [
            "location",
            "zone",
            "onset_date",
            "report_date",
            "publish_date",
            "onset_known",
            "onset_known_date",
            "removed_date",
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


RT_CAP = 3.0
"""Largest Rt in a calibrated scenario; onsets beyond it are seeds."""

SCENARIO_RT_SCALE = {"falling": 0.75, "flat": 1.0, "rising": 1.25}
"""Rt after the data, as a multiple of each province's Rt over the last
three weeks of the data period, for each calibrated scenario."""


def _weekly_delay_pmfs(
    delays: pd.DataFrame, weeks: list[str], as_of: str, dmax: int = 60
) -> np.ndarray:
    """Delay pmf by onset week, completed for right truncation.

    Delays up to the longest seen for a week come from that week and its
    neighbours; the rest of the mass and its shape come from the weeks
    with full follow-up, as `empirical` delays would be with no
    truncation.
    """
    rows = delays.loc[(delays["delay"] >= 0) & (delays["delay"] <= dmax)]
    full = pd.to_datetime(rows["onset_week"]) <= pd.Timestamp(as_of) - (
        pd.Timedelta(days=dmax + 7)
    )
    base = np.bincount(rows.loc[full, "delay"], minlength=dmax + 1) + 0.5
    base = base / base.sum()
    counts = (
        rows.groupby(["onset_week", "delay"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=weeks, columns=range(dmax + 1), fill_value=0)
        .to_numpy(dtype=float)
    )
    out = np.empty((len(weeks), dmax + 1))
    for w, week in enumerate(weeks):
        lo, hi = max(0, w - 2), min(len(weeks), w + 3)
        c = counts[lo:hi].sum(axis=0) + base
        cap = int(
            np.clip((pd.Timestamp(as_of) - pd.Timestamp(week)).days, 0, dmax)
        )
        tail = base[cap + 1 :].sum()
        out[w, : cap + 1] = (1 - tail) * c[: cap + 1] / c[: cap + 1].sum()
        out[w, cap + 1 :] = base[cap + 1 :]
    return out


def calibrated_settings(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    zones: pd.DataFrame,
    as_of: str,
    name: str = "flat",
    weeks_after: int = 10,
    settled_weeks: int = 4,
    seed: int = 1,
) -> SimulationSettings:
    r"""Build a scenario calibrated to the real data of one vintage.

    Each province's expected confirmed onsets follow its real weekly
    totals at `as_of`, spread over the days of each week and smoothed,
    up to `settled_weeks` before `as_of` (later weeks are still being
    reported). Rt is the ratio of onsets to the force of infection under
    the model's generation interval, $R_p(t) = \lambda_p(t) / \sum_s g(s)
    \lambda_p(t - s)$, capped at `RT_CAP`, and onsets beyond what that
    explains (first cases and reintroductions) are seeds, so the model's
    renewal reproduces the real curve. After that Rt is the
    mean of its last three weeks times `SCENARIO_RT_SCALE[name]`, for
    `weeks_after` weeks past `as_of`. Ascertainment is 1, importation and
    depletion are off, so the curve is the confirmed onsets themselves.

    The delay is the real delay by onset week (`_weekly_delay_pmfs`),
    the missing-onset share is the real imputed share by onset week moved
    later by the mean delay to date it by report day, and health zones
    take the real share of each province's cases by zone. Late entries
    and onset backfill keep the simulator's defaults, and 2% of cases are
    removed 7 to 42 days after publication, so totals can be revised
    down.

    Parameters
    ----------
    target, delays, zones : pandas.DataFrame
        Real target rows, delay rows and health zone rows.
    as_of : str
        The real vintage to calibrate to.
    name : {"falling", "flat", "rising"}
        Rt after the data.
    weeks_after : int
        Weeks simulated past `as_of`.
    settled_weeks : int
        Weeks before `as_of` treated as still being reported.
    seed : int
        Random seed.

    Returns
    -------
    SimulationSettings
        The scenario, named `calibrated_{name}`.
    """
    real = target.loc[
        (target["vintage"] == as_of)
        & (target["kind"] == "real")
        & (target["location"] != NATIONAL)
    ]
    wide = real.pivot_table(
        index="target_end_date",
        columns="location",
        values="total",
        aggfunc="sum",
    ).fillna(0)
    wide = wide.loc[:, wide.sum() > 0]
    first_week = pd.Timestamp(
        str(wide.index[int((wide.sum(axis=1) > 0).argmax())])
    )
    start = (first_week - pd.Timedelta(days=13)).date()
    end = pd.Timestamp(as_of)
    n_days = (end - pd.Timestamp(start)).days + 1 + 7 * weeks_after
    settled = end - pd.Timedelta(weeks=settled_weeks)
    n_data = (settled - pd.Timestamp(start)).days + 1
    locs = tuple(str(c) for c in wide.columns)
    g = generation_interval()
    rt_rows, seed_rows = [], []
    for loc in locs:
        daily = np.zeros(n_days)
        for week, total in zip(wide.index, wide[loc], strict=True):
            day0 = (pd.Timestamp(str(week)) - pd.Timestamp(start)).days - 6
            if day0 >= 0 and day0 + 7 <= n_data:
                daily[day0 : day0 + 7] = total / 7
        daily = np.convolve(daily, np.ones(7) / 7, mode="same")[:n_days]
        seeds = np.zeros(n_days)
        rt = np.zeros(n_days)
        for t in range(n_data):
            past = daily[max(0, t - g.size) : t][::-1]
            force = float(g[: past.size] @ past)
            # Rt up to RT_CAP carries the curve, and onsets beyond what
            # transmission explains (first cases, reintroductions) are
            # seeds, so the renewal gives back the real curve.
            rt[t] = min(daily[t] / force, RT_CAP) if force > 1e-6 else 0.0
            seeds[t] = max(daily[t] - rt[t] * force, 0.0)
        recent = rt[n_data - 21 : n_data]
        recent = recent[recent > 0]
        level = float(np.mean(recent)) if recent.size else 0.0
        rt[n_data:] = level * SCENARIO_RT_SCALE[name]
        rt_rows.append(tuple(rt))
        seed_rows.append(tuple(seeds))

    weeks = [
        (first_week + pd.Timedelta(weeks=k)).date().isoformat()
        for k in range(-1, n_days // 7 + 1)
    ]
    pmfs = _weekly_delay_pmfs(
        delays.loc[delays["vintage"] == as_of], weeks, as_of
    )
    daily_pmfs = np.repeat(pmfs, 7, axis=0)[:n_days]
    mean_delay = round(float(pmfs[:-1].mean(axis=0) @ np.arange(61)))
    nat = real.groupby("target_end_date")[["imputed", "total"]].sum()
    share = (nat["imputed"] + 1) / (nat["total"] + 2)
    share = share.rolling(5, center=True, min_periods=1).mean()
    miss = np.full(n_days, float(share.iloc[-settled_weeks - 1]))
    for week, value in zip(share.index, share, strict=True):
        end_day = pd.Timestamp(str(week))
        day0 = (end_day - pd.Timestamp(start)).days - 6
        if 0 <= day0 < n_days and end_day <= settled:
            miss[day0 : day0 + 7] = value
    miss = np.r_[np.full(mean_delay, miss[0]), miss][:n_days]
    zone_rows = zones.loc[zones["vintage"] == as_of]
    zone_tot = zone_rows.groupby(["province", "location"], as_index=False)[
        "total"
    ].sum()
    zone_shares = tuple(
        (str(p), str(z), float(v))
        for p, z, v in zone_tot.itertuples(index=False)
        if p in locs and v > 0
    )
    return SimulationSettings(
        name=f"calibrated_{name}",
        start=start,
        n_days=n_days,
        locations=locs,
        seed_day=tuple(-1 for _ in locs),
        importation=0.0,
        effective_fraction=1.0,
        ascertainment=(1.0, 1.0),
        rt_paths=tuple(rt_rows),
        seed_curves=tuple(seed_rows),
        delay_pmfs=tuple(tuple(r) for r in daily_pmfs),
        missing_path=tuple(miss),
        removal_share=0.02,
        zone_shares=zone_shares,
        seed=seed,
    )
