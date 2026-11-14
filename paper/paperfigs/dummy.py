"""Dummy forecasts and observations in the paper schema.

Germany has quantile forecasts; synthetic and Ebola have count samples.

These stand in for the real study outputs while the figures are drafted.
Each study gets an epidemic wave per location, reporting delays that fill
in recent counts, and models whose errors grow with the share of counts
not yet reported. Model skill is set by hand to look plausible; it says
nothing about the real models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from scipy.stats import norm

from paperfigs.schema import (
    DEFAULT_STRATUM,
    FORECAST_COLUMNS,
    OBSERVATION_COLUMNS,
    validate_forecasts,
    validate_observations,
)

__all__ = [
    "QUANTILE_LEVELS",
    "STUDY_SETUP",
    "StudySetup",
    "dummy_factors",
    "dummy_split",
    "dummy_study",
]

QUANTILE_LEVELS: Final = np.array(
    [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]
)
"""Quantile levels of the dummy forecasts."""


@dataclass(frozen=True)
class ModelSkill:
    """How a dummy model errs.

    Attributes
    ----------
    bias : float
        Mean log error of the median; positive overpredicts.
    error : float
        Scale of the log error of the median.
    spread : float
        Predictive width relative to the error; below 1 is overconfident.
    """

    bias: float
    error: float
    spread: float


@dataclass(frozen=True)
class StudySetup:
    """Shape of one dummy study.

    Attributes
    ----------
    locations : tuple of str
        Location codes.
    start : str
        First date of the series.
    n_steps : int
        Length of the series in time steps.
    step_days : int
        Days per time step.
    first_origin : int
        Step of the first origin.
    n_origins : int
        Number of origins.
    origin_every : int
        Steps between origins.
    horizons : tuple of int
        Smallest and largest horizon, in steps.
    mean_delay : float
        Mean reporting delay, in steps.
    scale : tuple of float
        Range of peak counts per step across series.
    models : dict
        Model name to `ModelSkill`.
    form : str
        `quantile` or `sample`, the forecast form the study scores.
    strata : tuple of str
        Strata (age groups) given for the first location as well as
        `all`; empty for none.
    """

    locations: tuple[str, ...]
    start: str
    n_steps: int
    step_days: int
    first_origin: int
    n_origins: int
    origin_every: int
    horizons: tuple[int, int]
    mean_delay: float
    scale: tuple[float, float]
    models: dict[str, ModelSkill]
    form: str
    strata: tuple[str, ...] = ()

    @property
    def series(self) -> list[tuple[str, str]]:
        """Return each (location, stratum) series."""
        out = [(loc, DEFAULT_STRATUM) for loc in self.locations]
        return out + [(self.locations[0], s) for s in self.strata]


_BASE = ModelSkill(0.0, 1.0, 1.4)
_RW = ModelSkill(-0.1, 1.3, 1.2)
_HUB_BASE = ModelSkill(-0.05, 1.1, 1.3)
_FM = ModelSkill(0.08, 1.15, 0.7)
_FM_COMPLETE = ModelSkill(-0.12, 1.3, 0.9)
_FM_WEEKLY_COMPLETE = ModelSkill(-0.15, 1.35, 0.95)
_FM_FT = ModelSkill(0.0, 0.75, 1.0)
_FM_WEEKLY = ModelSkill(-0.05, 1.25, 0.8)
_FM_WEEKLY_FT = ModelSkill(0.02, 0.85, 1.05)

STUDY_SETUP: Final[dict[str, StudySetup]] = {
    "synthetic": StudySetup(
        locations=tuple(f"sim{i:02d}" for i in range(1, 9)),
        start="2024-01-01",
        n_steps=200,
        step_days=1,
        first_origin=63,
        n_origins=18,
        origin_every=7,
        horizons=(-28, 0),
        mean_delay=5.0,
        scale=(40.0, 400.0),
        models={
            "randomwalk": _RW,
            "randomwalknowcast": _BASE,
            "fm_chronos_daily_complete": _FM_COMPLETE,
            "fm_chronos_daily_ragged": _FM,
            "fm_chronos_daily_ragged_ft": _FM_FT,
            "fm_chronos_weekly_complete": _FM_WEEKLY_COMPLETE,
            "fm_chronos_weekly_ragged": _FM_WEEKLY,
            "fm_chronos_weekly_ragged_ft": _FM_WEEKLY_FT,
            "epinowcast": ModelSkill(0.0, 0.65, 1.05),
        },
        form="sample",
    ),
    "germany": StudySetup(
        locations=(
            "DE",
            "DE-BW",
            "DE-BY",
            "DE-BE",
            "DE-NW",
            "DE-NI",
            "DE-HE",
            "DE-SN",
        ),
        start="2021-09-01",
        n_steps=275,
        step_days=1,
        first_origin=82,
        n_origins=23,
        origin_every=7,
        horizons=(-28, 0),
        mean_delay=6.0,
        scale=(30.0, 300.0),
        models={
            "randomwalk": _RW,
            "randomwalknowcast": _BASE,
            "fm_chronos_daily_complete": _FM_COMPLETE,
            "fm_chronos_daily_ragged": _FM,
            "fm_chronos_daily_ragged_ft": _FM_FT,
            "hub_baseline": _HUB_BASE,
            "hub_ensemble": ModelSkill(0.03, 0.8, 1.1),
            "hub_team_a": ModelSkill(0.15, 0.9, 0.9),
            "hub_team_b": ModelSkill(-0.1, 1.05, 1.3),
            "hub_team_c": ModelSkill(0.05, 1.3, 0.6),
        },
        form="quantile",
        strata=("00-04", "05-14", "15-34", "35-59", "60-79", "80+"),
    ),
    "ebola": StudySetup(
        locations=tuple(f"loc{i:02d}" for i in range(1, 7)),
        start="2025-08-04",
        n_steps=32,
        step_days=7,
        first_origin=8,
        n_origins=18,
        origin_every=1,
        horizons=(-3, 4),
        mean_delay=1.2,
        scale=(5.0, 60.0),
        models={
            "randomwalk": _RW,
            "randomwalknowcast": _BASE,
            "fm_chronos_weekly_complete": _FM_WEEKLY_COMPLETE,
            "fm_chronos_weekly_ragged": _FM_WEEKLY,
            "fm_chronos_weekly_ragged_ft": _FM_WEEKLY_FT,
            "BDBVHub-baseline": _HUB_BASE,
            "hub_ensemble": ModelSkill(0.05, 0.85, 1.0),
            "hub_team_a": ModelSkill(0.2, 1.0, 0.8),
            "hub_team_b": ModelSkill(-0.05, 1.2, 1.2),
        },
        form="sample",
    ),
}
"""Dummy setup for each study."""


def _completeness(delay: np.ndarray, mean_delay: float) -> np.ndarray:
    """Return the share of counts reported `delay` steps after the date."""
    return 1 - np.exp(-(np.asarray(delay, dtype=float) + 1) / mean_delay)


def _final_counts(setup: StudySetup, rng: np.random.Generator) -> np.ndarray:
    """Draw final counts, series by steps, as one noisy wave each."""
    t = np.arange(setup.n_steps)
    n_loc = len(setup.series)
    peak = rng.uniform(0.35, 0.65, n_loc)[:, None] * setup.n_steps
    width = rng.uniform(0.1, 0.2, n_loc)[:, None] * setup.n_steps
    scale = np.exp(rng.uniform(*np.log(setup.scale), n_loc))[:, None]
    wave = 0.08 + np.exp(-0.5 * ((t - peak) / width) ** 2)
    wiggle = np.cumsum(rng.normal(0, 0.03, (n_loc, setup.n_steps)), axis=1)
    mu = scale * wave * np.exp(wiggle - wiggle.mean(axis=1, keepdims=True))
    # Negative binomial counts with size 20.
    return rng.poisson(rng.gamma(20.0, mu / 20.0)).astype(float)


def _observations(
    setup: StudySetup,
    study: str,
    dates: pd.DatetimeIndex,
    final: np.ndarray,
    origins: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    parts = []
    for o in origins:
        t = np.arange(o + 1)
        share = _completeness(o - t, setup.mean_delay)
        reported = rng.binomial(final[:, : o + 1].astype(int), share)
        parts.append(_long_obs(study, setup, dates[t], dates[o], reported))
    last = dates[-1] + pd.Timedelta(days=setup.step_days)
    parts.append(_long_obs(study, setup, dates, last, final))
    return pd.concat(parts, ignore_index=True)


def _long_obs(
    study: str,
    setup: StudySetup,
    dates: pd.DatetimeIndex,
    as_of: pd.Timestamp,
    values: np.ndarray,
) -> pd.DataFrame:
    n_loc, n_t = values.shape
    locs, strata = zip(*setup.series, strict=True)
    return pd.DataFrame(
        {
            "study": study,
            "location": np.repeat(locs, n_t),
            "stratum": np.repeat(strata, n_t),
            "target_date": np.tile(dates, n_loc),
            "as_of_date": as_of,
            "value": values.ravel().astype(float),
        }
    )


def _forecasts(
    setup: StudySetup,
    study: str,
    dates: pd.DatetimeIndex,
    final: np.ndarray,
    origins: np.ndarray,
    rng: np.random.Generator,
    n_samples: int,
) -> pd.DataFrame:
    h = np.arange(setup.horizons[0], setup.horizons[1] + 1)
    # Error scale grows with the share not yet reported, and past the
    # origin with the steps ahead.
    unreported = 1 - _completeness(np.maximum(-h, 0), setup.mean_delay)
    difficulty = 0.06 + 0.5 * unreported + 0.12 * np.maximum(h, 0)
    z = norm.ppf(QUANTILE_LEVELS)
    n_loc = len(setup.series)
    o_grid, h_grid = np.meshgrid(origins, h, indexing="ij")
    target = o_grid + h_grid
    keep = target < setup.n_steps
    o_idx, h_idx = o_grid[keep], h_grid[keep]
    t_idx = target[keep]
    diff = difficulty[h_idx - h[0]]
    parts = []
    for model, skill in setup.models.items():
        loc_skill = np.exp(rng.normal(0, 0.15, n_loc))
        for li, (loc, stratum) in enumerate(setup.series):
            truth = final[li, t_idx]
            # Small counts are noisier on the log scale.
            sd = skill.error * loc_skill[li] * diff + 0.3 / np.sqrt(truth + 1)
            err = rng.normal(skill.bias, sd)
            median = (truth + 1) * np.exp(err) - 1
            sigma = skill.spread * sd
            if setup.form == "quantile":
                draws = z[None, :]
            else:
                draws = rng.standard_normal((len(truth), n_samples))
            q = (median[:, None] + 1) * np.exp(sigma[:, None] * draws) - 1
            q = np.maximum(q, 0)
            if setup.form == "sample":
                # Count samples, as a model of case counts gives.
                q = np.round(q)
            n, k = q.shape
            parts.append(
                pd.DataFrame(
                    {
                        "study": study,
                        "model": model,
                        "location": loc,
                        "stratum": stratum,
                        "origin_date": np.repeat(dates[o_idx], k),
                        "target_date": np.repeat(dates[t_idx], k),
                        "horizon": np.repeat(h_idx, k),
                        "quantile_level": (
                            np.tile(QUANTILE_LEVELS, n)
                            if setup.form == "quantile"
                            else np.nan
                        ),
                        "sample_id": (
                            pd.NA
                            if setup.form == "quantile"
                            else np.tile(np.arange(k), n)
                        ),
                        "value": q.ravel(),
                        "truth": np.repeat(truth, k),
                    }
                )
            )
    return pd.concat(parts, ignore_index=True)


def dummy_study(
    study: str, seed: int = 1, n_samples: int = 100
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Make dummy forecasts and observations for one study.

    Parameters
    ----------
    study : str
        A key of `STUDY_SETUP`.
    seed : int
        Random seed.
    n_samples : int
        Samples per forecast, for studies scored on samples.

    Returns
    -------
    tuple of pandas.DataFrame
        Forecasts and observations, both validated against the schema.
    """
    setup = STUDY_SETUP[study]
    rng = np.random.default_rng(seed)
    dates = pd.date_range(
        setup.start, periods=setup.n_steps, freq=f"{setup.step_days}D"
    )
    origins = setup.first_origin + setup.origin_every * np.arange(
        setup.n_origins
    )
    final = _final_counts(setup, rng)
    obs = _observations(setup, study, dates, final, origins, rng)
    fc = _forecasts(setup, study, dates, final, origins, rng, n_samples)
    return (
        validate_forecasts(fc.loc[:, list(FORECAST_COLUMNS)]),
        validate_observations(obs.loc[:, list(OBSERVATION_COLUMNS)]),
    )


def dummy_factors(forecasts: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    """Make dummy scenario factor values for each location.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Forecasts whose locations need factors.
    seed : int
        Random seed.

    Returns
    -------
    pandas.DataFrame
        `location` and raw values of `delay_mean`, `delay_q99`,
        `gt_mean`, `log_r0`, `n_places` and `report_days`, as the
        synthetic grid table gives.
    """
    rng = np.random.default_rng(seed)
    locs = sorted(forecasts["location"].unique())
    n = len(locs)
    delay_mean = np.round(np.exp(rng.uniform(0.5, 2.5, n)), 2)
    delay_sd = rng.uniform(0.3, 1.1, n)
    return pd.DataFrame(
        {
            "location": locs,
            "delay_mean": delay_mean,
            # 99% quantile of a lognormal with median delay_mean.
            "delay_q99": np.ceil(delay_mean * np.exp(2.326 * delay_sd)),
            "gt_mean": np.round(rng.uniform(6, 20, n), 1),
            "log_r0": np.round(rng.normal(0.2, 0.3, n), 2),
            "n_places": np.where(
                rng.uniform(size=n) < 0.4, 1, rng.integers(2, 27, n)
            ),
            "report_days": rng.choice([5, 7], n),
        }
    )


def dummy_split(forecasts: pd.DataFrame) -> pd.Timestamp:
    """Return the dummy split: the middle origin date.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Forecasts of one study.

    Returns
    -------
    pandas.Timestamp
        The first origin of the later half.
    """
    origins = sorted(forecasts["origin_date"].unique())
    return pd.Timestamp(origins[len(origins) // 2])
