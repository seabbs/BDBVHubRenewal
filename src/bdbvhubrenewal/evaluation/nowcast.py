# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Backtest of the random walk models on the real releases.

Every real release with a later release is an origin.
The models forecast from the releases up to the origin only
(`bdbvhubrenewal.forecasters.randomwalk`).
Two targets are scored.

**Next release.** The count the later release $o'$ shows for each hub
week up to four back from the origin's week.
`randomwalknowcast` predicts it by chaining the origin's days over the
actual gap to $o'$ only (`forecasters.chain.retro_plan`), with the
Skellam spread of the origin.
The prediction is scored only when the steps stand in for the gap within
`Settings.max_mismatch` days, as for the errors behind the spread.
A forecast dropped for one model is dropped for all.
`randomwalk` has no release to release chain, so its samples serve for
both targets.

**Complete weeks.** The count the latest release shows for the weeks it
reports in full, those with $\pi \ge 0.95$ under the latest release's
delay estimate.
Forecast weeks are scored only here.

A floor, the count as reported at the origin, is scored on the same
targets.
An origin that has no pair of releases within the delay window is listed
with no forecasts.
A model with no snapshots stops the run.

Scores are WIS, coverage and bias (`bdbvhubrenewal.evaluation.scoring`)
and every table counts its forecasts, $n$.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pandas as pd

from bdbvhubrenewal.baselinenowcast import chain_growth
from bdbvhubrenewal.data.dates import epiweek_end
from bdbvhubrenewal.data.snapshots import snapshot_arrays
from bdbvhubrenewal.evaluation.scoring import SCORE_COLUMNS, score_forecasts
from bdbvhubrenewal.forecasters.base import (
    NATIONAL,
    ForecastModel,
    ForecastTask,
    ModelUnavailableError,
    add_national,
    get_model,
)
from bdbvhubrenewal.forecasters.chain import (
    MissingSnapshots,
    Origin,
    Settings,
    build_origin,
    retro_plan,
)
from bdbvhubrenewal.forecasters.randomwalk import (
    RandomWalkNowcast,
    _frame,
)

__all__ = [
    "BASELINE",
    "MAX_WEEKS_BACK",
    "ReleaseBacktest",
    "complete_weeks",
    "release_backtest",
    "summarise",
]

log = logging.getLogger(__name__)

BASELINE = "as reported"
"""Name of the floor: the count reported at the origin, as a forecast."""

MAX_WEEKS_BACK = 3
"""Last week back from the origin's week scored on the next release."""

_LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
_KEYS = (
    "model",
    "origin",
    "target",
    "truth_vintage",
    "location",
    "target_end_date",
)


@dataclass(frozen=True)
class ReleaseBacktest:
    """The result of `release_backtest`.

    Attributes
    ----------
    scores : pandas.DataFrame
        One row per scored forecast: the keys, `weeks_back`, `truth`,
        `complete` and the scores.
    quantiles : pandas.DataFrame
        The 5, 25, 50, 75 and 95% quantiles of every forecast made for
        the weeks up to four back, with the count reported at the origin
        and the truth where there is one.
    origins : pandas.DataFrame
        One row per origin and model: `status`, `reason`, the days by
        which the chain steps differ from the gap to each later release
        and the number of forecasts scored on each target.
    fits : pandas.DataFrame
        Each model's fit summaries from `ForecastModel.fit_diagnostics`.
    """

    scores: pd.DataFrame
    quantiles: pd.DataFrame
    origins: pd.DataFrame
    fits: pd.DataFrame


def complete_weeks(
    daily: pd.DataFrame,
    vintage: str,
    locations: Sequence[str],
    settings: Settings | None = None,
) -> pd.DataFrame:
    r"""Say which weeks a release reports in full.

    Parameters
    ----------
    daily : pandas.DataFrame
        Daily onsets by release.
    vintage : str
        The release.
    locations : sequence of str
        Places to give, with the country.
    settings : Settings, optional
        Gives the delay window and the completeness level.

    Returns
    -------
    pandas.DataFrame
        Columns `location`, `target_end_date`, `share` ($\pi$) and
        `complete` ($\pi$ at least `Settings.completeness`).
    """
    cfg = settings or Settings()
    snap = snapshot_arrays(daily, vintage, tuple(locations))
    origin = build_origin(snap, cfg, errors=False)
    rows = pd.DataFrame(
        {
            "location": np.repeat(snap.locations, snap.n_weeks),
            "target_end_date": np.tile(snap.week_ends, len(snap.locations)),
            "share": origin.complete.reshape(-1),
        }
    )
    rows["complete"] = rows["share"] >= cfg.completeness
    keep = set(locations) | {NATIONAL}
    return rows.loc[rows["location"].isin(keep)].reset_index(drop=True)


def _week_sums(
    snap_weeks: np.ndarray, x: np.ndarray, n_weeks: int
) -> np.ndarray:
    onehot = np.zeros((snap_weeks.size, n_weeks))
    onehot[np.arange(snap_weeks.size), snap_weeks] = 1.0
    return np.asarray(x @ onehot)


def expected_point(
    origin: Origin, daily: pd.DataFrame, later: str, locations: Sequence[str]
) -> tuple[np.ndarray, int]:
    """Return the count expected at a later release, by hub week.

    Parameters
    ----------
    origin : Origin
        The fit at the origin.
    daily : pandas.DataFrame
        Daily onsets by release, including `later`.
    later : str
        The later release.
    locations : sequence of str
        The task's places.

    Returns
    -------
    tuple
        The mean weekly count the later release shows for the days up to
        the origin's cut-off, `[L, W]` over the origin's weeks, and the
        days by which the chain steps differ from the gap.

    Raises
    ------
    ValueError
        If the releases or places differ from the origin's.
    """
    big = snapshot_arrays(daily, later, tuple(locations))
    snap = origin.snap
    if (
        big.locations != snap.locations
        or big.releases[: len(snap.releases)] != snap.releases
    ):
        raise ValueError("Releases or places differ from the origin's.")
    p = big.releases.index(later)
    plan, mismatch = retro_plan(big, origin.o, p, origin.fit, origin.settings)
    raw = np.array(
        chain_growth(
            origin.fit.factors,
            (
                jnp.asarray(plan.step),
                jnp.asarray(plan.age),
                jnp.asarray(plan.valid),
            ),
        )
    )
    counts = big.counts[:, origin.o, :]
    point = _week_sums(big.week, counts * raw, big.n_weeks)
    return point[:, : snap.n_weeks], mismatch


def _weeks_back(vintage: str, weeks: pd.Series) -> pd.Series:
    here = pd.Timestamp(epiweek_end(vintage))
    return ((here - pd.to_datetime(weeks)).dt.days // 7).astype(int)


def _reported(
    real: pd.DataFrame, vintage: str, task: ForecastTask
) -> pd.DataFrame:
    """Return the floor's samples: the count reported at the origin."""
    rows = real.loc[real["vintage"] == vintage]
    have = rows.set_index(["location", "target_end_date"])["total"]
    locs = [*task.locations]
    out = []
    for loc in locs:
        value = [float(have.get((loc, w), 0.0)) for w in task.target_end_dates]
        out.append(
            pd.DataFrame(
                {
                    "location": loc,
                    "target_end_date": np.tile(
                        task.target_end_dates, task.n_samples
                    ),
                    "sample": np.repeat(
                        np.arange(1, task.n_samples + 1),
                        len(task.target_end_dates),
                    ),
                    "value": np.tile(value, task.n_samples),
                }
            )
        )
    return add_national(pd.concat(out, ignore_index=True))


def _label(
    samples: pd.DataFrame,
    model: str,
    origin: str,
    target: str,
    truth_vintage: str,
) -> pd.DataFrame:
    out = samples.copy()
    out["model"] = model
    out["origin"] = origin
    out["target"] = target
    out["truth_vintage"] = truth_vintage
    out["weeks_back"] = _weeks_back(origin, out["target_end_date"])
    return out


def _truth(real: pd.DataFrame, vintage: str) -> pd.DataFrame:
    rows = real.loc[real["vintage"] == vintage]
    return rows[["location", "target_end_date", "total"]].rename(
        columns={"total": "truth"}
    )


def release_backtest(
    tasks: Sequence[ForecastTask],
    daily: pd.DataFrame,
    target: pd.DataFrame,
    models: Sequence[str] = ("randomwalk", "randomwalknowcast"),
    settings: Settings | None = None,
) -> ReleaseBacktest:
    """Run the random walk models from every real release and score them.

    Parameters
    ----------
    tasks : sequence of ForecastTask
        One real task per release, with its release snapshots.
    daily : pandas.DataFrame
        Daily onsets by release, with every release.
    target : pandas.DataFrame
        The tidy target table.
    models : sequence of str
        Registered models. `randomwalknowcast` also gives the next
        release prediction from the chain.
    settings : Settings, optional
        The settings the models use.

    Returns
    -------
    ReleaseBacktest
        Scores, quantiles, the origin listing and the fit summaries.

    Raises
    ------
    MissingSnapshots
        If a task has no release snapshots.
    ValueError
        If two models give identical samples for a task.
    """
    cfg = settings or Settings()
    real = target.loc[target["kind"] == "real"]
    releases = sorted(str(v) for v in real["vintage"].unique())
    latest = releases[-1]
    built: dict[str, ForecastModel] = {m: get_model(m) for m in models}
    frames: list[pd.DataFrame] = []
    origin_rows: list[dict[str, object]] = []
    fit_rows: list[pd.DataFrame] = []
    for task in sorted(tasks, key=lambda t: t.vintage):
        if task.kind != "real":
            continue
        later = [v for v in releases if v > task.vintage]
        made: dict[str, np.ndarray] = {}
        for name, model in built.items():
            row: dict[str, object] = {
                "origin": task.vintage,
                "model": name,
                "status": "ran" if later else "no later release",
                "reason": "",
                "n_next": 0,
                "n_complete": 0,
            }
            if task.daily_history is None or task.daily_history.empty:
                raise MissingSnapshots(
                    f"{name} needs daily release snapshots, and the task "
                    f"for {task.vintage} has none."
                )
            try:
                samples = add_national(model.forecast(task))
            except MissingSnapshots:
                raise
            except ModelUnavailableError as err:
                log.warning("%s has no forecast from %s: %s", name,
                            task.vintage, err)  # fmt: skip
                row.update(status="no forecast", reason=str(err))
                origin_rows.append(row)
                continue
            values = samples["value"].to_numpy()
            for other, prior in made.items():
                if prior.shape == values.shape and np.array_equal(
                    prior, values
                ):
                    raise ValueError(
                        f"{name} and {other} gave identical samples from "
                        f"{task.vintage}."
                    )
            made[name] = values
            fits = pd.DataFrame(getattr(model, "fit_diagnostics", []))
            if not fits.empty:
                fits.insert(0, "model", name)
                fits.insert(1, "origin", task.vintage)
                fit_rows.append(fits)
            if task.vintage < latest:
                frames.append(
                    _label(samples, name, task.vintage, "complete", latest)
                )
            for o2 in later:
                if isinstance(model, RandomWalkNowcast):
                    origin = model._origin(task, errors=True)
                    point, miss = expected_point(
                        origin, daily, o2, task.locations
                    )
                    row[f"mismatch_{o2}"] = miss
                    if miss > cfg.max_mismatch:
                        continue
                    traj, weeks = model.trajectories(origin, task, point)
                    nxt = add_national(
                        _frame(
                            task,
                            weeks,
                            traj,
                            model._rows(origin, task),
                        )
                    )
                else:
                    nxt = samples
                frames.append(_label(nxt, name, task.vintage, "next", o2))
            origin_rows.append(row)
        if made and task.vintage < latest:
            floor = _reported(real, task.vintage, task)
            frames.append(
                _label(floor, BASELINE, task.vintage, "complete", latest)
            )
            for o2 in later:
                frames.append(_label(floor, BASELINE, task.vintage, "next", o2))
    return _score(
        frames, origin_rows, fit_rows, real, daily, latest, tasks, cfg
    )


def _score(
    frames: list[pd.DataFrame],
    origin_rows: list[dict[str, object]],
    fit_rows: list[pd.DataFrame],
    real: pd.DataFrame,
    daily: pd.DataFrame,
    latest: str,
    tasks: Sequence[ForecastTask],
    cfg: Settings,
) -> ReleaseBacktest:
    fits = (
        pd.concat(fit_rows, ignore_index=True) if fit_rows else pd.DataFrame()
    )
    if not frames:
        return ReleaseBacktest(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(origin_rows), fits
        )
    samples = pd.concat(frames, ignore_index=True)
    # A model with no forecast of a target (the chain has no step within
    # the allowed mismatch) drops it for every model, so all are scored
    # on the same forecasts.
    forecast_ids = ["target", "origin", "truth_vintage"]
    n_models = samples.groupby("target")["model"].transform("nunique")
    have = samples.groupby(forecast_ids)["model"].transform("nunique")
    samples = samples.loc[have == n_models].reset_index(drop=True)
    places = sorted({loc for t in tasks for loc in t.locations})
    done = complete_weeks(daily, latest, places, cfg)
    scored, quant = [], []
    for (target, truth_v), grp in samples.groupby(["target", "truth_vintage"]):
        truth = _truth(real, str(truth_v))
        grp = grp.assign(horizon=grp["weeks_back"])
        if target == "next":
            grp = grp.loc[grp["weeks_back"].between(0, MAX_WEEKS_BACK)]
        else:
            grp = grp.merge(
                done.loc[done["complete"], ["location", "target_end_date"]],
                on=["location", "target_end_date"],
            )
        if grp.empty:
            continue
        s = score_forecasts(grp, truth, keys=_KEYS)
        s["weeks_back"] = s.pop("horizon")
        scored.append(s)
    scores = pd.concat(scored, ignore_index=True) if scored else pd.DataFrame()
    reported = {
        v: _truth(real, v).rename(columns={"truth": "reported"})
        for v in sorted(set(samples["origin"]))
    }
    for key, grp in samples.loc[
        samples["weeks_back"].between(0, MAX_WEEKS_BACK)
    ].groupby([*_KEYS]):
        ids = dict(zip(_KEYS, key, strict=True))
        q = np.quantile(grp["value"].to_numpy(), _LEVELS)
        quant.append(
            {
                **ids,
                "weeks_back": int(grp["weeks_back"].iloc[0]),
                **{
                    f"q{int(100 * p)}": float(v)
                    for p, v in zip(_LEVELS, q, strict=True)
                },
            }
        )
    quantiles = pd.DataFrame(quant)
    if not quantiles.empty:
        quantiles = quantiles.merge(
            pd.concat([r.assign(origin=v) for v, r in reported.items()]),
            on=["origin", "location", "target_end_date"],
            how="left",
        )
        truths = pd.concat(
            [
                _truth(real, str(v)).assign(truth_vintage=str(v))
                for v in sorted(set(samples["truth_vintage"]))
            ]
        )
        quantiles = quantiles.merge(
            truths,
            on=["truth_vintage", "location", "target_end_date"],
            how="left",
        )
    origins = pd.DataFrame(origin_rows)
    if not scores.empty and not origins.empty:
        counts = (
            scores.groupby(["model", "origin", "target"])
            .size()
            .unstack("target", fill_value=0)
            .rename(columns={"next": "n_next", "complete": "n_complete"})
            .reset_index()
        )
        origins = origins.drop(columns=["n_next", "n_complete"]).merge(
            counts, on=["model", "origin"], how="left"
        )
        for c in ("n_next", "n_complete"):
            origins[c] = origins.get(c, 0)
            origins[c] = origins[c].fillna(0).astype(int)
    if not scores.empty:
        flags = done.rename(columns={"complete": "complete_at_latest"})
        scores = scores.merge(
            flags[["location", "target_end_date", "complete_at_latest"]],
            on=["location", "target_end_date"],
            how="left",
        )
    return ReleaseBacktest(scores, quantiles, origins, fits)


def summarise(
    scores: pd.DataFrame, by: Sequence[str] = ("target", "model")
) -> pd.DataFrame:
    """Average scores over forecasts, with the number of forecasts.

    Parameters
    ----------
    scores : pandas.DataFrame
        `ReleaseBacktest.scores`.
    by : sequence of str
        Columns to group by.

    Returns
    -------
    pandas.DataFrame
        The group columns, `n` and the mean WIS, coverage of the 50% and
        90% intervals and bias.
    """
    cols = ["wis", "coverage_50", "coverage_90", "bias"]
    assert set(cols) <= set(SCORE_COLUMNS)
    grp = scores.groupby(list(by))
    out = grp[cols].mean()
    out.insert(0, "n", grp.size())
    return out.reset_index()
