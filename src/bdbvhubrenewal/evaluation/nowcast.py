# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Backtest of the random walk models on the real releases.

Every real release with a later release is an origin.
The models forecast from the releases up to the origin only
(`bdbvhubrenewal.forecasters.randomwalk`).
Weeks back $k$ count from the week holding the origin's cut-off, so
forecast weeks have $k < 0$.
Three targets are scored.

**Next release.** The count the later release $o'$ shows for each hub
week up to `MAX_WEEKS_BACK` back from the origin's cut-off week.
`randomwalknowcast` predicts it by chaining the origin's days over the
actual gap to $o'$ only (`forecasters.chain.retro_plan`), with the
spread of the origin.
The prediction is scored only when the steps stand in for the gap within
`Settings.max_mismatch` days, as for the errors behind the spread.
A forecast dropped for one model is dropped for all.
Other models forecast the final count, not the count at the next
release, so they are not scored on this target, and the origin listing
says so.

**Complete weeks.** The count the latest release shows for the weeks it
reports in full, decided from the releases alone and never from a
model (`complete_weeks`).
By default a week is complete when it is at least `COMPLETE_WEEKS` weeks
old at the latest release and grew by less than `COMPLETE_GROWTH`
between the last two releases.

**Latest release.** The count the latest release shows for every week,
forecast weeks included. Recent weeks are still incomplete there, so
this target favours forecasts that are low, and it is the only one that
scores forecast weeks before they are complete.

Stored forecasts of models too slow to run here, such as the renewal
models, are scored on the complete and latest targets.

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

import numpy as np
import pandas as pd

from bdbvhubrenewal.data.dates import epiweek_end
from bdbvhubrenewal.data.snapshots import (
    release_cutoff,
    snapshot_arrays,
    week_sum,
)
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
    grow,
    retro_plan,
)
from bdbvhubrenewal.forecasters.randomwalk import (
    RandomWalk,
    RandomWalkNowcast,
    frame,
)

__all__ = [
    "BASELINE",
    "COMPLETE_GROWTH",
    "COMPLETE_WEEKS",
    "MAX_WEEKS_BACK",
    "NEAR_ZERO",
    "ReleaseBacktest",
    "clock_sensitivity",
    "complete_weeks",
    "pooled_summary",
    "relative_wis",
    "release_backtest",
    "summarise",
    "weeks_back_from_cutoff",
]

log = logging.getLogger(__name__)

BASELINE = "as reported"
"""Name of the floor: the count reported at the origin, as a forecast."""

MAX_WEEKS_BACK = 3
"""Last week back from the origin's week scored on the next release."""

MAX_WEEKS_AHEAD = 4
"""Last forecast week after the origin's week scored on the latest release."""

COMPLETE_WEEKS = 4
"""Weeks a week must be old at the latest release to count as complete."""

COMPLETE_GROWTH = 0.02
"""Most a complete week may grow between the last two releases."""

NEAR_ZERO = 5.0
"""Places whose weekly count never reaches this are near zero."""

NO_NEXT = "not scored: no next-release prediction"
"""Why a model that forecasts the final count has no next release score."""

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
    target: pd.DataFrame,
    daily: pd.DataFrame,
    vintage: str,
    rule: str = "growth",
    weeks: int = COMPLETE_WEEKS,
    growth: float = COMPLETE_GROWTH,
) -> pd.DataFrame:
    """Say which weeks a release reports in full, from the releases alone.

    Parameters
    ----------
    target : pandas.DataFrame
        The tidy target table, with every real release.
    daily : pandas.DataFrame
        Daily onsets by release, for the release's cut-off.
    vintage : str
        The release.
    rule : str
        `"growth"`: at least `weeks` old and grew by less than `growth`
        since the release before. `"age"`: at least `weeks` old.
    weeks : int
        Weeks back from the week holding the release's cut-off.
    growth : float
        Most a complete week may grow, as a share of its earlier count.
        A week at 0 in both releases has grown by 0.

    Returns
    -------
    pandas.DataFrame
        Columns `location`, `target_end_date`, `age` (weeks back),
        `growth` and `complete`.

    Raises
    ------
    ValueError
        If `rule` is not `"growth"` or `"age"`.
    """
    if rule not in ("growth", "age"):
        raise ValueError(f"Unknown rule {rule!r}: use 'growth' or 'age'.")
    real = target.loc[target["kind"] == "real"]
    releases = sorted(str(v) for v in real["vintage"].unique())
    before = [v for v in releases if v < vintage]
    rows = _truth(real, vintage).rename(columns={"truth": "now"})
    if before:
        prev = _truth(real, before[-1]).rename(columns={"truth": "then"})
        rows = rows.merge(prev, on=["location", "target_end_date"], how="left")
    else:
        rows["then"] = np.nan
    up = rows["now"] - rows["then"]
    rows["growth"] = np.where(
        rows["then"] > 0,
        up / rows["then"].where(rows["then"] > 0, 1.0),
        np.where(up > 0, np.inf, 0.0),
    )
    rows.loc[rows["then"].isna(), "growth"] = np.nan
    rows["age"] = weeks_back_from_cutoff(
        daily, vintage, rows["target_end_date"]
    )
    old = rows["age"] >= weeks
    if rule == "growth":
        old &= rows["growth"].fillna(np.inf) < growth
    rows["complete"] = old
    return rows[
        ["location", "target_end_date", "age", "growth", "complete"]
    ].reset_index(drop=True)


def weeks_back_from_cutoff(
    daily: pd.DataFrame, vintage: str, weeks: pd.Series
) -> pd.Series:
    """Return weeks back from the week holding a release's cut-off.

    Parameters
    ----------
    daily : pandas.DataFrame
        Daily onsets by release.
    vintage : str
        The release.
    weeks : pandas.Series
        Hub week ends.

    Returns
    -------
    pandas.Series
        Whole weeks, below 0 for weeks after the cut-off week.
    """
    cut = release_cutoff(daily.loc[daily["vintage"] == vintage], vintage)
    here = pd.Timestamp(epiweek_end(cut))
    back = (here - pd.to_datetime(weeks.reset_index(drop=True))).dt.days // 7
    return back.astype(int).set_axis(weeks.index)


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
    counts = big.counts[:, origin.o, :]
    growth = grow(origin.fit, plan)
    point = week_sum(big, counts * growth)
    return point[:, : snap.n_weeks], mismatch


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
    daily: pd.DataFrame,
) -> pd.DataFrame:
    out = samples.copy()
    out["model"] = model
    out["origin"] = origin
    out["target"] = target
    out["truth_vintage"] = truth_vintage
    out["weeks_back"] = weeks_back_from_cutoff(
        daily, origin, out["target_end_date"]
    )
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
    models: Sequence[str | ForecastModel] = ("randomwalk", "randomwalknowcast"),
    stored: pd.DataFrame | None = None,
    complete_rule: str = "growth",
    complete_after: int = COMPLETE_WEEKS,
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
    models : sequence of str or ForecastModel
        Registered model names, or models. `randomwalknowcast` also gives
        the next release prediction from the chain.
    stored : pandas.DataFrame, optional
        Samples of other models with columns `model`, `vintage`,
        `location`, `target_end_date`, `sample` and `value`. They are
        scored on the complete and latest targets.
    complete_rule, complete_after : str, int
        The `rule` and `weeks` of `complete_weeks`.

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
    real = target.loc[target["kind"] == "real"]
    releases = sorted(str(v) for v in real["vintage"].unique())
    latest = releases[-1]
    built: dict[str, ForecastModel] = {}
    for m in models:
        model = get_model(m) if isinstance(m, str) else m
        built[model.name] = model
    frames: list[pd.DataFrame] = []
    origin_rows: list[dict[str, object]] = []
    fit_rows: list[pd.DataFrame] = []

    def final(
        samples: pd.DataFrame, name: str, origin: str, kept: bool = False
    ) -> None:
        if origin < latest:
            for tgt in ("complete", "latest"):
                frames.append(
                    _label(samples, name, origin, tgt, latest, daily).assign(
                        stored=kept
                    )
                )

    for task in sorted(tasks, key=lambda t: t.vintage):
        if task.kind != "real":
            continue
        later = [v for v in releases if v > task.vintage]
        made: dict[str, np.ndarray] = {}
        for name, model in built.items():
            chained = isinstance(model, RandomWalkNowcast)
            row: dict[str, object] = {
                "origin": task.vintage,
                "model": name,
                "status": "ran" if later else "no later release",
                "reason": "",
                "next": "chain over the gap" if chained else NO_NEXT,
                "n_next": 0,
                "n_complete": 0,
                "n_latest": 0,
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
            final(samples, name, task.vintage)
            if chained:
                assert isinstance(model, RandomWalkNowcast)
                origin = model.origin_at(task, errors=True)
                for o2 in later:
                    nxt = _next_release(model, origin, task, daily, o2, row)
                    if nxt is not None:
                        frames.append(
                            _label(nxt, name, task.vintage, "next", o2, daily)
                        )
            origin_rows.append(row)
        if stored is not None and later:
            rows = stored.loc[stored["vintage"].astype(str) == task.vintage]
            for stored_name, grp in rows.groupby("model"):
                keep = grp[["location", "target_end_date", "sample", "value"]]
                final(keep, str(stored_name), task.vintage, kept=True)
                origin_rows.append(
                    {
                        "origin": task.vintage,
                        "model": str(stored_name),
                        "status": "stored",
                        "reason": "",
                        "next": NO_NEXT,
                    }
                )
        if made and task.vintage < latest:
            floor = _reported(real, task.vintage, task)
            final(floor, BASELINE, task.vintage)
            for o2 in later:
                frames.append(
                    _label(floor, BASELINE, task.vintage, "next", o2, daily)
                )
    done = complete_weeks(
        target, daily, latest, rule=complete_rule, weeks=complete_after
    )
    return _score(frames, origin_rows, fit_rows, real, done)


def _next_release(
    model: RandomWalkNowcast,
    origin: Origin,
    task: ForecastTask,
    daily: pd.DataFrame,
    later: str,
    row: dict[str, object],
) -> pd.DataFrame | None:
    """Return the prediction of the count at a later release, or None."""
    try:
        point, miss = expected_point(origin, daily, later, task.locations)
    except ModelUnavailableError as err:
        log.warning(
            "%s has no count expected at %s: %s", model.name, later, err
        )
        row[f"unavailable_{later}"] = str(err)
        return None
    row[f"mismatch_{later}"] = miss
    if miss > origin.settings.max_mismatch:
        return None
    traj, weeks = model.trajectories(origin, task, point)
    return add_national(frame(task, weeks, traj, model.rows(origin, task)))


def _score(
    frames: list[pd.DataFrame],
    origin_rows: list[dict[str, object]],
    fit_rows: list[pd.DataFrame],
    real: pd.DataFrame,
    done: pd.DataFrame,
) -> ReleaseBacktest:
    fits = (
        pd.concat(fit_rows, ignore_index=True) if fit_rows else pd.DataFrame()
    )
    if not frames:
        return ReleaseBacktest(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(origin_rows), fits
        )
    samples = pd.concat(frames, ignore_index=True)
    samples["stored"] = samples["stored"].fillna(False).astype(bool)
    # A model run here with no forecast of a target (the chain has no
    # step within the allowed mismatch) drops it for every model run here,
    # so all are scored on the same forecasts. Stored forecasts are scored
    # where they exist.
    forecast_ids = ["target", "origin", "truth_vintage"]
    run = samples.loc[~samples["stored"]]
    need = run.groupby("target")["model"].nunique().rename("need")
    have = (
        run.groupby(forecast_ids)["model"]
        .nunique()
        .rename("have")
        .reset_index()
        .merge(need, on="target")
    )
    full = have.loc[have["have"] == have["need"], forecast_ids]
    keep = (
        samples[forecast_ids]
        .merge(full.assign(ok=True), on=forecast_ids, how="left")["ok"]
        .fillna(False)
        .to_numpy(dtype=bool)
    )
    samples = samples.loc[keep | samples["stored"]].reset_index(drop=True)
    scored, quant = [], []
    for (target, truth_v), grp in samples.groupby(["target", "truth_vintage"]):
        truth = _truth(real, str(truth_v))
        grp = grp.assign(horizon=grp["weeks_back"])
        if target == "next":
            grp = grp.loc[grp["weeks_back"].between(0, MAX_WEEKS_BACK)]
        elif target == "latest":
            grp = grp.loc[grp["weeks_back"] >= -MAX_WEEKS_AHEAD]
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
    window = samples["weeks_back"].between(-MAX_WEEKS_AHEAD, MAX_WEEKS_BACK)
    for key, grp in samples.loc[window].groupby([*_KEYS]):
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
            .rename(columns=lambda t: f"n_{t}")
            .reset_index()
        )
        origins = origins.drop(
            columns=[c for c in origins if str(c).startswith("n_")]
        ).merge(counts, on=["model", "origin"], how="left")
        for c in ("n_next", "n_complete", "n_latest"):
            origins[c] = origins.get(c, 0)
            origins[c] = origins[c].fillna(0).astype(int)
    if not scores.empty:
        flags = done.rename(columns={"complete": "complete_at_latest"})
        scores = scores.merge(
            flags[["location", "target_end_date", "complete_at_latest"]],
            on=["location", "target_end_date"],
            how="left",
        )
        level = scores.groupby("location")["truth"].max().rename("level")
        scores = scores.merge(level, on="location", how="left")
        scores["near_zero"] = (scores["location"] != NATIONAL) & (
            scores["level"] < NEAR_ZERO
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


def pooled_summary(
    scores: pd.DataFrame,
    by: Sequence[str] = ("target", "model"),
    unit: str = "provinces",
) -> pd.DataFrame:
    """Summarise national and the pooled places, with and without near zero.

    National is the sum of the places, so it is kept apart from them
    rather than pooled with them.

    Parameters
    ----------
    scores : pandas.DataFrame
        `ReleaseBacktest.scores`, with `near_zero`.
    by : sequence of str
        Columns to group by within each pool.
    unit : str
        Name of the places in the pool labels.

    Returns
    -------
    pandas.DataFrame
        `summarise` output with a `pool` column: `national`, `unit` and
        `unit` without near-zero places (`NEAR_ZERO`).
    """
    national = scores["location"] == NATIONAL
    pools = {
        "national": scores.loc[national],
        unit: scores.loc[~national],
        f"{unit} without near-zero": scores.loc[
            ~national & ~scores["near_zero"]
        ],
    }
    out = [
        summarise(rows, by).assign(pool=name)
        for name, rows in pools.items()
        if not rows.empty
    ]
    table = pd.concat(out, ignore_index=True)
    return table[["pool", *[c for c in table if c != "pool"]]]


def relative_wis(
    scores: pd.DataFrame,
    by: Sequence[str] = ("target",),
    baseline: str = "randomwalk",
) -> pd.DataFrame:
    """Divide each model's mean WIS by the baseline's on the same forecasts.

    Parameters
    ----------
    scores : pandas.DataFrame
        `ReleaseBacktest.scores`.
    by : sequence of str
        Columns of a cell other than `model`.
    baseline : str
        The model to divide by. A cell without it uses the floor,
        `BASELINE`, and says so in `rel_to`.

    Returns
    -------
    pandas.DataFrame
        One row per cell and model: `n` matched forecasts, `wis`,
        `base_wis`, `rel_wis` and `rel_to`.
    """
    match = ["origin", "truth_vintage", "location", "target_end_date"]
    keys = list(by)
    out = []
    for cell, grp in (
        scores.groupby(keys, sort=False) if keys else [((), scores)]
    ):
        ids = dict(
            zip(keys, cell if isinstance(cell, tuple) else (cell,), strict=True)
        )
        ref = baseline if baseline in set(grp["model"]) else BASELINE
        base = grp.loc[grp["model"] == ref, [*match, "wis"]]
        for name, rows in grp.groupby("model", sort=False):
            pair = rows[[*match, "wis"]].merge(
                base, on=match, suffixes=("", "_base")
            )
            mean_base = float(pair["wis_base"].mean()) if len(pair) else np.nan
            out.append(
                {
                    **ids,
                    "model": name,
                    "n": len(pair),
                    "wis": float(pair["wis"].mean()) if len(pair) else np.nan,
                    "base_wis": mean_base,
                    "rel_wis": float(pair["wis"].mean()) / mean_base
                    if mean_base > 0
                    else np.nan,
                    "rel_to": ref,
                }
            )
    return pd.DataFrame(out)


def clock_sensitivity(
    tasks: Sequence[ForecastTask],
    daily: pd.DataFrame,
    target: pd.DataFrame,
    shifts: Sequence[int] = (-1, 0, 1),
    weeks: int = 1,
) -> pd.DataFrame:
    """Rerun both random walk models with the origin's clock moved.

    Each shift adds days to the ages of the origin's days
    (`Settings.clock_shift`), as if the release were extracted that many
    days later after its last onset.

    Parameters
    ----------
    tasks : sequence of ForecastTask
        Real tasks with their release snapshots.
    daily : pandas.DataFrame
        Daily onsets by release.
    target : pandas.DataFrame
        The tidy target table, for the count reported and later counts.
    shifts : sequence of int
        Days to move the clock by.
    weeks : int
        Weeks back from the cut-off week to give, from 0.

    Returns
    -------
    pandas.DataFrame
        One row per model, origin, shift, place and week: the 5, 25, 50,
        75 and 95% quantiles, the count reported, the count at the next
        release and at the latest one, where they exist.
    """
    real = target.loc[target["kind"] == "real"]
    releases = sorted(str(v) for v in real["vintage"].unique())
    rows = []
    for task in tasks:
        later = [v for v in releases if v > task.vintage]
        for shift in shifts:
            cfg = Settings(clock_shift=shift)
            for model in (
                RandomWalkNowcast(settings=cfg),
                RandomWalk(settings=cfg),
            ):
                try:
                    samples = add_national(model.forecast(task))
                except ModelUnavailableError:
                    continue
                samples["weeks_back"] = weeks_back_from_cutoff(
                    daily, task.vintage, samples["target_end_date"]
                )
                keep = samples.loc[samples["weeks_back"].between(0, weeks - 1)]
                q = (
                    keep.groupby(["location", "target_end_date", "weeks_back"])[
                        "value"
                    ]
                    .quantile(np.array(_LEVELS))
                    .unstack()
                )
                q.columns = [f"q{int(100 * p)}" for p in q.columns]
                q = q.reset_index().assign(
                    model=model.name, origin=task.vintage, clock_shift=shift
                )
                rows.append(q)
    out = pd.concat(rows, ignore_index=True)
    out = out.merge(
        pd.concat(
            [
                _truth(real, v).assign(origin=v)
                for v in sorted(set(out["origin"]))
            ]
        ).rename(columns={"truth": "reported"}),
        on=["origin", "location", "target_end_date"],
        how="left",
    )
    for name, pick in (("next", 0), ("latest", -1)):
        truths = []
        for v in sorted(set(out["origin"])):
            later = [r for r in releases if r > v]
            if later:
                truths.append(_truth(real, later[pick]).assign(origin=v))
        if truths:
            out = out.merge(
                pd.concat(truths).rename(columns={"truth": f"truth_{name}"}),
                on=["origin", "location", "target_end_date"],
                how="left",
            )
    first = ["model", "origin", "clock_shift", "location", "target_end_date"]
    return out[[*first, *[c for c in out if c not in first]]]
