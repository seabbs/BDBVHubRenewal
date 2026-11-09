# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Backtest of every model on the real releases.

Every real release with a later release is an origin.
The random walk models forecast from the releases up to the origin only
(`bdbvhubrenewal.forecasters.randomwalk`).
Stored forecasts of other models, such as the renewal models and hub
teams, are scored with them.
Weeks back $k$ count from the week holding the origin's cut-off, so
forecast weeks have $k < 0$.
Two targets are scored.

**Reported.** The count each later release $o'$ shows, for every week
of the forecast that $o'$ reports.
`randomwalk` is scored on its samples as they are, with no delay.
Every other model forecasts the final count $X$ of a week and is mapped
to what $o'$ shows through its own modelled delay, the share
$\pi$ of a day's final count reported by each age (`delay_curves`,
`week_shares`).
Of the count $X - R_o$ outstanding at the origin, with $R_o$ the count
reported there, $o'$ reports a draw from
$\text{Binomial}(X - R_o, q)$, with
$q = (\pi_{o'} - \pi_o) / (1 - \pi_o)$ (`draw_reported`).
A model with no modelled delay, such as a hub team, is scored on
complete weeks only.

**Complete weeks.** The count the latest release shows for the weeks it
reports in full, decided from the releases alone and never from a
model (`complete_weeks`).
By default a week is complete when it is at least `COMPLETE_WEEKS` weeks
old at the latest release and grew by less than `COMPLETE_GROWTH`
between the last two releases.

A model that cannot forecast from an origin, such as a nowcast with no
pair of releases within the delay window, is listed with no forecasts
there.
A model with no snapshots stops the run.

Scores are WIS, coverage and bias (`bdbvhubrenewal.evaluation.scoring`)
and every table counts its forecasts, $n$.
The energy score is scored from the same drawn samples, one per origin,
release, target and scale.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bdbvhubrenewal.data.dates import as_date, epiweek_end
from bdbvhubrenewal.data.snapshots import NATIONAL, release_cutoff
from bdbvhubrenewal.data.zones import province_of
from bdbvhubrenewal.evaluation.scoring import (
    SCORE_COLUMNS,
    energy_scores,
    relative_skill,
    score_forecasts,
)
from bdbvhubrenewal.forecasters.base import (
    ForecastModel,
    ForecastTask,
    ModelUnavailableError,
    add_national,
    get_model,
    list_models,
)
from bdbvhubrenewal.forecasters.chain import MissingSnapshots
from bdbvhubrenewal.forecasters.renewal import DELAY_SUMMARY_FILE

__all__ = [
    "COMPLETE_GROWTH",
    "COMPLETE_WEEKS",
    "RANDOM_WALK",
    "ReleaseBacktest",
    "complete_weeks",
    "delay_curves",
    "draw_reported",
    "relative_energy",
    "relative_wis",
    "release_backtest",
    "reported_samples",
    "stored_delays",
    "summarise",
    "week_shares",
    "weeks_back_from_cutoff",
]

log = logging.getLogger(__name__)

RANDOM_WALK = "randomwalk"
"""The model scored on its samples as they are, with no delay."""

COMPLETE_WEEKS = 4
"""Weeks a week must be old at the latest release to count as complete."""

COMPLETE_GROWTH = 0.02
"""Most a complete week may grow between the last two releases."""

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
        `complete_at_latest` and the scores.
    quantiles : pandas.DataFrame
        The 5, 25, 50, 75 and 95% quantiles of every forecast, with the
        count reported at the origin and the truth where there is one.
    energy : pandas.DataFrame
        `energy_scores` by `origin`, `target`, `truth_vintage` and
        `scale`, on the cells every model of the origin forecasts.
    origins : pandas.DataFrame
        One row per origin and model: `status`, `reason` and the number
        of forecasts scored on each target.
    fits : pandas.DataFrame
        Each model's fit summaries from `ForecastModel.fit_diagnostics`.
    """

    scores: pd.DataFrame
    quantiles: pd.DataFrame
    energy: pd.DataFrame
    origins: pd.DataFrame
    fits: pd.DataFrame


def delay_curves(summary: pd.DataFrame, places: Sequence[str]) -> pd.DataFrame:
    """Return a model's share of a day's final count reported by age.

    A delay fitted once for every place (location `"pooled"`) is first
    given to each place. Each place then takes the model's delay at that
    place, and a zone with none its province's.

    Parameters
    ----------
    summary : pandas.DataFrame
        The model's delay summary (`delay_summary`), with `location`,
        `x` (age in days), `quantity` and `q50`. The rows with quantity
        `reported_share` are used.
    places : sequence of str
        The places to map.

    Returns
    -------
    pandas.DataFrame
        Columns `location`, `age` and `share`, for the places the model
        has a delay for.
    """
    rows = summary.loc[summary["quantity"] == "reported_share"]
    rows = rows.rename(columns={"x": "age", "q50": "share"})
    pooled = rows.loc[rows["location"] == "pooled"]
    if not pooled.empty:
        rows = pd.concat([pooled.assign(location=p) for p in places])
    by_place = {str(loc): grp for loc, grp in rows.groupby("location")}
    frames = [
        by_place[key].assign(location=place)
        for place in places
        if (key := place if place in by_place else province_of(place))
        in by_place
    ]
    out = pd.concat(frames, ignore_index=True) if frames else rows.iloc[:0]
    return out.astype({"age": int})[["location", "age", "share"]]


def week_shares(
    curves: pd.DataFrame, cutoff: dt.date, weeks: Sequence[str]
) -> pd.DataFrame:
    r"""Return the share of each week's final count reported by a cut-off.

    Day $t$ is $a = c - t$ days old at cut-off $c$, with share $\pi(a)$
    from the place's curve, its last value past the curve's ages, and
    $\pi = 0$ for days after the cut-off.
    A week's share is the mean over its seven days.

    Parameters
    ----------
    curves : pandas.DataFrame
        `delay_curves` output.
    cutoff : datetime.date
        The cut-off of the release.
    weeks : sequence of str
        Hub week ends.

    Returns
    -------
    pandas.DataFrame
        Columns `location`, `target_end_date` and `share`, for every
        place in `curves`.
    """
    ends = [as_date(w) for w in weeks]
    age = np.array([[(cutoff - e).days + d for d in range(7)] for e in ends])
    frames = []
    for loc, grp in curves.groupby("location"):
        c = grp.sort_values("age")
        pi = np.where(
            age >= 0,
            np.interp(age, c["age"].to_numpy(), c["share"].to_numpy()),
            0.0,
        )
        frames.append(
            pd.DataFrame(
                {
                    "location": loc,
                    "target_end_date": list(weeks),
                    "share": pi.mean(axis=1),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def draw_reported(
    samples: pd.DataFrame,
    reported: pd.DataFrame,
    now: pd.DataFrame,
    then: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    r"""Draw the count a later release reports from samples of the final count.

    Of the count $X - R_o$ outstanding at the origin, with $R_o$ the count
    reported there, the later release $o'$ reports a binomial draw with
    probability $q = (\pi_{o'} - \pi_o) / (1 - \pi_o)$, the share it is
    expected to report. Each sample becomes
    $R_o + \text{Binomial}(X - R_o, q)$, with the sign carried when
    $X < R_o$. A week fully reported at the origin has $q = 1$.

    Parameters
    ----------
    samples : pandas.DataFrame
        Columns `location`, `target_end_date`, `sample` and `value`.
    reported : pandas.DataFrame
        Columns `location`, `target_end_date` and `reported`, the counts
        at the origin. Weeks not in it have 0 reported.
    now, then : pandas.DataFrame
        `week_shares` at the origin's cut-off and the later one's.
    rng : numpy.random.Generator
        Random numbers.

    Returns
    -------
    pandas.DataFrame
        The samples of the places and weeks in `now`, drawn.
    """
    keys = ["location", "target_end_date"]
    share = now.merge(then, on=keys, suffixes=("_o", "_l"))
    rows = samples.merge(share, on=keys).merge(reported, on=keys, how="left")
    pi_o, pi_l = rows.pop("share_o"), rows.pop("share_l")
    q = np.divide(
        pi_l - pi_o, 1.0 - pi_o, out=np.ones(len(rows)), where=pi_o < 1.0
    ).clip(0.0, 1.0)
    r = rows.pop("reported").fillna(0.0)
    left = np.round(rows["value"] - r).astype(np.int64)
    rows["value"] = r + np.sign(left) * rng.binomial(np.abs(left), q)
    return rows


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
    here = pd.Timestamp(epiweek_end(_cutoff(daily, vintage)))
    back = (here - pd.to_datetime(weeks.reset_index(drop=True))).dt.days // 7
    return back.astype(int).set_axis(weeks.index)


def reported_samples(
    samples: pd.DataFrame,
    summary: pd.DataFrame | None,
    model: str,
    origin: str,
    later: Sequence[str],
    daily: pd.DataFrame,
    reported: pd.DataFrame,
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    """Return a model's samples of what each later release shows.

    `randomwalk`'s samples are returned as they are, with no delay.
    Another model's places are mapped by its delay (`delay_curves`,
    `draw_reported`), a place it has no delay for is returned as it is,
    and the country is the sum of the provinces.
    A model with no delay summary, such as a hub team, has none.

    Parameters
    ----------
    samples : pandas.DataFrame
        The model's final-count samples, with `location`,
        `target_end_date`, `sample` and `value`.
    summary : pandas.DataFrame or None
        The model's delay summary at the origin.
    model : str
        The model's name.
    origin : str
        The origin release.
    later : sequence of str
        The later releases.
    daily : pandas.DataFrame
        Daily onsets by release, for the cut-offs.
    reported : pandas.DataFrame
        Columns `location`, `target_end_date` and `reported`, the counts
        at the origin.
    rng : numpy.random.Generator
        Random numbers.

    Returns
    -------
    dict of str to pandas.DataFrame
        Samples by later release.
    """
    if model == RANDOM_WALK:
        return dict.fromkeys(later, samples)
    if summary is None:
        return {}
    places = samples.loc[samples["location"] != NATIONAL]
    curves = delay_curves(summary, sorted(set(places["location"])))
    mapped = places["location"].isin(set(curves["location"]))
    weeks = sorted(set(samples["target_end_date"]))
    now = week_shares(curves, _cutoff(daily, origin), weeks)
    out = {}
    for o2 in later:
        then = week_shares(curves, _cutoff(daily, o2), weeks)
        drawn = draw_reported(places.loc[mapped], reported, now, then, rng)
        both = pd.concat([drawn, places.loc[~mapped]], ignore_index=True)
        whole = NATIONAL in set(samples["location"])
        out[o2] = add_national(both) if whole else both
    return out


def stored_delays(results: Path | str) -> pd.DataFrame:
    """Return the fitted models' stored delay summaries.

    `fit-store load` writes them to `results`.

    Parameters
    ----------
    results : pathlib.Path or str
        The results directory.

    Returns
    -------
    pandas.DataFrame
        The delay summaries of the fitted models, with `model` and
        `vintage`, and no rows if no fits were loaded.
    """
    path = Path(results) / DELAY_SUMMARY_FILE
    if not path.exists():
        return pd.DataFrame(columns=["model", "vintage"])
    learnt = pd.read_csv(path)
    fitted = [s.name for s in list_models() if s.fitted]
    return learnt.loc[learnt["model"].isin(fitted)]


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


def _cutoff(daily: pd.DataFrame, vintage: str) -> dt.date:
    return release_cutoff(daily.loc[daily["vintage"] == vintage], vintage)


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
    delays: pd.DataFrame | None = None,
    complete_rule: str = "growth",
    complete_after: int = COMPLETE_WEEKS,
    compare: Collection[str] | None = None,
) -> ReleaseBacktest:
    """Forecast from every real release and score every model.

    Parameters
    ----------
    tasks : sequence of ForecastTask
        One real task per release, with its release snapshots.
    daily : pandas.DataFrame
        Daily onsets by release, with every release.
    target : pandas.DataFrame
        The tidy target table.
    models : sequence of str or ForecastModel
        Registered model names, or models, to run.
    stored : pandas.DataFrame, optional
        Final-count samples of other models with columns `model`,
        `vintage`, `location`, `target_end_date`, `sample` and `value`.
    delays : pandas.DataFrame, optional
        The stored models' delay summaries, with `model` and `vintage`
        (`delay_summary`). A stored model with none is scored on complete
        weeks only.
    complete_rule, complete_after : str, int
        The `rule` and `weeks` of `complete_weeks`.
    compare : collection of str, optional
        Models the energy score compares, on the cells they all forecast.
        All models if None. Other models get no energy score.

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

    Examples
    --------
    Score stored forecasts against the counts later releases reported.

    >>> from bdbvhubrenewal.evaluation.nowcast import (
    ...     relative_wis, release_backtest
    ... )
    >>> result = release_backtest(
    ...     [task], daily, target, models=(), stored=forecasts
    ... )
    >>> scores = result.scores.query("target == 'reported'")
    >>> relative_wis(scores)
    """
    real = target.loc[target["kind"] == "real"]
    releases = sorted(str(v) for v in real["vintage"].unique())
    latest = releases[-1]
    built = [get_model(m) if isinstance(m, str) else m for m in models]
    cols = ["location", "target_end_date", "sample", "value"]
    kept = pd.DataFrame(columns=["model", "vintage", *cols])
    kept = kept if stored is None else stored
    frames: list[pd.DataFrame] = []
    origin_rows: list[dict[str, object]] = []
    fit_rows: list[pd.DataFrame] = []
    for task in sorted(tasks, key=lambda t: t.vintage):
        later = [v for v in releases if v > task.vintage]
        if task.kind != "real" or not later:
            continue
        if task.daily_history is None or task.daily_history.empty:
            raise MissingSnapshots(
                f"Release backtest needs daily release snapshots, and the "
                f"task for {task.vintage} has none."
            )
        here = kept.loc[kept["vintage"].astype(str) == task.vintage]
        forecasts: dict[str, pd.DataFrame] = {}
        summaries: dict[str, pd.DataFrame | None] = {}
        for model in built:
            try:
                samples = model.forecast(task)
            except ModelUnavailableError as err:
                log.warning("%s has no forecast at %s: %s", model.name,
                            task.vintage, err)  # fmt: skip
                origin_rows.append(
                    {
                        "origin": task.vintage,
                        "model": model.name,
                        "status": "no forecast",
                        "reason": str(err),
                    }
                )
                continue
            forecasts[model.name] = add_national(samples)
            summaries[model.name] = getattr(model, "delay_summary", None)
            fits = pd.DataFrame(getattr(model, "fit_diagnostics", []))
            fits.insert(0, "model", model.name)
            fits.insert(1, "origin", task.vintage)
            fit_rows.append(fits)
        _check_distinct(forecasts, task.vintage)
        learnt = (
            pd.DataFrame(columns=["model", "vintage"])
            if delays is None
            else delays.loc[delays["vintage"].astype(str) == task.vintage]
        )
        for name, grp in here.groupby("model"):
            forecasts[str(name)] = grp[cols]
            mine = learnt.loc[learnt["model"] == name]
            summaries[str(name)] = None if mine.empty else mine
        reported = _truth(real, task.vintage).rename(
            columns={"truth": "reported"}
        )
        # Each model draws from its own stream, so its scores do not depend
        # on which other models run.
        rngs = {name: task.rng(name) for name in forecasts}
        for name, samples in forecasts.items():
            frames += [
                _label(drawn, name, task.vintage, "reported", o2, daily)
                for o2, drawn in reported_samples(
                    samples, summaries[name], name, task.vintage, later,
                    daily, reported, rngs[name],
                ).items()
            ]  # fmt: skip
            frames.append(
                _label(samples, name, task.vintage, "complete", latest, daily)
            )
        origin_rows += [
            {
                "origin": task.vintage,
                "model": name,
                "status": "scored",
                "reason": "",
            }
            for name in forecasts
        ]
    done = complete_weeks(
        target, daily, latest, rule=complete_rule, weeks=complete_after
    )
    return _score(frames, origin_rows, fit_rows, real, done, compare)


def _check_distinct(forecasts: dict[str, pd.DataFrame], vintage: str) -> None:
    """Stop if two models give identical samples."""
    seen: dict[str, np.ndarray] = {}
    for name, samples in forecasts.items():
        values = samples["value"].to_numpy()
        for other, prior in seen.items():
            if np.array_equal(prior, values):
                raise ValueError(
                    f"{name} and {other} gave identical samples from {vintage}."
                )
        seen[name] = values


def _score(
    frames: list[pd.DataFrame],
    origin_rows: list[dict[str, object]],
    fit_rows: list[pd.DataFrame],
    real: pd.DataFrame,
    done: pd.DataFrame,
    compare: Collection[str] | None,
) -> ReleaseBacktest:
    fits = (
        pd.concat(fit_rows, ignore_index=True) if fit_rows else pd.DataFrame()
    )
    origins = pd.DataFrame(
        origin_rows, columns=["origin", "model", "status", "reason"]
    )
    if not frames:
        empty = pd.DataFrame()
        return ReleaseBacktest(empty, empty, empty, origins, fits)
    samples = pd.concat(frames, ignore_index=True)
    complete = done.loc[done["complete"], ["location", "target_end_date"]]
    scored, energy = [], []
    for (tgt, truth_v), grp in samples.groupby(["target", "truth_vintage"]):
        if tgt == "complete":
            grp = grp.merge(complete, on=["location", "target_end_date"])
        truth = _truth(real, str(truth_v))
        s = score_forecasts(
            grp.assign(horizon=grp["weeks_back"]), truth, keys=_KEYS
        )
        s["weeks_back"] = s.pop("horizon")
        scored.append(s)
        pool = grp if compare is None else grp[grp["model"].isin(compare)]
        energy.append(energy_scores(pool, truth, unit=_KEYS[1:4]))
    scores = pd.concat(scored, ignore_index=True)
    flags = done.rename(columns={"complete": "complete_at_latest"})
    scores = scores.merge(
        flags[["location", "target_end_date", "complete_at_latest"]],
        on=["location", "target_end_date"],
        how="left",
    )
    counts = (
        scores.groupby(["model", "origin", "target"])
        .size()
        .unstack("target", fill_value=0)
        .reindex(columns=["reported", "complete"], fill_value=0)
        .add_prefix("n_")
        .reset_index()
    )
    origins = origins.merge(counts, on=["model", "origin"], how="left")
    origins[["n_reported", "n_complete"]] = (
        origins[["n_reported", "n_complete"]].fillna(0).astype(int)
    )
    return ReleaseBacktest(
        scores,
        _quantiles(samples, real),
        pd.concat(energy, ignore_index=True),
        origins,
        fits,
    )


def _quantiles(samples: pd.DataFrame, real: pd.DataFrame) -> pd.DataFrame:
    """Quantiles of every forecast, with the count at the origin and truth."""
    grp = samples.groupby([*_KEYS, "weeks_back"])["value"]
    q = grp.quantile(np.array(_LEVELS)).unstack()
    q.columns = [f"q{int(100 * p)}" for p in _LEVELS]
    q = q.reset_index()
    at = {
        role: pd.concat(
            [
                _truth(real, str(v)).assign(**{col: str(v)})
                for v in sorted(set(q[col]))
            ]
        ).rename(columns={"truth": role})
        for role, col in (("reported", "origin"), ("truth", "truth_vintage"))
    }
    q = q.merge(
        at["reported"], on=["origin", "location", "target_end_date"], how="left"
    )
    return q.merge(
        at["truth"],
        on=["truth_vintage", "location", "target_end_date"],
        how="left",
    )


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
        The model to divide by.

    Returns
    -------
    pandas.DataFrame
        One row per cell and model, over the forecasts the baseline also
        made: `n`, `wis`, `base_wis`, `rel_wis`, `rel_log_wis` (mean WIS
        on the log scale over the baseline's), missing where the baseline's
        mean is 0, the coverage of the 50% and 90% intervals and the mean
        `bias`.
    """
    match = ("target", "origin", "truth_vintage", "location", "target_end_date")
    rel = relative_skill(
        scores,
        baseline,
        by=by,
        match_on=match,
        means=("wis", "log_wis", "coverage_50", "coverage_90", "bias"),
    )
    out = rel.rename(
        columns={"relative_wis": "rel_wis", "relative_log_wis": "rel_log_wis"}
    ).sort_values([*by, "model"])
    order = [
        "n",
        "wis",
        "base_wis",
        "rel_wis",
        "rel_log_wis",
        "coverage_50",
        "coverage_90",
        "bias",
    ]
    return out[[*by, "model", *order]].reset_index(drop=True)


def relative_energy(
    energy: pd.DataFrame,
    by: Sequence[str] = ("target", "scale"),
    baseline: str = "randomwalk",
) -> pd.DataFrame:
    """Divide each model's mean energy score by the baseline's.

    Parameters
    ----------
    energy : pandas.DataFrame
        `ReleaseBacktest.energy`.
    by : sequence of str
        Columns of a cell other than `model`. Origins and releases are
        matched within it.
    baseline : str
        The model to divide by.

    Returns
    -------
    pandas.DataFrame
        One row per cell and model, over the origins and releases the
        baseline also has: `n_energy`, `rel_energy` and `rel_log_energy`.
    """
    rel = relative_skill(
        energy,
        baseline,
        by=by,
        metrics=("energy", "log_energy"),
        match_on=("origin", "truth_vintage"),
    )
    return rel.rename(
        columns={
            "n": "n_energy",
            "relative_energy": "rel_energy",
            "relative_log_energy": "rel_log_energy",
        }
    )
