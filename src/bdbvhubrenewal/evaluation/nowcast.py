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
Every model forecasts the final count $X$ of a week.
Of the count $X - R_o$ outstanding at the origin, with $R_o$ the count
reported there, $o'$ reports a draw from
$\text{Binomial}(X - R_o, q)$, with
$q = (\pi_{o'} - \pi_o) / (1 - \pi_o)$, where $\pi$ is the share of the
week's final count reported by a cut-off under the delay fitted at the
origin (`reported_share`, `draw_reported`).
Its mean is the chain over the gap to $o'$ for the chain nowcast, and
the floor maps to itself.

**Complete weeks.** The count the latest release shows for the weeks it
reports in full, decided from the releases alone and never from a
model (`complete_weeks`).
By default a week is complete when it is at least `COMPLETE_WEEKS` weeks
old at the latest release and grew by less than `COMPLETE_GROWTH`
between the last two releases.

A floor, the count as reported at the origin, is scored on both targets.
An origin that has no pair of releases within the delay window has no
delay fit, so it is listed with no forecasts.
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

import numpy as np
import pandas as pd

from bdbvhubrenewal.baselinenowcast import growth
from bdbvhubrenewal.data.dates import as_date, epiweek_end
from bdbvhubrenewal.data.snapshots import release_cutoff
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
)
from bdbvhubrenewal.forecasters.chain import (
    MissingSnapshots,
    Origin,
)
from bdbvhubrenewal.forecasters.randomwalk import (
    RandomWalk,
)

__all__ = [
    "BASELINE",
    "COMPLETE_GROWTH",
    "COMPLETE_WEEKS",
    "ReleaseBacktest",
    "complete_weeks",
    "draw_reported",
    "relative_energy",
    "relative_wis",
    "release_backtest",
    "reported_share",
    "summarise",
    "weeks_back_from_cutoff",
]

log = logging.getLogger(__name__)

BASELINE = "as reported"
"""Name of the floor: the count reported at the origin, as a forecast."""

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


def reported_share(
    origin: Origin, cutoff: dt.date, weeks: Sequence[str]
) -> pd.DataFrame:
    r"""Return the share of each week's final count reported by a cut-off.

    Day $t$ is $a = c - t$ days old at cut-off $c$.
    The share of its final count reported then is $\pi(a) = 1 / G(a)$,
    with $G(a) = \exp \sum_{j=a}^{D-1} g(j)$ from the origin's daily
    delay, as in the nowcast (`bdbvhubrenewal.forecasters.chain`), and
    $\pi = 0$ for days after the cut-off.
    A week's share is the mean over its seven days,
    $\pi_w = \frac{1}{7} \sum_{t \in w} \pi(c - t)$.

    Parameters
    ----------
    origin : Origin
        The fit at the origin, which gives the delay.
    cutoff : datetime.date
        The cut-off of the later release.
    weeks : sequence of str
        Hub week ends.

    Returns
    -------
    pandas.DataFrame
        Columns `location`, `target_end_date` and `share`, for every place
        of the origin.
    """
    ends = [as_date(w) for w in weeks]
    age = np.array([[(cutoff - e).days + d for d in range(7)] for e in ends])
    g = growth(origin.fit.g, age, origin.settings.max_delay)
    pi = np.where(age >= 0, 1.0 / g, 0.0)
    places = origin.snap.locations
    return pd.DataFrame(
        {
            "location": np.repeat(places, len(ends)),
            "target_end_date": np.tile(list(weeks), len(places)),
            "share": pi.mean(axis=2).ravel(),
        }
    )


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
        `reported_share` at the origin's cut-off and the later one's.
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
        names = [m.name for m in built] + sorted(set(here["model"]))
        try:
            origin = RandomWalk().origin_at(task, errors=False)
        except ModelUnavailableError as err:
            log.warning("No delay fit at %s: %s", task.vintage, err)
            origin_rows += [
                {
                    "origin": task.vintage,
                    "model": name,
                    "status": "no forecast",
                    "reason": str(err),
                }
                for name in names
            ]
            continue
        forecasts: dict[str, pd.DataFrame] = {}
        for model in built:
            forecasts[model.name] = add_national(model.forecast(task))
            fits = pd.DataFrame(getattr(model, "fit_diagnostics", []))
            fits.insert(0, "model", model.name)
            fits.insert(1, "origin", task.vintage)
            fit_rows.append(fits)
        _check_distinct(forecasts, task.vintage)
        for name, grp in here.groupby("model"):
            forecasts[str(name)] = grp[cols]
        forecasts[BASELINE] = _reported(real, task.vintage, task)
        weeks = sorted(
            {w for s in forecasts.values() for w in s["target_end_date"]}
        )
        now = reported_share(origin, _cutoff(daily, task.vintage), weeks)
        reported = _truth(real, task.vintage).rename(
            columns={"truth": "reported"}
        )
        rng = task.rng()
        for o2 in later:
            then = reported_share(origin, _cutoff(daily, o2), weeks)
            for name, samples in forecasts.items():
                mapped = draw_reported(samples, reported, now, then, rng)
                frames.append(
                    _label(mapped, name, task.vintage, "reported", o2, daily)
                )
        frames += [
            _label(s, name, task.vintage, "complete", latest, daily)
            for name, s in forecasts.items()
        ]
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
    match = ["target", "origin", "truth_vintage", "location", "target_end_date"]
    base = scores.loc[scores["model"] == baseline, [*match, "wis", "log_wis"]]
    pair = scores.merge(
        base.rename(columns={"wis": "base_wis", "log_wis": "base_log_wis"}),
        on=match,
    )
    out = (
        pair.groupby([*by, "model"])
        .agg(
            n=("wis", "size"),
            wis=("wis", "mean"),
            base_wis=("base_wis", "mean"),
            log_wis=("log_wis", "mean"),
            base_log_wis=("base_log_wis", "mean"),
            coverage_50=("coverage_50", "mean"),
            coverage_90=("coverage_90", "mean"),
            bias=("bias", "mean"),
        )
        .reset_index()
    )
    # A zero baseline gives no relative score.
    out["rel_wis"] = out["wis"] / out["base_wis"].where(out["base_wis"] > 0)
    out["rel_log_wis"] = out["log_wis"] / out["base_log_wis"].where(
        out["base_log_wis"] > 0
    )
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
    return out[[*by, "model", *order]]


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
