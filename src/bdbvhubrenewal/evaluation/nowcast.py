# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Backtest of the random walk models on the real releases.

Every real release with a later release is an origin.
The models forecast from the releases up to the origin only
(`bdbvhubrenewal.forecasters.randomwalk`).
Weeks back $k$ count from the week holding the origin's cut-off, so
forecast weeks have $k < 0$.
Two targets are scored.

**Reported.** The count each later release $o'$ shows for every week
it covers.
Every model forecasts the final count $X$, so its samples are mapped to
the count expected at $o'$ (`expected_reported`).
The chain ladder fitted at the origin (`forecasters.chain`) gives
$\pi_o$, the share of a week's final count reported at the origin, and
$\pi_{o'}$, the share reported at the cut-off of $o'$
(`reported_share`).
With $R_o$ the count reported at the origin,

$$
\hat{y}_{o'} = R_o + q \, (X - R_o), \qquad
q = \frac{\pi_{o'} - \pi_o}{1 - \pi_o},
$$

so $q$ is the share of the count outstanding at the origin that $o'$ is
expected to report.
The map is the same for every model, applies sample by sample and keeps
their order.
For a model whose final count is the chain nowcast, it gives the chain
over the gap to $o'$.
An origin with no pair of releases within the delay window has no chain,
so no model is scored on this target from it.

**Complete weeks.** The count the latest release shows for the weeks it
reports in full, decided from the releases alone and never from a
model (`complete_weeks`).
By default a week is complete when it is at least `COMPLETE_WEEKS` weeks
old at the latest release and grew by less than `COMPLETE_GROWTH`
between the last two releases.

Stored forecasts of models too slow to run here, such as the renewal
models, and hub team forecasts are scored on both targets.

A floor, the count as reported at the origin, is scored on the same
targets.
An origin that has no pair of releases within the delay window is listed
with no forecasts.
A model with no snapshots stops the run.

Scores are WIS, coverage and bias (`bdbvhubrenewal.evaluation.scoring`)
and every table counts its forecasts, $n$.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bdbvhubrenewal.baselinenowcast import chain_growth, exact_plan
from bdbvhubrenewal.data.dates import (
    as_date,
    epiweek_end,
    reference_date_for_vintage,
)
from bdbvhubrenewal.data.snapshots import release_cutoff
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
)
from bdbvhubrenewal.forecasters.randomwalk import (
    RandomWalk,
    RandomWalkNowcast,
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
    "expected_reported",
    "pooled_summary",
    "relative_wis",
    "release_backtest",
    "reported_share",
    "stored_finals",
    "summarise",
    "weeks_back_from_cutoff",
]

log = logging.getLogger(__name__)

BASELINE = "as reported"
"""Name of the floor: the count reported at the origin, as a forecast."""

MAX_WEEKS_BACK = 3
"""Last week back from the origin's week kept in the quantiles."""

MAX_WEEKS_AHEAD = 4
"""Last forecast week after the origin's week scored and kept."""

COMPLETE_WEEKS = 4
"""Weeks a week must be old at the latest release to count as complete."""

COMPLETE_GROWTH = 0.02
"""Most a complete week may grow between the last two releases."""

NEAR_ZERO = 5.0
"""Places whose weekly count never reaches this are near zero."""

NO_CHAIN = "no delay pair at the origin, so no reported target"
"""Why a stored forecast has no score on the reported target."""

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
        One row per origin and model: `status`, `reason` and the number
        of forecasts scored on each target.
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


def reported_share(
    origin: Origin, cutoff: dt.date, weeks: Sequence[str]
) -> pd.DataFrame:
    r"""Return the share of each week's final count reported by a cut-off.

    Under the delay fitted at the origin, a day of age $a$ at the cut-off
    has $1 / G(a)$ of its final count reported, with $G$ the growth to
    $D$ along the steps of the origin's latest pair, as for
    `forecasters.chain.final_plan`.
    Days after the cut-off have none, and a week's share is the mean
    over its seven days.
    At the origin's own cut-off this is `Origin.complete`.

    Parameters
    ----------
    origin : Origin
        The fit at the origin. Only its delay factors are used.
    cutoff : datetime.date
        The cut-off of the release.
    weeks : sequence of str
        Hub week ends.

    Returns
    -------
    pandas.DataFrame
        Columns `location`, `target_end_date` and `share`, for every
        node of the origin.
    """
    fit, settings, snap = origin.fit, origin.settings, origin.snap
    d = settings.max_delay
    last = len(fit.pairs) - 1
    slot = int(fit.pair_step[last])
    ages = np.maximum(np.arange(d) + settings.clock_shift, 0)
    g = np.array(
        chain_growth(
            fit.factors,
            exact_plan(ages, int(fit.steps[slot]), slot, d),
        )
    )
    # A chain with a zero factor revises the day to nothing, so it counts
    # as fully reported, as in `build_origin`.
    by_age = np.concatenate(
        [1.0 / np.where(g > 0, g, 1.0), np.ones((g.shape[0], 1))], axis=1
    )
    first = snap.days[0].date()
    rows = []
    for w in weeks:
        days = [as_date(str(w)) - dt.timedelta(days=i) for i in range(7)]
        age = np.array([(cutoff - day).days for day in days])
        seen = (age >= 0) & np.array([day >= first for day in days])
        daily = np.where(seen[None, :], by_age[:, np.clip(age, 0, d)], 0.0)
        rows.append(
            pd.DataFrame(
                {
                    "location": snap.locations,
                    "target_end_date": str(w),
                    "share": daily.mean(axis=1),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def expected_reported(
    samples: pd.DataFrame,
    origin: Origin,
    reported: pd.DataFrame,
    cutoff: dt.date,
) -> pd.DataFrame:
    r"""Map samples of the final count to the count expected by a cut-off.

    Each sample $X$ becomes $R_o + q (X - R_o)$, floored at 0, with $R_o$
    the count reported at the origin and $q$ the share of the count
    outstanding at the origin that the delay expects to be reported by
    `cutoff` (`reported_share`).
    A week the delay counts complete at the origin takes $q = 1$ at any
    later cut-off, so the forecast final count stands.

    Parameters
    ----------
    samples : pandas.DataFrame
        Columns `location`, `target_end_date`, `sample` and `value`.
    origin : Origin
        The fit at the origin.
    reported : pandas.DataFrame
        Columns `location`, `target_end_date` and `reported`, the counts
        at the origin. Weeks not in it have 0 reported.
    cutoff : datetime.date
        The cut-off of the later release.

    Returns
    -------
    pandas.DataFrame
        The samples with mapped values, for the origin's nodes only.
    """
    keys = ["location", "target_end_date"]
    snap = origin.snap
    weeks = sorted(str(w) for w in samples["target_end_date"].unique())
    start = snap.days[snap.cut[origin.o]].date()
    now = reported_share(origin, start, weeks)
    then = reported_share(origin, cutoff, weeks)
    share = now.merge(then, on=keys, suffixes=("_o", "_l"))
    left = 1.0 - share["share_o"]
    gain = (share["share_l"] - share["share_o"]) / left.where(left > 0, 1.0)
    full = 1.0 if cutoff > start else 0.0
    share["q"] = np.where(left > 1e-9, gain.clip(0.0, 1.0), full)
    out = samples.astype({"target_end_date": str}).merge(
        share[[*keys, "q"]], on=keys
    )
    out = out.merge(
        reported[[*keys, "reported"]].astype({"target_end_date": str}),
        on=keys,
        how="left",
    )
    r = out.pop("reported").fillna(0.0)
    q = out.pop("q")
    out["value"] = np.maximum(r + q * (out["value"] - r), 0.0)
    return out[list(samples.columns)]


def stored_finals(
    forecasts: pd.DataFrame,
    hub: pd.DataFrame,
    vintages: Sequence[str],
    exclude: Sequence[str] = (),
) -> pd.DataFrame:
    """Return stored and hub samples of the final count by origin.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Backtest forecasts, as in `results/forecasts.csv.gz`. Only real
        rows of the final count, with no `as_of`, are kept.
    hub : pandas.DataFrame
        Hub team samples with `reference_date`, as from
        `hubcompare.load_hub_submissions`. Each round takes the latest
        real vintage serving it as its origin, and rounds with none are
        dropped.
    vintages : sequence of str
        The real vintages.
    exclude : sequence of str
        Models to leave out, such as those run here.

    Returns
    -------
    pandas.DataFrame
        Columns `model`, `vintage`, `location`, `target_end_date`,
        `sample` and `value`.
    """
    cols = ["model", "vintage", "location", "target_end_date", "sample",
            "value"]  # fmt: skip
    frames = []
    if not forecasts.empty:
        final = forecasts["kind"] == "real"
        if "as_of" in forecasts:
            final &= forecasts["as_of"].isna()
        frames.append(forecasts.loc[final, cols])
    if not hub.empty:
        serving: dict[str, str] = {}
        for v in sorted(vintages):
            serving[reference_date_for_vintage(v).isoformat()] = v
        rows = hub.assign(
            vintage=hub["reference_date"].astype(str).map(serving)
        )
        frames.append(rows.loc[rows["vintage"].notna(), cols])
    if not frames:
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames, ignore_index=True)
    out = out.loc[~out["model"].isin(set(exclude))].reset_index(drop=True)
    return out.astype({"vintage": str, "target_end_date": str})


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
    """Forecast from every real release and score on both targets.

    Parameters
    ----------
    tasks : sequence of ForecastTask
        One real task per release, with its release snapshots.
    daily : pandas.DataFrame
        Daily onsets by release, with every release.
    target : pandas.DataFrame
        The tidy target table.
    models : sequence of str or ForecastModel
        Registered model names, or models, run here.
    stored : pandas.DataFrame, optional
        Samples of the final count from other models, such as the renewal
        models or hub teams, with columns `model`, `vintage`, `location`,
        `target_end_date`, `sample` and `value`. Places other than the
        task's and national are dropped.
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

    for task in sorted(tasks, key=lambda t: t.vintage):
        if task.kind != "real":
            continue
        if task.daily_history is None or task.daily_history.empty:
            raise MissingSnapshots(
                f"The task for {task.vintage} has no daily release snapshots."
            )
        later = [v for v in releases if v > task.vintage]
        finals: dict[str, tuple[pd.DataFrame, bool]] = {}
        made: dict[str, np.ndarray] = {}
        for name, model in built.items():
            row: dict[str, object] = {
                "origin": task.vintage,
                "model": name,
                "status": "ran" if later else "no later release",
                "reason": "",
            }
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
            finals[name] = (samples, False)
            origin_rows.append(row)
        if not later:
            continue
        chain: Origin | None
        try:
            chain = RandomWalk().origin_at(task, errors=False)
        except ModelUnavailableError as err:
            log.warning("No chain at %s: %s", task.vintage, err)
            chain = None
        if stored is not None:
            places = {*task.locations, NATIONAL}
            rows = stored.loc[
                (stored["vintage"].astype(str) == task.vintage)
                & stored["location"].isin(places)
            ]
            for stored_name, grp in rows.groupby("model"):
                keep = grp[["location", "target_end_date", "sample", "value"]]
                finals[str(stored_name)] = (keep, True)
                origin_rows.append(
                    {
                        "origin": task.vintage,
                        "model": str(stored_name),
                        "status": "stored",
                        "reason": "" if chain is not None else NO_CHAIN,
                    }
                )
        if made:
            finals[BASELINE] = (_reported(real, task.vintage, task), False)
        reported = _truth(real, task.vintage).rename(
            columns={"truth": "reported"}
        )
        for name, (samples, kept) in finals.items():
            frames.append(
                _label(samples, name, task.vintage, "complete", latest, daily)
                .assign(stored=kept)
            )  # fmt: skip
            if chain is None:
                continue
            for o2 in later:
                cut = release_cutoff(daily, o2)
                pred = expected_reported(samples, chain, reported, cut)
                frames.append(
                    _label(pred, name, task.vintage, "reported", o2, daily)
                    .assign(stored=kept)
                )  # fmt: skip
    done = complete_weeks(
        target, daily, latest, rule=complete_rule, weeks=complete_after
    )
    return _score(frames, origin_rows, fit_rows, real, done)


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
        if target == "reported":
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
        for c in ("n_reported", "n_complete"):
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
