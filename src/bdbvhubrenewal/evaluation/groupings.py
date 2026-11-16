# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""How the Chronos-2 models' places are split into blocks, compared.

Four arms, each a Chronos-2 model with one `FoundationModel.grouping`:

- `joint`: every place in one block;
- `level`: one block per level;
- `bottom`: the lowest level alone, summed draw by draw to the levels
  above;
- `paired`: the lowest level of `bottom`, paired into trajectories by the
  rolling copula (`forecasters.copula.RollingCopula`) and then summed.

On BDBV the levels are the country, the provinces and the zones.
On the German benchmark they are Germany, the states and the age groups.
Its states and age groups tasks (`data.germany.nowcast_tasks`) are merged
into one task for `joint`; otherwise each task is fitted apart and Germany
is taken from the states task, as `hub_forecasts` does.

`release_grouping_forecasts` and `germany_grouping_samples` fit the
arms. `pair_and_sum` makes the `paired` arm from the `bottom` arm's draws.
Arms are named `<model>:<grouping>` (`arm_name`).
"""

from __future__ import annotations

import dataclasses
import functools
import re
from collections.abc import Callable, Sequence
from typing import Final

import numpy as np
import pandas as pd

from bdbvhubrenewal.data.dates import as_date
from bdbvhubrenewal.data.zones import weekly_zone_counts
from bdbvhubrenewal.evaluation.nowcast import (
    ReleaseForecasts,
    forecast_releases,
)
from bdbvhubrenewal.forecasters import get_model, slots
from bdbvhubrenewal.forecasters.base import NATIONAL, ForecastTask
from bdbvhubrenewal.forecasters.copula import RollingCopula
from bdbvhubrenewal.forecasters.foundation import (
    SLOTS_PREFIX,
    FoundationModel,
    Series,
    chronos_sampler,
    release_tag,
    resolve_checkpoint,
)

__all__ = [
    "AGGREGATION_COLUMNS",
    "CHECKPOINT",
    "FITTED_GROUPINGS",
    "GROUPING_ARMS",
    "OPTIONS",
    "SLOTS_RULE",
    "SLOTS_RULES",
    "ZERO_SHOT",
    "aggregation_rows",
    "arm_name",
    "cased_group_slots",
    "check_checkpoint_rows",
    "checkpoint_label",
    "checkpoint_rows",
    "germany_grouping_samples",
    "grouping_model",
    "merge_checkpoint_rows",
    "merged_task",
    "pair_and_sum",
    "release_grouping_forecasts",
    "rolling_zone_truth",
    "sample_coverage",
    "slots_rule",
    "split_arm",
]

FITTED_GROUPINGS: Final = ("joint", "level", "bottom")
"""The arms fitted by Chronos-2, each a `FoundationModel.grouping`."""

GROUPING_ARMS: Final = (*FITTED_GROUPINGS, "paired")
"""Every arm: the fitted ones and `paired`, made from `bottom`."""

OPTIONS: Final = {
    "joint": "joint",
    "level": "level",
    "bottom": "zones_summed",
    "paired": "zones_copula",
}
"""The paper's name of each arm."""

REFERENCE: Final = "randomwalknowcast"
"""The model every arm is compared with, option `reference`."""

AGGREGATION_COLUMNS: Final = (
    "study",
    "model",
    "option",
    "level",
    "location",
    "origin_date",
    "target_date",
    "wis",
    "log_wis",
    "cover50",
    "cover95",
)
"""Columns of the paper's aggregation scores (`aggregation_rows`)."""

_SAMPLES = ["model", "origin", "location", "target_end_date", "sample"]

CHECKPOINT: Final = "checkpoint"
"""Column of the checkpoint a row's model loaded (`checkpoint_rows`)."""

ZERO_SHOT: Final = "zero-shot"
"""The `CHECKPOINT` of the zero-shot models' and the reference's rows."""

SLOTS_RULE: Final = "slots_rule"
"""Column of how a row's blocks gave out slots (`SLOTS_RULES`)."""

SLOTS_RULES: Final = {
    "none": "no slots: zero-shot, the reference, or a checkpoint without",
    "all": "every place of each block has a slot, as in forecasting",
    "cases-only": (
        "a block had more places than slots, so only places with cases in "
        "its context have slots, the others none (`cased_group_slots`)"
    ),
}
"""The values of `SLOTS_RULE`."""

_LABEL = re.compile(rf"^(?:{SLOTS_PREFIX})?[0-9a-f]{{12}}$")


def arm_name(model: str, grouping: str, checkpoint: str | None = None) -> str:
    """Return an arm's name, `<model>:<grouping>`.

    A checkpoint is added after the model as `@<checkpoint>`.
    """
    at = "" if checkpoint is None else f"@{checkpoint}"
    return f"{model}{at}:{grouping}"


def grouping_model(
    model: str, grouping: str, checkpoint: str | None = None
) -> FoundationModel:
    """Return a registered Chronos-2 model with a grouping and checkpoint.

    Parameters
    ----------
    model : str
        A registered Chronos-2 model.
    grouping : str
        One of `FITTED_GROUPINGS`.
    checkpoint : str or None
        A fine-tuned checkpoint's settings hash, or None for the model's.

    Returns
    -------
    FoundationModel
    """
    base = get_model(model)
    if not isinstance(base, FoundationModel):
        raise TypeError(f"{model} is not a Chronos-2 model")
    built = dataclasses.replace(base, grouping=grouping)
    if checkpoint is not None:
        built = dataclasses.replace(
            built, checkpoint=resolve_checkpoint(checkpoint)
        )
    if built.checkpoint is not None and built.sample is None:
        # A slots checkpoint gives blocks over its slots `cased_group_slots`.
        built = dataclasses.replace(
            built,
            sample=functools.partial(
                chronos_sampler,
                checkpoint=built.checkpoint,
                group_slots=cased_group_slots,
            ),
        )
    return built


def release_grouping_forecasts(
    tasks: Sequence[ForecastTask],
    target: pd.DataFrame,
    model: str,
    grouping: str,
    checkpoint: str | None = None,
) -> ReleaseForecasts:
    """Forecast from every release with one model and grouping.

    Parameters
    ----------
    tasks : sequence of ForecastTask
        The release backtest's tasks (`evaluation.store.origin_tasks`).
    target : pandas.DataFrame
        The tidy target table.
    model : str
        A registered Chronos-2 model.
    grouping : str
        One of `FITTED_GROUPINGS`.
    checkpoint : str, optional
        A fine-tuned checkpoint's settings hash to load in place of the
        model's own (`resolve_checkpoint`).

    Returns
    -------
    ReleaseForecasts
        As `forecast_releases`, with the model named by `arm_name`.
    """
    run = forecast_releases(
        tasks, target, [grouping_model(model, grouping, checkpoint)]
    )
    name = arm_name(model, grouping, checkpoint)
    return ReleaseForecasts(
        **{
            k: v.assign(model=name) if "model" in v else v
            for k, v in dataclasses.asdict(run).items()
        }
    )


def rolling_zone_truth(daily_zones: pd.DataFrame, delay: int) -> pd.DataFrame:
    """Return each zone-week's count in the first release `delay` days on.

    Parameters
    ----------
    daily_zones : pandas.DataFrame
        Daily zone onsets of every release (`data.load_daily_zones`).
    delay : int
        $D$, days from the end of a week to its truth's release cut-off.

    Returns
    -------
    pandas.DataFrame
        `location`, `target_end_date`, `truth` (0 for a zone the release
        has no onset in) and `truth_vintage`. Weeks no release has reached
        have no rows.
    """
    cutoffs = (
        daily_zones.groupby("vintage")["cutoff"].max().map(as_date)
    ).sort_values()
    counts = {v: weekly_zone_counts(daily_zones, v) for v in cutoffs.index}
    weeks = sorted({w for c in counts.values() for w in c["target_end_date"]})
    zones = sorted({z for c in counts.values() for z in c["location"]})
    rows = []
    for week in weeks:
        due = as_date(week) + pd.Timedelta(days=delay)
        later = cutoffs[cutoffs >= due]
        if later.empty:
            continue
        release = later.index[0]
        c = counts[release]
        got = c.loc[c["target_end_date"] == week].set_index("location")
        rows.append(
            pd.DataFrame(
                {
                    "location": zones,
                    "target_end_date": week,
                    "truth": got["total"]
                    .reindex(zones, fill_value=0)
                    .to_numpy(),
                    "truth_vintage": release,
                }
            )
        )
    cols = ["location", "target_end_date", "truth", "truth_vintage"]
    return (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=cols)
    )


def pair_and_sum(
    samples: pd.DataFrame,
    truth: pd.DataFrame,
    seed: int,
    parent: Callable[[str], str] | None = None,
    floors: pd.DataFrame | None = None,
    copula: RollingCopula | None = None,
    national: str = NATIONAL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair the lowest level's draws by the rolling copula, then sum up.

    Each lowest-level place and week keeps its draws (`RollingCopula`),
    estimated from each model's own matured errors before each origin.
    The paired draws are summed sample by sample to their parents
    (`parent`), and the parents, or the places themselves with no
    `parent`, to the country. A sum is raised to its floor where it is
    below.

    Parameters
    ----------
    samples : pandas.DataFrame
        The lowest level's draws: `model`, `origin`, `location`,
        `target_end_date`, `sample` and `value`.
    truth : pandas.DataFrame
        The lowest level's truth: `location`, `target_end_date`, `truth`
        and `truth_vintage`.
    seed : int
        Seed of the pairing.
    parent : callable, optional
        A place's parent, such as `province_of`. None to sum to the
        country directly.
    floors : pandas.DataFrame, optional
        The reported count at each origin: `origin`, `location`,
        `target_end_date` and `floor`.
    copula : RollingCopula, optional
        The copula, its defaults by default.
    national : str
        The country's location code.

    Returns
    -------
    tuple of pandas.DataFrame
        The paired draws of every level, columns of `samples`, with the
        country's location `national`, and each model and origin's copula
        fit.
    """
    copula = copula or RollingCopula(national=national)
    cols = [*_SAMPLES, "value"]
    paired, fits = copula.pair(samples[cols], truth, seed)
    levels = [paired]
    lowest = paired
    if parent is not None:
        up = lowest.assign(location=lowest["location"].map(parent))
        lowest = _floor(
            up.groupby(_SAMPLES, as_index=False)["value"].sum(), floors
        )
        levels.append(lowest)
    country = lowest.assign(location=national)
    levels.append(
        _floor(country.groupby(_SAMPLES, as_index=False)["value"].sum(), floors)
    )
    out = pd.concat(levels, ignore_index=True)[cols]
    return out, fits


def _floor(sums: pd.DataFrame, floors: pd.DataFrame | None) -> pd.DataFrame:
    """Raise each sum to its place's reported count at the origin."""
    if floors is None:
        return sums
    keys = ["origin", "location", "target_end_date"]
    out = sums.merge(floors[[*keys, "floor"]], on=keys, how="left")
    out["value"] = np.maximum(out["value"], out["floor"].fillna(0.0))
    return out.drop(columns="floor")


def merged_task(states: ForecastTask, ages: ForecastTask) -> ForecastTask:
    """Return one task of Germany, its states and its age groups.

    Parameters
    ----------
    states, ages : ForecastTask
        The two tasks of a nowcast date (`data.germany.nowcast_tasks`).

    Returns
    -------
    ForecastTask
        The states task with the age groups' places and rows added.
    """

    def both(a: pd.DataFrame | None, b: pd.DataFrame | None) -> pd.DataFrame:
        return (
            pd.concat([a, b], ignore_index=True)
            .drop_duplicates()
            .reset_index(drop=True)
        )

    return dataclasses.replace(
        states,
        locations=(*states.locations, *ages.locations),
        target=both(states.target, ages.target),
        target_history=both(states.target_history, ages.target_history),
        daily_history=both(states.daily_history, ages.daily_history),
    )


def germany_grouping_samples(
    model: str,
    grouping: str,
    triangle: pd.DataFrame,
    date: str,
    n_samples: int,
    seed: int,
    checkpoint: str | None = None,
) -> pd.DataFrame:
    """Nowcast one German date with one model and grouping, as samples.

    Parameters
    ----------
    model : str
        A registered model. `randomwalknowcast` takes no grouping.
    grouping : str
        One of `FITTED_GROUPINGS`, ignored for a model with none.
    triangle : pandas.DataFrame
        Counts by Meldedatum and report date.
    date : str
        The nowcast date.
    n_samples, seed : int
        Samples per forecast, and the seed.
    checkpoint : str, optional
        As for `release_grouping_forecasts`.

    Returns
    -------
    pandas.DataFrame
        `model` (`arm_name`, or the model's name with no grouping),
        `origin` (the date), `location` (the place), `target_end_date`,
        `horizon` (days), `sample` and `value`, on the hub's dates and
        places. Germany is from the states task, or the merged task for
        `joint`.
    """
    from bdbvhubrenewal.data.germany import hub_forecasts, nowcast_tasks
    from bdbvhubrenewal.evaluation.backtest import run_forecasts

    states, ages = nowcast_tasks(triangle, date, n_samples, seed)
    built = get_model(model)
    label = model
    if hasattr(built, "grouping"):
        built = grouping_model(model, grouping, checkpoint)
        label = arm_name(model, grouping, checkpoint)
    if label.endswith(":joint"):
        runs = [(merged_task(states, ages), states)]
    else:
        runs = [(states, states), (ages, ages)]
    frames = [
        hub_forecasts(run_forecasts([built], [task]), as_task, date)
        for task, as_task in runs
    ]
    out = pd.concat(frames, ignore_index=True).rename(
        columns={"forecast_date": "origin"}
    )
    return out.assign(model=label)[
        [*_SAMPLES[:4], "horizon", "sample", "value"]
    ]


def split_arm(arm: str) -> tuple[str, str]:
    """Return an arm's model and its option, as the paper names them.

    Parameters
    ----------
    arm : str
        An `arm_name`, or `REFERENCE`.

    Returns
    -------
    tuple of str
        The model, with any `@<checkpoint>`, and its option in `OPTIONS`,
        `reference` for `REFERENCE`.
    """
    if arm == REFERENCE:
        return arm, "reference"
    model, _, grouping = arm.rpartition(":")
    return model, OPTIONS[grouping]


def sample_coverage(
    samples: pd.DataFrame, truth: pd.DataFrame, level: float
) -> pd.DataFrame:
    """Return whether each truth is in its samples' central interval.

    Parameters
    ----------
    samples : pandas.DataFrame
        `model`, `origin`, `location`, `target_end_date`, `sample` and
        `value`.
    truth : pandas.DataFrame
        `location`, `target_end_date` and `truth`.
    level : float
        The interval's width, such as 0.95.

    Returns
    -------
    pandas.DataFrame
        One row per scored cell: `model`, `origin`, `location`,
        `target_end_date` and `cover<100 level>`, 1 inside (bounds
        included) and 0 outside, as `scoring.coverage_sample`.
    """
    cell = _SAMPLES[:4]
    bounds = (
        samples.groupby(cell)["value"]
        .quantile([(1 - level) / 2, (1 + level) / 2])
        .unstack()
    )
    bounds.columns = ["lo", "hi"]
    rows = bounds.reset_index().merge(
        truth[["location", "target_end_date", "truth"]],
        on=["location", "target_end_date"],
    )
    inside = (rows["lo"] <= rows["truth"]) & (rows["truth"] <= rows["hi"])
    name = f"cover{round(level * 100)}"
    return rows[cell].assign(**{name: inside.astype(float)})


def aggregation_rows(scores: pd.DataFrame, study: str) -> pd.DataFrame:
    """Return scores as the paper's aggregation table.

    Only cells `REFERENCE` scored are kept, so every option is matched
    against it.

    Parameters
    ----------
    scores : pandas.DataFrame
        `model` (an arm or `REFERENCE`), `origin`, `location`,
        `target_end_date`, `level`, `wis`, `log_wis`, `cover50` and
        `cover95`.
    study : str
        `ebola` or `germany`.

    Returns
    -------
    pandas.DataFrame
        `AGGREGATION_COLUMNS`.
    """
    cell = ["origin", "location", "target_end_date"]
    ref = scores.loc[scores["model"] == REFERENCE, cell].drop_duplicates()
    rows = scores.merge(ref, on=cell)
    split = rows["model"].map(split_arm)
    out = rows.assign(
        study=study,
        model=split.str[0],
        option=split.str[1],
        origin_date=rows["origin"],
        target_date=rows["target_end_date"],
    )
    return out[list(AGGREGATION_COLUMNS)].reset_index(drop=True)


def checkpoint_label(label: str) -> tuple[str, str]:
    """Return a checkpoint label's settings and its release.

    Parameters
    ----------
    label : str
        `slots-<hash>` for a checkpoint with per-series slots, or a plain
        settings hash.

    Returns
    -------
    tuple of str
        The value of `BDBV_CHRONOS_FT` that pins it, and its release tag
        (`forecasters.foundation.release_tag`).

    Raises
    ------
    ValueError
        If the label is not a checkpoint's.
    """
    if not _LABEL.match(label):
        raise ValueError(
            f"{label!r} is no checkpoint label: use slots-<hash> or <hash>, "
            "a 12-character settings hash."
        )
    return label, release_tag(label)


def checkpoint_rows(
    scores: pd.DataFrame,
    reference: pd.DataFrame,
    study: str,
    checkpoint: str,
) -> pd.DataFrame:
    """Return one checkpoint's arms as the paper's rows, with a checkpoint.

    Only the cells the reference has in `reference` are kept, so the rows
    match the zero-shot ones.

    Parameters
    ----------
    scores : pandas.DataFrame
        As for `aggregation_rows`, with arms `arm_name(model, grouping,
        checkpoint)` and no reference rows.
    reference : pandas.DataFrame
        The paper's rows (`AGGREGATION_COLUMNS`) with the reference's.
    study : str
        `ebola` or `germany`.
    checkpoint : str
        The checkpoint's label, as in the arm names.

    Returns
    -------
    pandas.DataFrame
        `AGGREGATION_COLUMNS`, `CHECKPOINT` and `SLOTS_RULE`, the model
        named without `@<checkpoint>`.
    """
    cell = ["origin_date", "location", "target_date"]
    ref = reference.loc[
        (reference["study"] == study) & (reference["model"] == REFERENCE),
        cell,
    ].drop_duplicates()
    rows = scores.rename(
        columns={"origin": "origin_date", "target_end_date": "target_date"}
    ).merge(ref, on=cell)
    split = rows["model"].map(split_arm)
    out = rows.assign(
        study=study,
        model=split.str[0].str.replace(f"@{checkpoint}", "", regex=False),
        option=split.str[1],
        **{CHECKPOINT: checkpoint},
    )
    out[SLOTS_RULE] = slots_rule(out, checkpoint.startswith(SLOTS_PREFIX))
    cols = [*AGGREGATION_COLUMNS, CHECKPOINT, SLOTS_RULE]
    return out[cols].reset_index(drop=True)


def merge_checkpoint_rows(
    existing: pd.DataFrame, new: pd.DataFrame, checkpoint: str
) -> pd.DataFrame:
    """Return the paper's rows with one checkpoint's rows put in.

    Rows with no `CHECKPOINT` (the file as fm-agg wrote it) are the
    zero-shot models' and the reference's, `ZERO_SHOT`, with no slots
    (`SLOTS_RULE` `none`); its fine-tuned
    rows (`_ft`, `_ft_n<k>`), of older checkpoints, are dropped. Rows of
    `checkpoint` are replaced by `new`; other checkpoints' rows are kept.

    Parameters
    ----------
    existing : pandas.DataFrame
        The paper's rows, with or without `CHECKPOINT`.
    new : pandas.DataFrame
        `checkpoint_rows` of `checkpoint`.
    checkpoint : str
        The checkpoint's label.

    Returns
    -------
    pandas.DataFrame
        `AGGREGATION_COLUMNS`, `CHECKPOINT` and `SLOTS_RULE`.
    """
    old = existing
    if CHECKPOINT not in old:
        fine = old["model"].astype(str).str.contains(r"_ft(?:_|$)")
        old = old.loc[~fine].assign(**{CHECKPOINT: ZERO_SHOT})
    if SLOTS_RULE not in old:
        old = old.assign(**{SLOTS_RULE: "none"})
    if SLOTS_RULE not in new:
        new = new.assign(**{SLOTS_RULE: "none"})
    old = old.loc[old[CHECKPOINT] != checkpoint]
    cols = [*AGGREGATION_COLUMNS, CHECKPOINT, SLOTS_RULE]
    return pd.concat([old[cols], new[cols]], ignore_index=True)


def check_checkpoint_rows(rows: pd.DataFrame) -> None:
    """Check each arm of each checkpoint has every reference cell.

    Parameters
    ----------
    rows : pandas.DataFrame
        The paper's rows with `CHECKPOINT`.

    Raises
    ------
    ValueError
        If an arm has more or fewer rows than the reference in its study.
    """
    n = (
        rows.groupby(["study", "model", "option", CHECKPOINT])
        .size()
        .rename("n")
        .reset_index()
    )
    ref = n.loc[n["model"] == REFERENCE].set_index("study")["n"]
    want = n["study"].map(ref)
    bad = [
        f"{r.study} {r.model} {r.option} {r.checkpoint}: {r.n} != {w}"
        for r, w in zip(n.itertuples(), want, strict=True)
        if r.n != w
    ]
    if bad:
        raise ValueError("rows do not match the reference: " + "; ".join(bad))


def cased_group_slots(group: Series, n_slots: int) -> np.ndarray:
    """Return each row's slot, as `_group_slots`, even over the slots.

    A block with at most `n_slots` places takes the usual slots
    (`forecasters.foundation._group_slots`). A larger block, beyond what
    the slots were trained on, gives a slot only to places with a case in
    its context (in the target rows or, for cases in the newest weeks
    only, the place's lag series), drawn as there with
    `slots.EVAL_SEED`; the others, and their lag series, get -1, no slot,
    as base Chronos-2. Rows are in Chronos-2's order: the targets, then
    the covariates by sorted name, past-only ones first.

    Parameters
    ----------
    group : Series
        A ragged group with `places`.
    n_slots : int
        Slots of the model.

    Returns
    -------
    numpy.ndarray
        Each row's slot, -1 for none.

    Raises
    ------
    ValueError
        If more places have cases than there are slots.
    """
    from bdbvhubrenewal.forecasters.foundation import _group_slots

    target = np.atleast_2d(group["target"])
    if target.shape[0] <= n_slots:
        return _group_slots(group, n_slots)
    places = list(group["places"])
    past = list(group.get("past_covariates", {}))
    # Places with cases in the context: in their counts, or, for cases in
    # the newest weeks only, in their lag series (`_lags_of` keeps those of
    # places with cases only).
    lagged = {n.rsplit("_lag_", 1)[0] for n in past}
    has = (np.nansum(target, axis=1) > 0) | np.isin(places, list(lagged))
    cased = np.flatnonzero(has)
    if len(cased) > n_slots:
        raise ValueError(
            f"{len(cased)} places with cases but only {n_slots} slots"
        )
    chosen = slots.slot_ids(
        len(cased), 0, np.random.default_rng(slots.EVAL_SEED), n_slots
    )
    ids = np.full(len(places), -1)
    ids[cased] = chosen
    known = set(group.get("future_covariates", {}))
    rows = sorted(n for n in past if n not in known)
    rows += sorted(n for n in past if n in known)
    at = {p: i for i, p in enumerate(places)}
    owner = [at[n.rsplit("_lag_", 1)[0]] for n in rows]
    return np.concatenate([ids, ids[owner]]).astype(int)


def slots_rule(rows: pd.DataFrame, slots: bool) -> pd.Series:
    """Return how each row's blocks gave out slots (`SLOTS_RULES`).

    On BDBV the joint block, the zones' block of `level` and the zones of
    `bottom` (summed, or paired and summed, to the levels above) hold
    over 128 places, so `cases-only`; every other block, and every German one,
    `all`. Without slots, `none`.

    Parameters
    ----------
    rows : pandas.DataFrame
        The paper's rows: `study`, `option` and `level`.
    slots : bool
        Whether the checkpoint has slots.

    Returns
    -------
    pandas.Series
        `none`, `all` or `cases-only`, on `rows`' index.
    """
    if not slots:
        return pd.Series("none", index=rows.index)
    big = (rows["study"] == "ebola") & (
        rows["option"].isin(["joint", "zones_summed", "zones_copula"])
        | ((rows["option"] == "level") & (rows["level"] == "zones"))
    )
    return pd.Series(np.where(big, "cases-only", "all"), index=rows.index)
