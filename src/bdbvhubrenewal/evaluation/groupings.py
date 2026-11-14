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
from bdbvhubrenewal.forecasters import get_model
from bdbvhubrenewal.forecasters.base import NATIONAL, ForecastTask
from bdbvhubrenewal.forecasters.copula import RollingCopula
from bdbvhubrenewal.forecasters.foundation import resolve_checkpoint

__all__ = [
    "FITTED_GROUPINGS",
    "GROUPING_ARMS",
    "arm_name",
    "germany_grouping_samples",
    "merged_task",
    "pair_and_sum",
    "release_grouping_forecasts",
    "rolling_zone_truth",
]

FITTED_GROUPINGS: Final = ("joint", "level", "bottom")
"""The arms fitted by Chronos-2, each a `FoundationModel.grouping`."""

GROUPING_ARMS: Final = (*FITTED_GROUPINGS, "paired")
"""Every arm: the fitted ones and `paired`, made from `bottom`."""

_SAMPLES = ["model", "origin", "location", "target_end_date", "sample"]


def arm_name(model: str, grouping: str, checkpoint: str | None = None) -> str:
    """Return an arm's name, `<model>:<grouping>`.

    A checkpoint is added after the model as `@<checkpoint>`.
    """
    at = "" if checkpoint is None else f"@{checkpoint}"
    return f"{model}{at}:{grouping}"


def _built(model: str, grouping: str, checkpoint: str | None) -> object:
    """Return a registered Chronos-2 model with a grouping and checkpoint."""
    built = dataclasses.replace(get_model(model), grouping=grouping)
    if checkpoint is not None:
        built = dataclasses.replace(
            built, checkpoint=resolve_checkpoint(checkpoint)
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
        tasks, target, [_built(model, grouping, checkpoint)]
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
        built = _built(model, grouping, checkpoint)
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
