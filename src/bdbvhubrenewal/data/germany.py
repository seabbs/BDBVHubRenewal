# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""The German COVID-19 hospitalisation nowcasting hub data as forecast tasks.

The German hub (Wolffram et al. 2023) collected nowcasts of the 7-day
hospitalisation incidence by Meldedatum, the date a case was reported
positive, for Germany, its 16 federal states and six age groups.
Its reporting triangle gives each Meldedatum's count as first reported, the
change reported on each of the next 80 days, and the changes after that in
one column.
`hub_triangle` turns it into one row per Meldedatum, place and report date
with the count reported by then, the layout of epinowcast's
`germany_covid19_hosp`, which `read_triangle` reads.

`nowcast_tasks` makes the tasks of one nowcast date in the
`ForecastTask` form, so every registered model runs on them unchanged.

- Each report date up to the nowcast date is a release with that date as
  its cut-off, and the Meldedatum is the onset date.
- Germany is the country. The states are provinces in one task and the age
  groups provinces in another, each summing to Germany (`place_code`).
- Hub weeks end on Saturdays. Every date of a nowcast moves forward by the
  0 to 6 days from the nowcast date to the next Saturday, so the 7-day sums
  ending on the nowcast date and 7, 14, 21 and 28 days before it are hub
  weeks (`HORIZONS`).

`hub_forecasts` moves the forecasts back and names the places as the hub
does, and `hub_target` gives the 7-day sums as each report date gives
them, the truth they are scored on.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import pandas as pd

from bdbvhubrenewal.data.dates import as_date, epiweek_end
from bdbvhubrenewal.forecasters.base import NATIONAL, ForecastTask

__all__ = [
    "AGE_GROUPS",
    "ALL_AGES",
    "GERMANY",
    "HORIZONS",
    "PLACE_NAMES",
    "STATES",
    "hub_forecasts",
    "hub_target",
    "hub_triangle",
    "level",
    "nowcast_tasks",
    "place_code",
    "read_triangle",
]

GERMANY: Final = "DE"
"""The hub's code for Germany."""

ALL_AGES: Final = "00+"
"""The hub's age group of all ages."""

AGE_GROUPS: Final = ("00-04", "05-14", "15-34", "35-59", "60-79", "80+")
"""The hub's age groups, for Germany only."""

STATES: Final = {
    "DE-BB": "Brandenburg",
    "DE-BE": "Berlin",
    "DE-BW": "Baden-Württemberg",
    "DE-BY": "Bavaria",
    "DE-HB": "Bremen",
    "DE-HE": "Hesse",
    "DE-HH": "Hamburg",
    "DE-MV": "Mecklenburg-Vorpommern",
    "DE-NI": "Lower Saxony",
    "DE-NW": "North Rhine-Westphalia",
    "DE-RP": "Rhineland-Palatinate",
    "DE-SH": "Schleswig-Holstein",
    "DE-SL": "Saarland",
    "DE-SN": "Saxony",
    "DE-ST": "Saxony-Anhalt",
    "DE-TH": "Thuringia",
}
"""The hub's codes of the federal states and their names."""

PLACE_NAMES: Final = {
    GERMANY: "Germany",
    **STATES,
    **{a: f"Age {a}" for a in AGE_GROUPS},
}
"""Names of the places the benchmark scores, by `hub_forecasts` place."""

HORIZONS: Final = (0, -7, -14, -21, -28)
"""Days from the nowcast date to the end of each 7-day sum forecast."""

_DAILY = ["vintage", "location", "onset_date", "observed", "imputed", "cutoff"]


def place_code(location: str, age_group: str = ALL_AGES) -> str:
    """Return the location code a hub place has in the tasks.

    Parameters
    ----------
    location : str
        The hub's location, `DE` or a state such as `DE-BY`.
    age_group : str
        The hub's age group. Only Germany has age groups other than
        `ALL_AGES`.

    Returns
    -------
    str
        `CD` for Germany, `CD-<state>` for a state and `CD-A<i>` for the
        i-th of `AGE_GROUPS`.
    """
    if age_group != ALL_AGES:
        return f"{NATIONAL}-A{AGE_GROUPS.index(age_group)}"
    if location == GERMANY:
        return NATIONAL
    return f"{NATIONAL}-{location.removeprefix(f'{GERMANY}-')}"


def _place(code: str) -> str:
    """Return the place, as in `PLACE_NAMES`, of a task location code."""
    if code == NATIONAL:
        return GERMANY
    suffix = code.removeprefix(f"{NATIONAL}-")
    if suffix.startswith("A"):
        return AGE_GROUPS[int(suffix[1:])]
    return f"{GERMANY}-{suffix}"


def level(place: str) -> str:
    """Return the paper's level of a place.

    Parameters
    ----------
    place : str
        A place in `PLACE_NAMES`.

    Returns
    -------
    str
        `national` for Germany, `age` for an age group and `states` for a
        state.
    """
    if place == GERMANY:
        return "national"
    return "age" if place in AGE_GROUPS else "states"


def hub_triangle(path: Path | str) -> pd.DataFrame:
    """Read the hub's reporting triangle as counts by report date.

    The changes after 80 days are put at 81 days, as epinowcast's
    `germany_covid19_hosp` does. A missing change is a report date after
    the file was made, so it has no row.

    Parameters
    ----------
    path : pathlib.Path or str
        The hub's `COVID-19_hospitalizations.csv`: columns `date`,
        `location`, `age_group`, `value_0d` to `value_80d` and
        `value_>80d`.

    Returns
    -------
    pandas.DataFrame
        As `read_triangle`.
    """
    wide = pd.read_csv(path)
    changes = [c for c in wide.columns if c.startswith("value_")]
    long = wide.melt(
        id_vars=["date", "location", "age_group"],
        value_vars=changes,
        var_name="delay",
        value_name="change",
    ).dropna(subset=["change"])
    long["delay"] = long["delay"].map({c: i for i, c in enumerate(changes)})
    long = long.sort_values(["location", "age_group", "date", "delay"])
    long["confirm"] = (
        long.groupby(["location", "age_group", "date"])["change"]
        .cumsum()
        .astype("int64")
    )
    report = pd.to_datetime(long["date"]) + pd.to_timedelta(
        long["delay"], unit="D"
    )
    out = pd.DataFrame(
        {
            "reference_date": long["date"],
            "location": long["location"],
            "age_group": long["age_group"],
            "confirm": long["confirm"],
            "report_date": report.dt.date.astype(str),
        }
    )
    return out.reset_index(drop=True)


def read_triangle(path: Path | str) -> pd.DataFrame:
    """Read counts by Meldedatum and report date, as epinowcast lays them out.

    Parameters
    ----------
    path : pathlib.Path or str
        A CSV with the columns below, such as the test fixture taken from
        epinowcast's `germany_covid19_hosp`.

    Returns
    -------
    pandas.DataFrame
        Columns `reference_date` (the Meldedatum), `location`,
        `age_group`, `confirm` (the count reported by the report date) and
        `report_date`, dates as ISO strings.
    """
    text = ("reference_date", "location", "age_group", "report_date")
    return pd.read_csv(
        path, dtype={**dict.fromkeys(text, "str"), "confirm": "int64"}
    )


def _shift(nowcast_date: str) -> dt.timedelta:
    """Return the days from a nowcast date to the next Saturday, or 0."""
    return epiweek_end(nowcast_date) - as_date(nowcast_date)


def _moved(dates: pd.Series, shift: dt.timedelta) -> pd.Series:
    return (pd.to_datetime(dates) + shift).dt.date.astype(str)


def nowcast_tasks(
    triangle: pd.DataFrame,
    nowcast_date: str,
    n_samples: int,
    seed: int,
) -> list[ForecastTask]:
    """Return the states task and the age groups task of a nowcast date.

    Parameters
    ----------
    triangle : pandas.DataFrame
        Counts by Meldedatum and report date (`read_triangle`).
    nowcast_date : str
        The nowcast date, ISO format. Report dates after it are left out.
    n_samples : int
        Samples per forecast.
    seed : int
        Seed of the tasks' random numbers.

    Returns
    -------
    list of ForecastTask
        The task of the states in the triangle, then that of the age
        groups, each with Germany as the country. Dates are moved to the
        next Saturday, so the reference date is the nowcast date moved and
        the weeks end `HORIZONS` days from it.
    """
    shift = _shift(nowcast_date)
    ref = as_date(nowcast_date) + shift
    # The hub has age groups for every state, but only Germany's are
    # nowcast.
    rows = triangle.loc[
        (triangle["report_date"] <= nowcast_date)
        & (triangle["confirm"] != 0)
        & (
            (triangle["age_group"] == ALL_AGES)
            | (triangle["location"] == GERMANY)
        )
    ]
    rows = _carried(rows)
    vintage = _moved(rows["report_date"], shift)
    daily = pd.DataFrame(
        {
            "vintage": vintage,
            "location": [
                place_code(loc, age)
                for loc, age in zip(
                    rows["location"], rows["age_group"], strict=True
                )
            ],
            "onset_date": _moved(rows["reference_date"], shift),
            "observed": rows["confirm"],
            "imputed": 0,
            "cutoff": vintage,
        }
    )[_DAILY]
    weekly = _weekly(daily)
    present = set(daily["location"])
    groups = (
        tuple(place_code(s) for s in STATES if place_code(s) in present),
        tuple(
            place_code(GERMANY, a)
            for a in AGE_GROUPS
            if place_code(GERMANY, a) in present
        ),
    )
    weeks = tuple(
        (ref + dt.timedelta(days=h)).isoformat() for h in sorted(HORIZONS)
    )
    tasks = []
    for places in groups:
        keep = {NATIONAL, *places}
        own = daily.loc[daily["location"].isin(keep)].reset_index(drop=True)
        history = weekly.loc[weekly["location"].isin(keep)]
        tasks.append(
            ForecastTask(
                vintage=ref.isoformat(),
                reference_date=ref,
                kind="real",
                cutoff=ref.isoformat(),
                target=history.loc[
                    history["vintage"] == ref.isoformat()
                ].reset_index(drop=True),
                delays=None,
                locations=places,
                target_end_dates=weeks,
                n_samples=n_samples,
                seed=seed,
                target_history=history.reset_index(drop=True),
                daily_history=own,
            )
        )
    return tasks


def _carried(rows: pd.DataFrame) -> pd.DataFrame:
    """Carry each Meldedatum's last count to every later report date.

    The triangle has no row for a Meldedatum after its last change column,
    81 days on, so without this a release would drop the older dates.
    """
    keys = ["location", "age_group", "reference_date"]
    reports = pd.DataFrame({"report_date": sorted(set(rows["report_date"]))})
    last = rows.sort_values("report_date").drop_duplicates(keys, keep="last")
    later = last.rename(columns={"report_date": "last"}).merge(
        reports, how="cross"
    )
    later = later.loc[later["report_date"] > later["last"]]
    return pd.concat([rows, later[rows.columns]], ignore_index=True)


def _weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Sum each release's daily counts to hub weeks, as the target table."""
    ends = {d: epiweek_end(d).isoformat() for d in set(daily["onset_date"])}
    out = (
        daily.assign(target_end_date=daily["onset_date"].map(ends))
        .groupby(["vintage", "location", "target_end_date"], as_index=False)[
            ["observed", "imputed"]
        ]
        .sum()
    )
    out["total"] = out["observed"] + out["imputed"]
    out["cutoff"] = out["vintage"]
    out["location_name"] = [PLACE_NAMES[_place(c)] for c in out["location"]]
    out.insert(0, "kind", "real")
    return out[
        [
            "kind",
            "vintage",
            "location",
            "location_name",
            "target_end_date",
            "observed",
            "imputed",
            "total",
            "cutoff",
        ]
    ]


def hub_forecasts(
    samples: pd.DataFrame, task: ForecastTask, nowcast_date: str
) -> pd.DataFrame:
    """Return one task's forecasts on the hub's dates and places.

    Germany is taken from the states task, as the sum of the states, and
    left out of the age groups task.

    Parameters
    ----------
    samples : pandas.DataFrame
        `run_forecasts` output for `task` alone.
    task : ForecastTask
        A task of `nowcast_tasks`.
    nowcast_date : str
        Its nowcast date.

    Returns
    -------
    pandas.DataFrame
        Columns `model`, `location` (the place, as in `PLACE_NAMES`),
        `forecast_date`, `target_end_date`, `horizon` (days, as
        `HORIZONS`), `sample` and `value`.
    """
    ages = any(_place(c) in AGE_GROUPS for c in task.locations)
    rows = samples.loc[samples["as_of"].isna()]
    if ages:
        rows = rows.loc[rows["location"] != NATIONAL]
    shift = _shift(nowcast_date)
    end = pd.to_datetime(rows["target_end_date"]) - shift
    return pd.DataFrame(
        {
            "model": rows["model"].to_numpy(),
            "location": [_place(c) for c in rows["location"]],
            "forecast_date": nowcast_date,
            "target_end_date": end.dt.date.astype(str).to_numpy(),
            "horizon": (end - pd.Timestamp(nowcast_date)).dt.days.to_numpy(),
            "sample": rows["sample"].to_numpy(),
            "value": rows["value"].to_numpy(),
        }
    )


def hub_target(triangle: pd.DataFrame, ends: Sequence[str]) -> pd.DataFrame:
    """Return 7-day sums as each report date gives them, as a target table.

    Parameters
    ----------
    triangle : pandas.DataFrame
        Counts by Meldedatum and report date (`read_triangle`).
    ends : sequence of str
        Last days of the 7-day sums.

    Returns
    -------
    pandas.DataFrame
        One row per place in `PLACE_NAMES`, end and report date from the
        end on: `kind` (`real`), `vintage` and `cutoff` (the report date),
        `location` (the place), `target_end_date` (the end) and `total`.
    """
    places = triangle["location"].where(
        triangle["age_group"] == ALL_AGES, triangle["age_group"]
    )
    keep = places.isin(set(PLACE_NAMES)) & (
        (triangle["age_group"] == ALL_AGES) | (triangle["location"] == GERMANY)
    )
    first = (as_date(min(ends)) - dt.timedelta(days=6)).isoformat()
    rows = triangle.loc[keep & (triangle["reference_date"] >= first)]
    reports = sorted(set(rows["report_date"]) | set(ends))
    days = pd.date_range(first, max(ends), freq="D").date.astype(str)
    frames = []
    for place, own in rows.groupby(places.loc[rows.index]):
        # Count of each Meldedatum (row) by each report date (column).
        counts = (
            own.pivot_table(
                index="reference_date",
                columns="report_date",
                values="confirm",
                aggfunc="last",
            )
            .reindex(index=days, columns=reports)
            .ffill(axis=1)
            .fillna(0.0)
        )
        sums = counts.rolling(7).sum()
        for end in ends:
            total = sums.loc[end]
            total = total.loc[total.index >= end]
            frames.append(
                pd.DataFrame(
                    {
                        "kind": "real",
                        "vintage": total.index,
                        "cutoff": total.index,
                        "location": place,
                        "target_end_date": end,
                        "total": total.to_numpy(),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)
