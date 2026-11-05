# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Health-zone nowcasts at the latest release, for the Nowcast page.

The random walk models are run on each zone and summed to its provinces
(`zone_forecast`), as in the submitted file. The page shows the zones
with cases in the recent weeks in the Nowcast page's picker, beside the
country and provinces (`place_chart_data`), then static grids by
province and a table of the newest weeks for `randomwalknowcast`.
A week with no onsets in a zone has a count of 0.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from bdbvhubrenewal.data.zones import province_of, weekly_zone_counts
from bdbvhubrenewal.forecasters import ForecastTask, get_model
from bdbvhubrenewal.forecasters.zones import zone_forecast
from bdbvhubrenewal.reporting.explorer import compact, trajectories
from bdbvhubrenewal.reporting.legend import starts_on

__all__ = [
    "ZONE_MODEL",
    "place_chart_data",
    "zone_counts",
    "zone_nowcast",
    "zone_table",
]

ZONE_MODEL = "randomwalknowcast"
"""The model shown for the zones, as submitted."""

_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
_QCOLS = ("q5", "q25", "q50", "q75", "q95")


def zone_nowcast(
    model: str,
    task: ForecastTask,
    province_samples: pd.DataFrame,
    daily_zones: pd.DataFrame,
    zones: tuple[str, ...],
) -> pd.DataFrame:
    """Forecast every zone at the task's release, summed to the provinces.

    Parameters
    ----------
    model : str
        A registered model.
    task : ForecastTask
        The province task at the latest release.
    province_samples : pandas.DataFrame
        `model`'s province samples for `task`.
    daily_zones : pandas.DataFrame
        The daily zone table (`load_daily_zones`).
    zones : tuple of str
        The hub's zone codes (`hub_zones`).

    Returns
    -------
    pandas.DataFrame
        Zone samples with `model`, `kind`, `vintage`, `reference_date` and
        `horizon` in weeks from the reference date, as `run_forecasts`.
    """
    counts = weekly_zone_counts(daily_zones, task.vintage)
    samples = zone_forecast(
        get_model(model),
        task,
        province_samples[["location", "target_end_date", "sample", "value"]],
        counts,
        zones,
        daily_zones,
    )
    ref = pd.Timestamp(task.reference_date)
    weeks = (pd.to_datetime(samples["target_end_date"]) - ref).dt.days // 7
    return samples.assign(
        model=model,
        kind="real",
        vintage=task.vintage,
        reference_date=str(task.reference_date),
        horizon=weeks,
    )


def zone_counts(
    daily_zones: pd.DataFrame,
    vintage: str,
    start: str,
    names: dict[str, str],
) -> pd.DataFrame:
    """Weekly zone counts at a release, for zones with cases since `start`.

    Parameters
    ----------
    daily_zones : pandas.DataFrame
        The daily zone table (`load_daily_zones`).
    vintage : str
        The release.
    start : str
        The first week ending shown, ISO format.
    names : dict of str to str
        Province code to name.

    Returns
    -------
    pandas.DataFrame
        Target-like rows (`kind`, `vintage`, `location`, `location_name`,
        `target_end_date`, `total`) from `start` to the release's last
        week, with 0 for weeks with no onsets, and `province`.
    """
    counts = weekly_zone_counts(daily_zones, vintage)
    recent = counts.loc[counts["target_end_date"] >= start]
    shown = sorted(set(recent["location"]))
    weeks = pd.date_range(start, counts["target_end_date"].max(), freq="7D")
    grid = pd.MultiIndex.from_product(
        [shown, weeks.strftime("%Y-%m-%d")],
        names=["location", "target_end_date"],
    )
    out = (
        counts.set_index(["location", "target_end_date"])["total"]
        .reindex(grid, fill_value=0)
        .reset_index()
    )
    province = out["location"].map(province_of)
    return out.assign(
        kind="real",
        vintage=vintage,
        province=province,
        location_name=province.map(names).fillna(province)
        + " "
        + out["location"].str[-2:],
    )


def _quantiles(samples: pd.DataFrame) -> pd.DataFrame:
    grp = samples.groupby(["location", "target_end_date"])["value"]
    out = pd.concat(
        {c: grp.quantile(p) for c, p in zip(_QCOLS, _QUANTILES, strict=True)},
        axis=1,
    )
    return out.reset_index()


def zone_table(
    samples: pd.DataFrame,
    counts: pd.DataFrame,
    weeks: int = 2,
) -> pd.DataFrame:
    """Tabulate the newest nowcast weeks by zone.

    Parameters
    ----------
    samples : pandas.DataFrame
        `zone_nowcast` output.
    counts : pandas.DataFrame
        `zone_counts` output.
    weeks : int
        Newest weeks before the reference date shown.

    Returns
    -------
    pandas.DataFrame
        One row per zone and week: the reported count, and the median and
        90% interval.
    """
    now = samples.loc[samples["horizon"] < 0]
    newest = sorted(set(now["target_end_date"]))[-weeks:]
    q = _quantiles(now.loc[now["target_end_date"].isin(newest)])
    rows = q.merge(
        counts[["location", "province", "target_end_date", "total"]],
        on=["location", "target_end_date"],
    )
    interval = (
        rows["q5"].round().astype(int).astype(str)
        + " to "
        + rows["q95"].round().astype(int).astype(str)
    )
    out = pd.DataFrame(
        {
            "province": rows["province"],
            "zone": rows["location"],
            "week ending": rows["target_end_date"],
            "reported": rows["total"],
            "median": rows["q50"].round().astype(int),
            "90% interval": interval,
        }
    )
    return out.sort_values(["province", "zone", "week ending"]).reset_index(
        drop=True
    )


def place_chart_data(
    samples: pd.DataFrame,
    counts: pd.DataFrame,
    names: Mapping[str, str],
    colours: Mapping[str, tuple[str, str, str]],
    complete_after_weeks: int,
) -> dict[str, object]:
    """Return the data `docs/javascripts/nowcastchart.js` draws.

    Parameters
    ----------
    samples : pandas.DataFrame
        Samples of any of the models at one release, for the country, the
        provinces and the zones, with `model`, `vintage`,
        `reference_date`, `location`, `target_end_date`, `sample` and
        `value`.
    counts : pandas.DataFrame
        Weekly counts at the release, `location`, `target_end_date` and
        `total`. A place with samples but no counts is shown with no data.
    names : mapping of str to str
        Place names by id. Zones not named show their code.
    colours : mapping of str to (str, str, str)
        Every model we fit, in order, to its light colour, dark colour
        and dash.
    complete_after_weeks : int
        Weeks before the release after which a week is complete.

    Returns
    -------
    dict
        `release`, `reference_date`, `complete_through` (the last week
        end that is complete), `models` (name, light and dark
        colour, dash, shown at first, and whether it has samples at the
        release), `places` (id, name, `scale` and, for a zone, its
        `province`), each scale by recent cases, most first, and per place
        `data` (week, count), `q` (model index, week, q5, q25, q50, q75,
        q95) and `traj` (model index, draw, first week, then the weekly
        values of a fixed subset of draws, the same in every place).
    """
    models = list(colours)
    # Places with forecasts but no cases are shown too, with no data.
    places = sorted(set(counts["location"]) | set(samples["location"]))
    size = (
        counts.groupby("location")["total"].sum().reindex(places, fill_value=0)
    )
    fans = set(samples["model"])

    def scale(loc: str) -> str:
        return {0: "national", 1: "provinces"}.get(loc.count("-"), "zones")

    order = {"national": 0, "provinces": 1, "zones": 2}
    shown = sorted(size.index, key=lambda x: (order[scale(x)], -size[x]))
    keep = samples.loc[samples["location"].isin(shown)]
    q = pd.concat(
        [_quantiles(grp).assign(model=m) for m, grp in keep.groupby("model")]
    )

    def _num(v: float) -> float:
        return float(np.round(v, 3))

    release = str(samples["vintage"].iloc[0])
    cut = pd.Timestamp(release) - pd.Timedelta(weeks=complete_after_weeks)
    return {
        "release": release,
        "reference_date": str(samples["reference_date"].iloc[0]),
        "complete_through": cut.date().isoformat(),
        "models": [
            {
                "name": m,
                "light": colours[m][0],
                "dark": colours[m][1],
                "dash": colours[m][2],
                "on": starts_on(m, fans),
                "fan": m in fans,
            }
            for m in models
        ],
        "places": [
            {
                "id": str(p),
                "name": names.get(str(p), str(p)),
                "scale": scale(str(p)),
                **(
                    {"province": province_of(str(p))}
                    if scale(str(p)) == "zones"
                    else {}
                ),
            }
            for p in shown
        ],
        "data": {
            str(z): [
                [str(w), int(n)]
                for w, n in zip(g["target_end_date"], g["total"], strict=True)
            ]
            for z, g in counts.groupby("location")
        },
        "q": {
            str(z): [
                [
                    models.index(r["model"]),
                    str(r["target_end_date"]),
                    *(_num(r[c]) for c in _QCOLS),
                ]
                for r in g.to_dict("records")
                if r["model"] in models
            ]
            for z, g in q.groupby("location")
        },
        "traj": {
            str(z): [
                [models.index(m), int(k), start, *(compact(v) for v in vals)]
                for m, _, _, k, start, vals in g.itertuples(index=False)
                if m in models
            ]
            for z, g in trajectories(keep).groupby("location")
        },
    }
