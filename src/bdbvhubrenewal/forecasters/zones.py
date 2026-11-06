# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Health-zone forecasts allocated from any model's province samples.

Each province sample is split across the province's zones, so the zone
samples sum to it, sample by sample and week by week.
The national total stays the sum of the provinces.

For a province $p$ with zones $z$, let $o_z(w)$ be zone $z$'s published
count for week $w$ at the vintage and $O(w) = \sum_z o_z(w)$.
The split shares of province sample $k$ are drawn once and kept for all
of its weeks:

$$
\theta^{(k)} \sim \text{Dirichlet}(\alpha), \qquad
\alpha_z = r_z + \kappa\, \frac{l_z + a}{\sum_{z'} (l_{z'} + a)},
$$

with $r_z$ the zone's onsets in the `recent_weeks` weeks before the
vintage and $l_z$ those in the `longer_weeks` weeks before it.
So recent shares are shrunk towards the longer run with strength
$\kappa$ (`prior_strength`), and the floor $a$ (`floor`) gives a zone
with no cases a small share, so it gets near-zero draws.
A province sample $Y^{(k)}(w)$ keeps what each zone has published and
splits the rest:

$$
y^{(k)}_z(w) = o_z(w) + m_z, \qquad
m \sim \text{Multinomial}\big(Y^{(k)}(w) - O(w),\ \theta^{(k)}\big).
$$

Over samples this is a Dirichlet-multinomial split of the unpublished
cases.
If a sample is below the published total, $y^{(k)}(w)$ is drawn without
replacement from the published zone counts instead.
Province values are rounded first, as in the hub file.

The random walk models instead run at zone level, on the zones' own
release snapshots (`zone_task`), with the zone, then province, then
national pooling of `pooling`.
Their zone samples are then made to sum to the province samples
(`reconcile_zones`).
Province sample $Y^{(k)}(w)$ is apportioned over the zones in proportion
to the zone model's own sample $x^{(k)}_z(w)$, by largest remainders, so
the values are whole numbers that sum exactly to $Y^{(k)}(w)$:

$$
y^{(k)}_z(w) \approx
  Y^{(k)}(w)\, \frac{x^{(k)}_z(w)}{\sum_{z'} x^{(k)}_{z'}(w)}.
$$

Where every zone sample is 0 the split uses the Dirichlet mean shares of
the allocator above.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bdbvhubrenewal.data.dates import as_date
from bdbvhubrenewal.data.zones import province_of
from bdbvhubrenewal.forecasters.base import (
    NATIONAL,
    SAMPLE_COLUMNS,
    ForecastModel,
    ForecastTask,
)

__all__ = [
    "ZoneAllocator",
    "check_zone_coherence",
    "reconcile_zones",
    "zone_forecast",
    "zone_task",
]

_ZONE_STREAM = 7
"""Extra seed word, so zone draws do not reuse the province stream."""


def _final(samples: pd.DataFrame) -> pd.DataFrame:
    """Return the final-count rows, without `as_of`."""
    if "as_of" not in samples:
        return samples
    return samples.loc[samples["as_of"].isna()].drop(columns="as_of")


def _window(counts: pd.DataFrame, vintage: str, weeks: int) -> pd.Series:
    """Sum each zone's counts over the weeks ending in the window."""
    end = pd.Timestamp(as_date(vintage))
    start = end - pd.Timedelta(weeks=weeks)
    dates = pd.to_datetime(counts["target_end_date"])
    rows = counts.loc[(dates > start) & (dates <= end)]
    return rows.groupby("location")["total"].sum()


@dataclass
class ZoneAllocator:
    r"""Split province samples across their health zones.

    Attributes
    ----------
    recent_weeks : int
        Weeks before the vintage whose zone counts set the recent shares.
    longer_weeks : int
        Weeks before the vintage whose zone counts set the shares the
        recent ones are shrunk towards.
    prior_strength : float
        Pseudo-cases $\kappa$ placed on the longer-run shares.
    floor : float
        Pseudo-cases $a$ added to every zone's longer-run count.
    """

    recent_weeks: int = 2
    longer_weeks: int = 16
    prior_strength: float = 100.0
    floor: float = 0.5

    def alpha(
        self, counts: pd.DataFrame, zones: tuple[str, ...], vintage: str
    ) -> np.ndarray:
        r"""Return the Dirichlet parameters of one province's zones.

        Parameters
        ----------
        counts : pandas.DataFrame
            Weekly zone counts (`weekly_zone_counts`).
        zones : tuple of str
            The province's zones.
        vintage : str
            The vintage, ISO format.

        Returns
        -------
        numpy.ndarray
            $\alpha_z$, one per zone, in the order of `zones`.
        """
        recent = _window(counts, vintage, self.recent_weeks)
        longer = _window(counts, vintage, self.longer_weeks)
        r = recent.reindex(list(zones), fill_value=0).to_numpy(dtype=float)
        lw = longer.reindex(list(zones), fill_value=0).to_numpy(dtype=float)
        prior = (lw + self.floor) / (lw + self.floor).sum()
        return np.asarray(r + self.prior_strength * prior)

    def shares(
        self, counts: pd.DataFrame, zones: tuple[str, ...], vintage: str
    ) -> dict[str, float]:
        """Return the expected share of each zone.

        Parameters
        ----------
        counts : pandas.DataFrame
            Weekly zone counts (`weekly_zone_counts`).
        zones : tuple of str
            The province's zones.
        vintage : str
            The vintage, ISO format.

        Returns
        -------
        dict
            Zone code to its share, the Dirichlet mean.
        """
        a = self.alpha(counts, zones, vintage)
        return dict(zip(zones, (a / a.sum()).tolist(), strict=True))

    def allocate(
        self,
        samples: pd.DataFrame,
        task: ForecastTask,
        counts: pd.DataFrame,
        zones: tuple[str, ...],
    ) -> pd.DataFrame:
        """Split province samples across the zones of each province.

        Parameters
        ----------
        samples : pandas.DataFrame
            A model's output for `task`. National rows and `as_of` rows
            are ignored.
        task : ForecastTask
            The task the samples answer.
        counts : pandas.DataFrame
            Weekly zone counts at the task's vintage
            (`weekly_zone_counts`).
        zones : tuple of str
            Zone codes to cover. Each province in `samples` gets all of
            its zones here.

        Returns
        -------
        pandas.DataFrame
            Columns `SAMPLE_COLUMNS`, one row per zone, week and sample.
        """
        rng = np.random.default_rng(
            [task.seed, as_date(task.vintage).toordinal(), _ZONE_STREAM]
        )
        final = _final(samples)
        final = final.loc[final["location"] != NATIONAL]
        weeks = list(task.target_end_dates)
        ids = np.arange(1, task.n_samples + 1)
        seen = counts.set_index(["location", "target_end_date"])["total"]
        frames = []
        for prov in sorted(set(final["location"])):
            mine = tuple(z for z in zones if province_of(z) == prov)
            if not mine:
                continue
            totals = (
                final.loc[final["location"] == prov]
                .pivot(
                    index="sample", columns="target_end_date", values="value"
                )
                .reindex(index=ids, columns=weeks)
                .clip(lower=0)
                .round()
                .to_numpy(dtype=np.int64)
            )
            published = np.array(
                [[int(seen.get((z, w), 0)) for w in weeks] for z in mine]
            )
            values = self._split(
                totals,
                published,
                self.alpha(counts, mine, task.vintage),
                rng,
            )
            frames.append(_zone_frame(mine, weeks, ids, values))
        if not frames:
            return pd.DataFrame(columns=list(SAMPLE_COLUMNS))
        return pd.concat(frames, ignore_index=True)[list(SAMPLE_COLUMNS)]

    @staticmethod
    def _split(
        totals: np.ndarray,
        published: np.ndarray,
        alpha: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Split `(samples, weeks)` totals into `(samples, weeks, zones)`."""
        n, n_weeks = totals.shape
        gam = rng.gamma(alpha, size=(n, alpha.size))
        rows = gam.sum(axis=1, keepdims=True)
        mean = alpha / alpha.sum()
        theta = np.where(rows > 0, gam / np.where(rows > 0, rows, 1), mean)
        extra = totals - published.sum(axis=0)[None, :]
        out = np.broadcast_to(published.T, (n, n_weeks, alpha.size)).copy()
        out += rng.multinomial(np.maximum(extra, 0), theta[:, None, :])
        for k, w in zip(*np.nonzero(extra < 0), strict=True):
            out[k, w] = rng.multivariate_hypergeometric(
                published[:, w], totals[k, w]
            )
        return out


def _zone_frame(
    mine: tuple[str, ...],
    weeks: list[str],
    ids: np.ndarray,
    values: np.ndarray,
) -> pd.DataFrame:
    """Return sample rows from `values` of shape `[samples, weeks, zones]`."""
    return pd.DataFrame(
        {
            "location": np.repeat(mine, len(ids) * len(weeks)),
            "target_end_date": np.tile(np.repeat(weeks, len(ids)), len(mine)),
            "sample": np.tile(ids, len(weeks) * len(mine)),
            "value": values.transpose(2, 1, 0).reshape(-1).astype(float),
        }
    )


def _apportion(total: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Split whole `total` of shape `[S, W]` over the last axis of `weights`.

    Largest remainders, so the parts are whole numbers that sum to `total`.
    Rows of weights must sum to more than 0.
    """
    quota = total[..., None] * weights / weights.sum(axis=-1, keepdims=True)
    base = np.floor(quota)
    left = total - base.sum(axis=-1)
    rank = np.argsort(np.argsort(-(quota - base), axis=-1, kind="stable"), -1)
    out: np.ndarray = (base + (rank < left[..., None])).astype(np.int64)
    return out


def reconcile_zones(
    zone_samples: pd.DataFrame,
    province_samples: pd.DataFrame,
    task: ForecastTask,
    counts: pd.DataFrame,
    zones: tuple[str, ...],
    allocator: ZoneAllocator | None = None,
) -> pd.DataFrame:
    """Make zone samples sum to their province samples.

    Each province sample is apportioned over the province's zones in
    proportion to the zone model's own sample of the same id and week.

    Parameters
    ----------
    zone_samples : pandas.DataFrame
        A model's output for the zone task (`zone_task`).
    province_samples : pandas.DataFrame
        The same model's output for `task`. National rows and `as_of`
        rows are ignored.
    task : ForecastTask
        The province task.
    counts : pandas.DataFrame
        Weekly zone counts at the task's vintage
        (`weekly_zone_counts`), which set the split where all the zone
        samples of a week are 0.
    zones : tuple of str
        Zone codes to cover. Each province in `province_samples` gets all
        of its zones here.
    allocator : ZoneAllocator, optional
        The fallback split. Defaults to `ZoneAllocator()`.

    Returns
    -------
    pandas.DataFrame
        Columns `SAMPLE_COLUMNS`, one row per zone, week and sample, with
        zones summing to their province (`check_zone_coherence`).
    """
    allocator = allocator or ZoneAllocator()
    final = _final(province_samples)
    final = final.loc[final["location"] != NATIONAL]
    weeks = list(task.target_end_dates)
    ids = np.arange(1, task.n_samples + 1)
    shape = {"index": "sample", "columns": "target_end_date", "values": "value"}
    frames = []
    for prov in sorted(set(final["location"])):
        mine = tuple(z for z in zones if province_of(z) == prov)
        if not mine:
            continue
        totals = (
            final.loc[final["location"] == prov]
            .pivot(**shape)
            .reindex(index=ids, columns=weeks)
            .clip(lower=0)
            .round()
            .to_numpy(dtype=np.int64)
        )
        weight = np.zeros((len(ids), len(weeks), len(mine)))
        for i, z in enumerate(mine):
            rows = zone_samples.loc[zone_samples["location"] == z]
            if not rows.empty:
                weight[:, :, i] = (
                    rows.pivot(**shape)
                    .reindex(index=ids, columns=weeks)
                    .clip(lower=0)
                    .fillna(0.0)
                    .to_numpy()
                )
        fallback = allocator.alpha(counts, mine, task.vintage)
        empty = weight.sum(axis=-1) <= 0
        weight[empty] = fallback
        values = _apportion(totals, weight)
        frames.append(_zone_frame(mine, weeks, ids, values))
    if not frames:
        return pd.DataFrame(columns=list(SAMPLE_COLUMNS))
    return pd.concat(frames, ignore_index=True)[list(SAMPLE_COLUMNS)]


def zone_forecast(
    model: ForecastModel,
    task: ForecastTask,
    province_samples: pd.DataFrame,
    counts: pd.DataFrame,
    zones: tuple[str, ...],
    daily_zones: pd.DataFrame,
    allocator: ZoneAllocator | None = None,
) -> pd.DataFrame:
    """Run a snapshot model at zone level and sum it to the provinces.

    Parameters
    ----------
    model : ForecastModel
        `randomwalk` or `randomwalknowcast`, any model that reads
        `daily_history`.
    task : ForecastTask
        The province task, with `daily_history` set.
    province_samples : pandas.DataFrame
        The model's province output for `task`.
    counts : pandas.DataFrame
        Weekly zone counts at the task's vintage.
    zones : tuple of str
        Zone codes. Those of the task's provinces are forecast.
    daily_zones : pandas.DataFrame
        The daily zone table (`load_daily_zones`).
    allocator : ZoneAllocator, optional
        The fallback split of `reconcile_zones`.

    Returns
    -------
    pandas.DataFrame
        Columns `SAMPLE_COLUMNS`, with zones summing to their province.

    Raises
    ------
    ModelUnavailableError
        As the model, if it cannot forecast the zone task.
    """
    history = task.daily_history
    daily = (
        daily_zones if history is None else pd.concat([history, daily_zones])
    )
    zt = zone_task(task, counts, zones, daily)
    raw = model.forecast(zt)
    return reconcile_zones(
        raw, province_samples, task, counts, zones, allocator
    )


def check_zone_coherence(
    provinces: pd.DataFrame, zones: pd.DataFrame
) -> list[str]:
    """Check that zone samples sum to their province samples.

    Parameters
    ----------
    provinces : pandas.DataFrame
        Province samples (`SAMPLE_COLUMNS`). Values are rounded and
        clipped at zero first, as in the hub file.
    zones : pandas.DataFrame
        Zone samples (`SAMPLE_COLUMNS`).

    Returns
    -------
    list of str
        One message per province and week whose zones do not sum to it
        in every sample. Empty when coherent.
    """
    keys = ["location", "target_end_date", "sample"]
    summed = (
        zones.assign(location=zones["location"].map(province_of))
        .groupby(keys)["value"]
        .sum()
    )
    prov = _final(provinces)
    want = (
        prov.loc[prov["location"].isin(summed.index.unique("location"))]
        .set_index(keys)["value"]
        .clip(lower=0)
        .round()
    )
    gap = summed.reindex(want.index, fill_value=np.nan) - want
    gap = gap.reindex(want.index.union(summed.index))
    bad = gap.loc[gap.isna() | (gap != 0)]
    return [
        f"{loc}, {week}: zone samples do not sum to the province samples"
        for loc, week in sorted({(i[0], i[1]) for i in bad.index})
    ]


def zone_task(
    task: ForecastTask,
    counts: pd.DataFrame,
    zones: tuple[str, ...],
    daily: pd.DataFrame | None = None,
) -> ForecastTask:
    """Return the task with zones in place of provinces.

    This is how the random walk models run at zone level. Each zone gets
    a weekly series from the first week with zone data to the vintage,
    with zeros filled in.

    Parameters
    ----------
    task : ForecastTask
        A province task.
    counts : pandas.DataFrame
        Weekly zone counts at the task's vintage.
    zones : tuple of str
        Zone codes. Those of the task's provinces are kept.
    daily : pandas.DataFrame, optional
        Daily onsets by release of the country, provinces and zones.
        The task keeps the rows of its releases for the country, its
        provinces and its zones as `daily_history`. Without it the task
        has none, so the random walk models cannot run.

    Returns
    -------
    ForecastTask
        The same weeks, vintage, delays and seed, with the zones as its
        locations and their counts as its target.
    """
    mine = tuple(z for z in zones if province_of(z) in task.locations)
    first = pd.Timestamp(counts["target_end_date"].min())
    weeks = pd.date_range(first, pd.Timestamp(as_date(task.vintage)), freq="7D")
    grid = pd.MultiIndex.from_product(
        [mine, weeks.date.astype(str)], names=["location", "target_end_date"]
    )
    target = (
        counts.set_index(["location", "target_end_date"])[
            ["observed", "imputed", "total"]
        ]
        .reindex(grid, fill_value=0)
        .reset_index()
        .assign(vintage=task.vintage, kind=task.kind)
    )
    history = None
    if daily is not None:
        keep = {NATIONAL, *task.locations, *mine}
        history = daily.loc[
            daily["location"].isin(keep) & (daily["vintage"] <= task.vintage)
        ].reset_index(drop=True)
    return dataclasses.replace(
        task,
        target=target,
        locations=mine,
        mask=(),
        report_by=(),
        daily_history=history,
    )
