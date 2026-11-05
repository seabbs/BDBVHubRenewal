# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Toto 1.0 on the reporting delays, used zero-shot (`fm_delay`).

`FMDelay` (`fm_delay`) gives the reporting triangle of the task's delay
line data to the pretrained Toto 1.0 foundation model
(`Datadog/Toto-Open-Base-1.0`) as one multivariate series and reads the
nowcast and the forecast off the same sample trajectories.
There is no fine-tuning and no fitted parameter.

**Triangle.** Each case of the vintage's delay line data has an onset
week $w$ and a delay of $d$ days, in delay class $k = \lfloor d / 7
\rfloor$, for $k < K$, with $K$ the first class past
`data.delays.DEFAULT_MAX_DELAY` days.
Negative delays go to class 0, as in the renewal model's delay counts,
and delays of $7K$ days or more (about 0.2%) are left out.
$n_{p,w,k}$ counts the cases of province $p$ in cell $(w, k)$.
A cell is complete at vintage $T$ when its last possible report day,
$w + 7k + 6$, is on or before $T$.

**Variates.** Toto forecasts every variate from the same last time
step, and its padding mask only weights its scaler, so a ragged end
cannot be masked.
Each delay class is therefore put on the report clock: variate $(p, k)$
at report week $s$ holds $n_{p, s - k, k}$.
Cell $(w, k)$ is complete exactly when $w + k \le s^*$, the latest week
whose Saturday is at least six days before $T$, so every variate is
complete up to $s^*$ and ends there.
All $P K$ variates share one group, so the model's variate attention
sees every province and delay together.
No delay covariate is added: Toto takes known-future covariates only
through variate labels learnt in fine-tuning, and its scaler maps a
constant channel to zero.

**Trajectories.** Toto samples every variate jointly for $H$ weeks past
$s^*$, enough for the last delay class of the last target week.
Each draw is rounded and floored at zero.
Sample $i$ of week $w$ is the vintage's reported total plus the new
reports of its incomplete cells,

$$
y^{(i)}_{p,w} = R_{p,w}
+ \sum_{k : w + k > s^*} \max\bigl(\hat n^{(i)}_{p,w,k} - n_{p,w,k}, 0\bigr),
$$

with $R_{p,w}$ the reported total (known and hub-imputed onsets) and
$\hat n^{(i)}_{p,w,k}$ the draw of variate $(p, k)$ at report week
$w + k$.
Weeks after the data have $R = 0$ and $n = 0$.
The delay line data hold known onsets only, so cases whose onset the
hub will impute later are not forecast.
National is the sum of provinces by sample.
Health zones have no delay line data, so the model runs at province
level only.

Toto is an optional dependency (`uv sync --extra toto`).
Without it the model raises `ModelUnavailableError` and the harness
skips it.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import pandas as pd

from bdbvhubrenewal.data.dates import as_date, epiweek_end
from bdbvhubrenewal.data.delays import DEFAULT_MAX_DELAY
from bdbvhubrenewal.forecasters.base import (
    SAMPLE_COLUMNS,
    ForecastTask,
    ModelUnavailableError,
    register_model,
)

__all__ = [
    "N_DELAYS",
    "TOTO_MODEL",
    "TOTO_REVISION",
    "DelayTriangle",
    "FMDelay",
    "Sampler",
    "delay_triangle",
    "report_series",
    "toto_sampler",
]

N_DELAYS = DEFAULT_MAX_DELAY // 7 + 1
"""Weekly delay classes $K$, the first past `DEFAULT_MAX_DELAY` days."""

TOTO_MODEL = "Datadog/Toto-Open-Base-1.0"
"""The Hugging Face checkpoint."""

TOTO_REVISION = "0411ceb27bdf7fc3e4892e99edc8ad08192dc3c5"
"""The checkpoint's commit, pinned so runs are reproducible."""

Sampler = Callable[[np.ndarray, int, int, int], np.ndarray]
"""Draws joint trajectories: `(series [V, S], horizon, n, seed)` to
`[n, V, horizon]`."""

_WEEK = dt.timedelta(days=7)


@dataclass(frozen=True)
class DelayTriangle:
    """Cases by province, onset week and weekly delay class.

    Attributes
    ----------
    weeks : tuple of str
        Onset weeks, ISO Saturdays, from the first onset in the data to
        the last week asked for.
    counts : numpy.ndarray
        Cases reported by the vintage, `[P, W, K]`.
    last : int
        Index of $s^*$, the latest report week whose cells are complete.
    """

    weeks: tuple[str, ...]
    counts: np.ndarray
    last: int


def delay_triangle(
    delays: pd.DataFrame,
    vintage: str,
    locations: tuple[str, ...],
    last_week: str,
    n_delays: int = N_DELAYS,
) -> DelayTriangle:
    """Count the vintage's cases by onset week and weekly delay class.

    Parameters
    ----------
    delays : pandas.DataFrame
        Delay line data of one vintage, with columns `location`,
        `onset_week` and `delay` (days).
    vintage : str
        The vintage date $T$.
    locations : tuple of str
        Provinces, in the order of the output.
    last_week : str
        Last onset week to include, an ISO Saturday.
    n_delays : int
        Delay classes $K$.

    Returns
    -------
    DelayTriangle
        The counts and the latest complete report week.
    """
    first = as_date(str(delays["onset_week"].min()))
    n_weeks = (as_date(last_week) - first).days // 7 + 1
    weeks = tuple((first + i * _WEEK).isoformat() for i in range(n_weeks))
    place = delays["location"].map({c: i for i, c in enumerate(locations)})
    week = (
        pd.to_datetime(delays["onset_week"]) - pd.Timestamp(first)
    ).dt.days.to_numpy() // 7
    k = np.maximum(delays["delay"].to_numpy(), 0) // 7
    keep = place.notna().to_numpy() & (k < n_delays) & (week < n_weeks)
    counts = np.zeros((len(locations), n_weeks, n_delays))
    np.add.at(
        counts,
        (place.to_numpy()[keep].astype(int), week[keep], k[keep]),
        1.0,
    )
    complete = epiweek_end(as_date(vintage) - dt.timedelta(days=12))
    return DelayTriangle(weeks, counts, (complete - first).days // 7)


def report_series(triangle: DelayTriangle) -> np.ndarray:
    """Put each delay class on the report clock, up to $s^*$.

    Parameters
    ----------
    triangle : DelayTriangle
        The counts.

    Returns
    -------
    numpy.ndarray
        `[P * K, s^* + 1]`, variate $(p, k)$ at row $p K + k$, holding
        $n_{p, s - k, k}$ at report week $s$ and 0 for $s < k$.
    """
    n_loc, _, n_delays = triangle.counts.shape
    length = triangle.last + 1
    out = np.zeros((n_loc, n_delays, length))
    for k in range(n_delays):
        out[:, k, k:] = triangle.counts[:, : length - k, k]
    return out.reshape(n_loc * n_delays, length)


@cache
def _toto(device: str) -> Any:
    from huggingface_hub import snapshot_download
    from toto.inference.forecaster import TotoForecaster
    from toto.model.toto import Toto

    # Toto's `from_pretrained` passes arguments huggingface_hub 1.0 dropped,
    # so the checkpoint is downloaded and loaded from its folder.
    path = snapshot_download(TOTO_MODEL, revision=TOTO_REVISION)
    toto = Toto.load_from_checkpoint(path, map_location=device)
    return TotoForecaster(toto.model)


def toto_sampler(
    series: np.ndarray, horizon: int, n_samples: int, seed: int
) -> np.ndarray:
    """Draw joint trajectories of every variate from Toto 1.0.

    Runs on the GPU when torch sees one.

    Parameters
    ----------
    series : numpy.ndarray
        Context, `[V, S]`, every variate in one group.
    horizon : int
        Weeks to draw.
    n_samples : int
        Trajectories.
    seed : int
        Seed for torch's generator.

    Returns
    -------
    numpy.ndarray
        Draws, `[n_samples, V, horizon]`.

    Raises
    ------
    ModelUnavailableError
        If Toto is not installed.
    """
    try:
        import torch
        from toto.data.util.dataset import MaskedTimeseries
    except ImportError as err:
        raise ModelUnavailableError(
            "fm_delay needs Toto: uv sync --extra toto."
        ) from err
    device = "cuda" if torch.cuda.is_available() else "cpu"
    forecaster = _toto(device)
    x = torch.as_tensor(series, dtype=torch.float32, device=device)
    inputs = MaskedTimeseries(
        series=x,
        padding_mask=torch.ones_like(x, dtype=torch.bool),
        id_mask=torch.zeros_like(x),
        timestamp_seconds=torch.zeros_like(x, dtype=torch.int64),
        time_interval_seconds=torch.full(
            (x.shape[0],), 7 * 86400, device=device
        ),
    )
    torch.manual_seed(seed)
    out = forecaster.forecast(
        inputs,
        prediction_length=horizon,
        num_samples=n_samples,
        samples_per_batch=n_samples,
    )
    # [1, V, horizon, n] to [n, V, horizon].
    return np.moveaxis(out.samples[0].cpu().numpy(), -1, 0)


@dataclass
class FMDelay:
    """Toto 1.0 on the delay classes, read as a nowcast and forecast.

    Attributes
    ----------
    name : str
        Model name.
    n_delays : int
        Weekly delay classes $K$.
    sample : Sampler
        Draws the trajectories, Toto 1.0 by default.
    """

    name: str = "fm_delay"
    n_delays: int = N_DELAYS
    sample: Sampler = toto_sampler

    def forecast(self, task: ForecastTask) -> pd.DataFrame:
        """Return joint trajectories for a task.

        Parameters
        ----------
        task : ForecastTask
            The data and weeks to forecast. Its `delays` hold the line
            data of the vintage.

        Returns
        -------
        pandas.DataFrame
            Columns `location`, `target_end_date`, `sample`, `value`.

        Raises
        ------
        ModelUnavailableError
            If the task has no delay line data, or Toto is not installed.
        """
        if task.delays is None or task.delays.empty:
            raise ModelUnavailableError(
                f"{self.name} needs delay line data, and the task for "
                f"{task.vintage} has none."
            )
        tri = delay_triangle(
            task.delays,
            task.vintage,
            task.locations,
            max(task.target_end_dates),
            self.n_delays,
        )
        n_loc, n_weeks, n_k = tri.counts.shape
        horizon = n_weeks + n_k - 2 - tri.last
        seed = int(task.rng(self.name).integers(0, 2**31 - 1))
        draws = self.sample(
            report_series(tri), horizon, task.n_samples, seed
        ).reshape(task.n_samples, n_loc, n_k, horizon)
        draws = np.maximum(np.round(draws), 0.0)
        reported = (
            task.target.pivot_table(
                index="location",
                columns="target_end_date",
                values="total",
                aggfunc="sum",
            )
            .reindex(index=list(task.locations), columns=list(tri.weeks))
            .fillna(0.0)
            .to_numpy()
        )
        values = np.broadcast_to(
            reported, (task.n_samples, n_loc, n_weeks)
        ).copy()
        for k in range(n_k):
            for w in range(max(tri.last - k + 1, 0), n_weeks):
                new = draws[:, :, k, w + k - tri.last - 1]
                values[:, :, w] += np.maximum(new - tri.counts[:, w, k], 0.0)
        col = {w: i for i, w in enumerate(tri.weeks)}
        targets = list(task.target_end_dates)
        picked = values[:, :, [col[w] for w in targets]]  # [S, P, T]
        n = task.n_samples
        return pd.DataFrame(
            {
                "location": np.repeat(list(task.locations), n * len(targets)),
                "target_end_date": np.tile(targets, n * n_loc),
                "sample": np.tile(
                    np.repeat(np.arange(1, n + 1), len(targets)), n_loc
                ),
                "value": picked.transpose(1, 0, 2).reshape(-1),
            }
        )[list(SAMPLE_COLUMNS)]


register_model(
    "fm_delay",
    FMDelay,
    "Toto 1.0 zero-shot on the delay classes, with nowcast and forecast "
    "from the same trajectories.",
    fitted=True,
)
