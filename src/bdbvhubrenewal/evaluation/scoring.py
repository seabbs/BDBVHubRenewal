# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Score sample forecasts.

Scores follow scoringutils conventions, as in
epiforecasts/BVDOutbreakSize (`src/scoring.jl`).

- **WIS** from the sample quantiles at `QUANTILE_LEVELS` (the 23 levels
  the US hubs use). With $K$ central intervals plus the median, written as
  quantile levels $\tau$ with sample quantiles $q_\tau$,
  $$\text{WIS}(F, y) = \frac{1}{K + 1/2}\sum_\tau
  \left(\mathbb{1}\{y < q_\tau\} - \tau\right)(q_\tau - y),$$
  which equals the usual interval form.
  It splits into dispersion, overprediction and underprediction.
  `wis_quantile` gives the same for forecasts made as quantiles, and
  `score_quantiles` scores those by WIS, the absolute error of the median
  and the coverage of each central interval.
- **Log WIS**: WIS of $\log(1 + y)$ against $\log(1 + x)$ for the samples.
- **CRPS** in the fair form (Ferro 2014) over $m$ samples,
  $$\frac{1}{m}\sum_i |x_i - y| -
  \frac{1}{2m(m - 1)}\sum_{i \ne j} |x_i - x_j|,$$
  scoringrules' `crps_ensemble` with `estimator="fair"`, as the paper
  scores. Its expectation does not depend on $m$, so forecasts with
  different numbers of draws compare. `estimator="nrg"` gives the CRPS of
  the empirical distribution. The BDBV pages score by CRPS, as the hub
  takes samples; the German benchmark scores quantiles by WIS.
- **Log CRPS**: CRPS of $\log(1 + y)$ against $\log(1 + x)$.
  It scores relative error, so large provinces do not dominate.
- **Coverage** of the central 50% and 90% sample intervals.
- **Bias** $1 - (P(X \le y) + P(X \le y - 1))$ for counts, in $[-1, 1]$.
  Positive means overprediction.
- **Energy score** of a vector forecast, in the fair form over $m$ samples,
  $$\text{ES}(F, \mathbf{y}) = \frac{1}{m}\sum_i \|\mathbf{X}_i - \mathbf{y}\|
  - \frac{1}{2m(m-1)}\sum_{i \ne j} \|\mathbf{X}_i - \mathbf{X}_j\|,$$
  scoringrules' `es_ensemble` with `estimator="fair"`.
  The vector is every place at one scale by every scored week, for one
  origin, so the score rewards getting the joint pattern right.
  The sample index is the joint draw.
- **Log energy score**: the energy score of $\log(1 + \mathbf{y})$ against
  $\log(1 + \mathbf{X})$.

`relative_skill` divides a model's mean score by the baseline's mean score
over the forecasts both made.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import pandas as pd
import scoringrules as sr  # type: ignore[import-untyped]
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import pdist

__all__ = [
    "QUANTILE_LEVELS",
    "SCORE_COLUMNS",
    "bias_sample",
    "coverage_sample",
    "crps_sample",
    "energy_sample",
    "energy_scores",
    "relative_skill",
    "scale_of",
    "score_cells",
    "score_forecasts",
    "score_quantiles",
    "score_sample",
    "summarise_scores",
    "variogram_sample",
    "wis_quantile",
    "wis_sample",
]

QUANTILE_LEVELS: Final = np.array(
    [0.01, 0.025, *np.round(np.arange(0.05, 0.951, 0.05), 3), 0.975, 0.99]
)
"""Quantile levels used for WIS: 11 central intervals plus the median."""

SCORE_COLUMNS: Final = (
    "wis",
    "dispersion",
    "overprediction",
    "underprediction",
    "log_wis",
    "crps",
    "log_crps",
    "ae_median",
    "coverage_50",
    "coverage_90",
    "bias",
)
"""Score columns written by `score_forecasts`."""


def _as_samples(samples: ArrayLike) -> NDArray[np.float64]:
    x = np.asarray(samples, dtype=np.float64).ravel()
    if x.size == 0:
        raise ValueError("Need at least one sample.")
    return x


def wis_sample(
    obs: float,
    samples: ArrayLike,
    quantile_levels: NDArray[np.float64] = QUANTILE_LEVELS,
) -> dict[str, float]:
    """Return the WIS of a sample forecast and its components.

    Parameters
    ----------
    obs : float
        The observed value.
    samples : array-like
        Sample values.
    quantile_levels : numpy.ndarray
        Symmetric quantile levels including 0.5.

    Returns
    -------
    dict
        `wis`, `dispersion`, `overprediction` and `underprediction`.
        The components sum to `wis`.
    """
    x = _as_samples(samples)
    tau = np.asarray(quantile_levels, dtype=np.float64)
    out = wis_quantile(np.array([obs]), np.quantile(x, tau)[None, :], tau)
    return {k: float(v[0]) for k, v in out.items()}


def wis_quantile(
    obs: ArrayLike, quantiles: ArrayLike, quantile_levels: ArrayLike
) -> dict[str, NDArray[np.float64]]:
    """Return the WIS of quantile forecasts and its components.

    Parameters
    ----------
    obs : array-like
        The observed values, `[N]`.
    quantiles : array-like
        Each forecast's quantiles at `quantile_levels`, `[N, K]`.
    quantile_levels : array-like
        Symmetric quantile levels including 0.5, ascending, `[K]`.

    Returns
    -------
    dict
        `wis`, `dispersion`, `overprediction` and `underprediction`, each
        `[N]`. The components sum to `wis`.
    """
    y = np.asarray(obs, dtype=np.float64)[:, None]
    q = np.asarray(quantiles, dtype=np.float64)
    tau = np.asarray(quantile_levels, dtype=np.float64)
    lower = tau < 0.5
    # Pair each lower level with its upper level.
    lo, hi = q[:, lower], q[:, tau > 0.5][:, ::-1]
    alpha = 2 * tau[lower]
    med = q[:, np.isclose(tau, 0.5)]
    disp = np.sum(alpha / 2 * (hi - lo), axis=1)
    over = np.sum(np.maximum(lo - y, 0), axis=1)
    over += np.maximum(med - y, 0)[:, 0] / 2
    under = np.sum(np.maximum(y - hi, 0), axis=1)
    under += np.maximum(y - med, 0)[:, 0] / 2
    norm = lower.sum() + 0.5
    return {
        "wis": (disp + over + under) / norm,
        "dispersion": disp / norm,
        "overprediction": over / norm,
        "underprediction": under / norm,
    }


def crps_sample(
    obs: float, samples: ArrayLike, estimator: str = "fair"
) -> float:
    r"""Return the CRPS of a sample forecast.

    $\frac{1}{m}\sum_i |x_i - y| - \frac{1}{2c}\sum_{i, j} |x_i - x_j|$,
    with the pair sum in sorted closed form. The fair form (Ferro 2014)
    has $c = m(m - 1)$ and an expectation that does not depend on $m$;
    `nrg`, the CRPS of the empirical distribution, has $c = m^2$.

    Parameters
    ----------
    obs : float
        The observed value.
    samples : array-like
        Sample values.
    estimator : str
        `"fair"`, as scoringrules' `crps_ensemble(estimator="fair")`, or
        `"nrg"`.

    Returns
    -------
    float
        The CRPS. With one sample the fair form has no spread term.
    """
    if estimator not in ("fair", "nrg"):
        raise ValueError(
            f"estimator must be 'fair' or 'nrg', not {estimator!r}."
        )
    x = np.sort(_as_samples(samples))
    n = x.size
    mae = np.mean(np.abs(x - obs))
    weights = 2 * np.arange(1, n + 1) - n - 1
    # Half the sum of |x_i - x_j| over all pairs.
    half = np.sum(weights * x)
    if estimator == "nrg":
        return float(mae - half / n**2)
    return float(mae - (half / (n * (n - 1)) if n > 1 else 0.0))


def energy_sample(obs: ArrayLike, samples: ArrayLike) -> float:
    r"""Return the fair energy score of a sample forecast of a vector.

    Parameters
    ----------
    obs : array-like
        The observed vector, length $d$.
    samples : array-like
        Joint samples, shape `[m, d]` with $m \ge 2$.

    Returns
    -------
    float
        The energy score, as scoringrules' `es_ensemble` with
        `estimator="fair"`. In one dimension it is the fair CRPS.
    """
    y = np.asarray(obs, dtype=np.float64)
    x = np.asarray(samples, dtype=np.float64)
    m = x.shape[0]
    fit = np.mean(np.linalg.norm(x - y, axis=1))
    spread = np.sum(pdist(x)) / (m * (m - 1))
    return float(fit - spread)


def variogram_sample(
    obs: ArrayLike, samples: ArrayLike, p: float = 0.5
) -> float:
    r"""Return the variogram score of a sample forecast of a vector.

    $\sum_{i, j} \left(\frac{1}{m} \sum_k |x_{k,i} - x_{k,j}|^p
    - |y_i - y_j|^p\right)^2$ over every ordered pair of components, as
    Scheuerer and Hamill (2015) define it, with equal weights.

    Parameters
    ----------
    obs : array-like
        The observed vector, length $d$.
    samples : array-like
        Joint samples, shape `[m, d]`.
    p : float
        The order.

    Returns
    -------
    float
        The variogram score.
    """
    y = np.asarray(obs, dtype=np.float64)
    x = np.asarray(samples, dtype=np.float64)
    fcst = np.mean(np.abs(x[:, :, None] - x[:, None, :]) ** p, axis=0)
    seen = np.abs(y[:, None] - y[None, :]) ** p
    return float(np.sum((fcst - seen) ** 2))


def scale_of(location: pd.Series) -> pd.Series:
    """Return `national`, `provinces` or `zones` for each location code.

    Parameters
    ----------
    location : pandas.Series
        Location codes, such as `CD`, `CD-IT` and `CD-IT-10`.

    Returns
    -------
    pandas.Series
        The scale of each code.
    """
    depth = location.str.count("-")
    return depth.map({0: "national", 1: "provinces"}).fillna("zones")


def energy_scores(
    samples: pd.DataFrame,
    truth: pd.DataFrame,
    unit: Sequence[str],
) -> pd.DataFrame:
    """Score every model's joint forecast of each scale with the energy score.

    Within one `unit` and scale, the vector is the place-week cells that
    every model in the unit forecasts and the truth has. Each model's
    samples are its rows with the same `sample` index.

    Parameters
    ----------
    samples : pandas.DataFrame
        The `unit` columns, `model`, `location`, `target_end_date`,
        `sample` and `value`.
    truth : pandas.DataFrame
        Columns `location`, `target_end_date` and `truth`, plus any
        columns it should be matched on.
    unit : sequence of str
        Columns that identify one forecast of every scale, such as an
        origin and the release scored against.

    Returns
    -------
    pandas.DataFrame
        One row per unit, `scale` and `model`: `n_cells`, `n_samples`,
        `energy` and `log_energy`.
    """
    unit = list(unit)
    cell = ["location", "target_end_date"]
    on = [c for c in truth.columns if c != "truth"]
    rows = samples.merge(truth, on=on)
    rows["scale"] = scale_of(rows["location"])
    out: list[dict[str, object]] = []
    for key, grp in rows.groupby([*unit, "scale"], sort=True):
        have = grp[["model", *cell]].drop_duplicates()
        count = have.groupby(cell).size()
        shared = count[count == have["model"].nunique()].index
        grp = grp.set_index(cell).loc[shared].reset_index()
        y = grp.drop_duplicates(cell).set_index(cell)["truth"]
        for model, own in grp.groupby("model"):
            x = own.pivot(index="sample", columns=cell, values="value")
            obs = y.reindex(x.columns).to_numpy()
            out.append(
                {
                    **dict(zip([*unit, "scale"], key, strict=True)),
                    "model": model,
                    "n_cells": x.shape[1],
                    "n_samples": x.shape[0],
                    "energy": energy_sample(obs, x),
                    "log_energy": energy_sample(np.log1p(obs), np.log1p(x)),
                }
            )
    return pd.DataFrame(out)


def coverage_sample(obs: float, samples: ArrayLike, level: float) -> bool:
    """Whether the observation is inside the central sample interval.

    Parameters
    ----------
    obs : float
        The observed value.
    samples : array-like
        Sample values.
    level : float
        Interval width, such as 0.5 or 0.9.

    Returns
    -------
    bool
        True if the observation is inside, bounds included.
    """
    x = _as_samples(samples)
    lo, hi = np.quantile(x, [(1 - level) / 2, (1 + level) / 2])
    return bool(lo <= obs <= hi)


def bias_sample(obs: float, samples: ArrayLike) -> float:
    r"""Return the bias of a sample forecast of a count.

    Parameters
    ----------
    obs : float
        The observed count.
    samples : array-like
        Sample values.

    Returns
    -------
    float
        $1 - (P(X \le y) + P(X \le y - 1))$, in $[-1, 1]$. Positive
        means the forecast is too high.
    """
    x = _as_samples(samples)
    n_lt = np.sum(x < obs)
    n_eq = np.sum(x == obs)
    return float(1 - (2 * n_lt + n_eq) / x.size)


def score_sample(obs: float, samples: ArrayLike) -> dict[str, float]:
    """Return every score in `SCORE_COLUMNS` for one forecast.

    Parameters
    ----------
    obs : float
        The observed value.
    samples : array-like
        Sample values.

    Returns
    -------
    dict
        Scores keyed by `SCORE_COLUMNS`.
    """
    x = _as_samples(samples)
    out = wis_sample(obs, x)
    out["log_wis"] = wis_sample(float(np.log1p(obs)), np.log1p(x))["wis"]
    out["crps"] = crps_sample(obs, x)
    out["log_crps"] = crps_sample(float(np.log1p(obs)), np.log1p(x))
    out["ae_median"] = float(abs(np.median(x) - obs))
    out["coverage_50"] = float(coverage_sample(obs, x, 0.5))
    out["coverage_90"] = float(coverage_sample(obs, x, 0.9))
    out["bias"] = bias_sample(obs, x)
    return out


def score_cells(
    obs: ArrayLike,
    samples: ArrayLike,
    coverage: Sequence[int] = (50, 90),
) -> dict[str, NDArray[np.float64]]:
    """Return every score of many forecasts at once.

    The scores of `score_sample` for each row, computed on the whole
    array: WIS and log WIS from the sample quantiles, fair CRPS and log
    CRPS (scoringrules' `crps_ensemble`), the absolute error of the
    median, coverage and bias.

    Parameters
    ----------
    obs : array-like
        The observed values, `[N]`.
    samples : array-like
        Each forecast's draws, `[N, m]`.
    coverage : sequence of int
        Central interval widths in percent, each a `coverage_<p>` column.

    Returns
    -------
    dict
        `[N]` arrays keyed by `SCORE_COLUMNS` (with the `coverage`
        columns asked for).
    """
    y = np.asarray(obs, dtype=np.float64)
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] == 0 or x.shape[0] != y.size:
        raise ValueError(f"Need [N, m] samples for N={y.size}, not {x.shape}.")
    tau = QUANTILE_LEVELS
    out = wis_quantile(y, np.quantile(x, tau, axis=1).T, tau)
    ly, lx = np.log1p(y), np.log1p(x)
    out["log_wis"] = wis_quantile(ly, np.quantile(lx, tau, axis=1).T, tau)[
        "wis"
    ]
    out["crps"] = _crps(y, x)
    out["log_crps"] = _crps(ly, lx)
    out["ae_median"] = np.abs(np.median(x, axis=1) - y)
    for p in coverage:
        level = p / 100
        lo, hi = np.quantile(x, [(1 - level) / 2, (1 + level) / 2], axis=1)
        out[f"coverage_{p}"] = ((lo <= y) & (y <= hi)).astype(np.float64)
    n_lt = np.sum(x < y[:, None], axis=1)
    n_eq = np.sum(x == y[:, None], axis=1)
    out["bias"] = 1 - (2 * n_lt + n_eq) / x.shape[1]
    return out


def _crps(
    y: NDArray[np.float64], x: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Fair CRPS of each row; one draw has no spread term.

    scoringrules' `pwm` estimator equals its `fair` one, in sorted
    O(m log m) form rather than over every pair of draws.
    """
    if x.shape[1] == 1:
        return np.abs(x[:, 0] - y)
    return np.asarray(
        sr.crps_ensemble(y, x, m_axis=-1, estimator="pwm"), dtype=np.float64
    )


def score_forecasts(
    forecasts: pd.DataFrame,
    truth: pd.DataFrame,
    keys: Sequence[str] = (
        "model",
        "kind",
        "vintage",
        "reference_date",
        "location",
        "target_end_date",
    ),
) -> pd.DataFrame:
    """Score sample forecasts against observed values.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        One row per sample, with the `keys` columns, `horizon`, `sample`
        and `value`.
    truth : pandas.DataFrame
        Columns `location`, `target_end_date` and `truth`, plus any key
        columns it should be matched on (such as `kind` for simulated
        data).
    keys : sequence of str
        Columns that identify one forecast.

    Returns
    -------
    pandas.DataFrame
        One row per forecast with a truth value: the keys, `horizon`,
        `truth`, `n_samples` and `SCORE_COLUMNS`.

    Examples
    --------
    Score against the latest release, with `forecasts` from `run_forecasts`.

    >>> from bdbvhubrenewal.data import load_target, vintages
    >>> from bdbvhubrenewal.evaluation.scoring import (
    ...     score_forecasts, summarise_scores
    ... )
    >>> real = load_target().query("kind == 'real'")
    >>> latest = real.loc[real["vintage"] == vintages(real)[-1]]
    >>> truth = latest.rename(columns={"total": "truth"})[
    ...     ["location", "target_end_date", "truth"]
    ... ]
    >>> scores = score_forecasts(forecasts, truth)
    >>> summarise_scores(scores, by=["model"])
    """
    keys = [*keys, "horizon"]
    on = [c for c in truth.columns if c != "truth"]
    merged = forecasts.merge(truth, on=on, how="inner")
    if merged.empty:
        return pd.DataFrame(
            columns=[*keys, "truth", "n_samples", *SCORE_COLUMNS]
        )
    group = merged.groupby(keys, sort=True)
    merged["_cell"] = group.ngroup()
    cells = group.agg(truth=("truth", "first"), n_samples=("value", "size"))
    cells = cells.reset_index()
    merged = merged.sort_values("_cell", kind="stable")
    parts = []
    # One array per draw count: cells with m draws form an [N, m] array.
    for _, idx in cells.groupby("n_samples").groups.items():
        sub = cells.loc[idx]
        values = merged.loc[merged["_cell"].isin(sub.index), "value"]
        draws = values.to_numpy(dtype=np.float64).reshape(len(sub), -1)
        scores = score_cells(sub["truth"].to_numpy(), draws)
        parts.append(sub.assign(**{c: scores[c] for c in SCORE_COLUMNS}))
    out = pd.concat(parts).sort_index()
    out["truth"] = out["truth"].astype(float)
    return out.reset_index(drop=True)


def score_quantiles(
    forecasts: pd.DataFrame,
    truth: pd.DataFrame,
    keys: Sequence[str],
    quantile_levels: ArrayLike,
) -> pd.DataFrame:
    """Score quantile forecasts against observed values.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        One row per quantile, with the `keys` columns, `horizon`,
        `quantile` and `value`.
    truth : pandas.DataFrame
        Columns `location`, `target_end_date` and `truth`, plus any key
        columns it should be matched on.
    keys : sequence of str
        Columns that identify one forecast.
    quantile_levels : array-like
        Symmetric quantile levels including 0.5. Every forecast must give
        each of them.

    Returns
    -------
    pandas.DataFrame
        One row per forecast with a truth value: the keys, `horizon`,
        `truth`, the `wis_quantile` columns, `ae_median` and
        `coverage_<w>`, whether the truth is inside the central $w$%
        interval, bounds included, for each pair of levels.

    Raises
    ------
    ValueError
        If a forecast lacks one of `quantile_levels`.
    """
    tau = np.sort(np.asarray(quantile_levels, dtype=np.float64))
    on = [c for c in truth.columns if c != "truth"]
    rows = forecasts.merge(truth, on=on, how="inner")
    rows = rows.assign(quantile=rows["quantile"].round(6))
    wide = rows.pivot_table(
        index=[*keys, "horizon", "truth"],
        columns="quantile",
        values="value",
        aggfunc="first",
    ).reindex(columns=tau.round(6))
    if wide.isna().to_numpy().any():
        raise ValueError(
            f"Each forecast needs the quantile levels {tau.tolist()}."
        )
    q = wide.to_numpy()
    y = wide.index.get_level_values("truth").to_numpy(dtype=np.float64)
    out = pd.DataFrame(wis_quantile(y, q, tau), index=wide.index)
    out["ae_median"] = np.abs(q[:, np.isclose(tau, 0.5)][:, 0] - y)
    for i in np.flatnonzero(tau < 0.5):
        width = round(100 * (1 - 2 * tau[i]))
        inside = (q[:, i] <= y) & (y <= q[:, -1 - i])
        out[f"coverage_{width}"] = inside.astype(float)
    return out.reset_index()


def summarise_scores(
    scores: pd.DataFrame, by: Sequence[str] = ("model",)
) -> pd.DataFrame:
    """Average scores over forecasts within groups.

    Parameters
    ----------
    scores : pandas.DataFrame
        Output of `score_forecasts`.
    by : sequence of str
        Columns to group by.

    Returns
    -------
    pandas.DataFrame
        Group columns, `n` and the mean of each score.
    """
    grp = scores.groupby(list(by))
    out = grp[list(SCORE_COLUMNS)].mean()
    out.insert(0, "n", grp.size())
    return out.reset_index()


def relative_skill(
    scores: pd.DataFrame,
    baseline: str = "randomwalk",
    by: Sequence[str] = (),
    metrics: Sequence[str] = ("wis", "log_wis"),
    match_on: Sequence[str] = (
        "kind",
        "vintage",
        "reference_date",
        "location",
        "target_end_date",
    ),
    means: Sequence[str] = (),
) -> pd.DataFrame:
    """Return each model's mean score divided by the baseline's.

    Only forecasts made by both the model and the baseline are compared.
    Values below 1 mean the model beats the baseline.

    Parameters
    ----------
    scores : pandas.DataFrame
        Output of `score_forecasts`.
    baseline : str
        The baseline model name.
    by : sequence of str
        Extra columns to compare within, such as `("horizon",)`.
    metrics : sequence of str
        Scores to compare.
    match_on : sequence of str
        Columns that identify the same forecast across models.
    means : sequence of str
        Columns to average over the same forecasts, for the model and the
        baseline.

    Returns
    -------
    pandas.DataFrame
        Columns `model`, the `by` columns, `n`, each `means` column and
        its baseline mean `base_<column>`, and `relative_<metric>` for
        each metric: the model's mean over the baseline's, missing where
        the baseline's mean is 0. The baseline has 1.
    """
    match = list(dict.fromkeys([*match_on, *by]))
    cols = list(dict.fromkeys([*metrics, *means]))
    base = scores.loc[scores["model"] == baseline, [*match, *cols]]
    merged = scores.merge(base, on=match, suffixes=("", "_base"))
    group = ["model", *by]
    grp = merged.groupby(group)
    out = grp.size().rename("n").to_frame()
    for c in means:
        out[c] = grp[c].mean()
        out[f"base_{c}"] = grp[f"{c}_base"].mean()
    for m in metrics:
        num = grp[m].mean()
        den = grp[f"{m}_base"].mean()
        out[f"relative_{m}"] = num / den.where(den > 0)
    return out.reset_index()
