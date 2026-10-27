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
- **Log WIS**: WIS of $\log(1 + y)$ against $\log(1 + x)$ for the samples.
  It scores relative error, so large provinces do not dominate.
- **CRPS** of the empirical sample distribution, for reference.
- **Coverage** of the central 50% and 90% sample intervals.
- **Bias** $1 - (P(X \le y) + P(X \le y - 1))$ for counts, in $[-1, 1]$.
  Positive means overprediction.

`relative_skill` divides a model's mean score by the baseline's mean score
over the forecasts both made.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "QUANTILE_LEVELS",
    "SCORE_COLUMNS",
    "bias_sample",
    "coverage_sample",
    "crps_sample",
    "relative_skill",
    "score_forecasts",
    "score_sample",
    "summarise_scores",
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
    q = np.quantile(x, tau)
    k = (tau.size - 1) / 2
    lower = tau < 0.5
    upper = tau > 0.5
    # Pair each lower level with its upper level.
    lo, hi = q[lower], q[upper][::-1]
    alpha = 2 * tau[lower]
    med = float(q[np.isclose(tau, 0.5)][0])
    disp = np.sum(alpha / 2 * (hi - lo))
    over = np.sum(np.maximum(lo - obs, 0)) + max(med - obs, 0) / 2
    under = np.sum(np.maximum(obs - hi, 0)) + max(obs - med, 0) / 2
    norm = k + 0.5
    return {
        "wis": float((disp + over + under) / norm),
        "dispersion": float(disp / norm),
        "overprediction": float(over / norm),
        "underprediction": float(under / norm),
    }


def crps_sample(obs: float, samples: ArrayLike) -> float:
    """Return the CRPS of the empirical distribution of the samples.

    Uses $E|X - y| - E|X - X'|/2$ with the sorted closed form for the
    second term.

    Parameters
    ----------
    obs : float
        The observed value.
    samples : array-like
        Sample values.

    Returns
    -------
    float
        The CRPS.
    """
    x = np.sort(_as_samples(samples))
    n = x.size
    mae = np.mean(np.abs(x - obs))
    weights = 2 * np.arange(1, n + 1) - n - 1
    spread = np.sum(weights * x) / n**2
    return float(mae - spread)


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
    out["ae_median"] = float(abs(np.median(x) - obs))
    out["coverage_50"] = float(coverage_sample(obs, x, 0.5))
    out["coverage_90"] = float(coverage_sample(obs, x, 0.9))
    out["bias"] = bias_sample(obs, x)
    return out


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
    """
    keys = list(keys)
    on = [c for c in truth.columns if c != "truth"]
    merged = forecasts.merge(truth, on=on, how="inner")
    rows: list[dict[str, object]] = []
    for key, grp in merged.groupby([*keys, "horizon"], sort=True):
        obs = float(grp["truth"].iloc[0])
        scores = score_sample(obs, grp["value"].to_numpy())
        ids: dict[str, object] = dict(zip([*keys, "horizon"], key, strict=True))
        rows.append(
            {
                **ids,
                "truth": obs,
                "n_samples": len(grp),
                **scores,
            }
        )
    return pd.DataFrame(rows)


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
    baseline: str = "persistence",
    by: Sequence[str] = (),
    metrics: Sequence[str] = ("wis", "log_wis"),
    match_on: Sequence[str] = (
        "kind",
        "vintage",
        "reference_date",
        "location",
        "target_end_date",
    ),
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

    Returns
    -------
    pandas.DataFrame
        Columns `model`, the `by` columns, `n` and `relative_<metric>`
        for each metric. The baseline has 1.
    """
    match = list(dict.fromkeys([*match_on, *by]))
    base = scores.loc[scores["model"] == baseline, [*match, *metrics]]
    merged = scores.merge(base, on=match, suffixes=("", "_base"))
    group = ["model", *by]
    grp = merged.groupby(group)
    out = grp.size().rename("n").to_frame()
    for m in metrics:
        num = grp[m].mean()
        den = grp[f"{m}_base"].mean()
        out[f"relative_{m}"] = num / den.where(den > 0)
    return out.reset_index()
