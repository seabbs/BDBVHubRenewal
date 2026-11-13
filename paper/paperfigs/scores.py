r"""Score quantile and sample forecasts in the paper schema.

Scores follow scoringutils' defaults, and `tests/test_scoringutils.py`
checks them against scoringutils 2.2 on a fixed fixture.

Quantile forecasts (`metric` is `wis`):

- **WIS** (Bracher et al. 2021) from scoringrules' `interval_score` on the
  central intervals the levels make, plus the median:
  $\frac{1}{K + 1/2}\left(\frac{1}{2}|y - m|
  + \sum_k \frac{\alpha_k}{2} \text{IS}_{\alpha_k}\right)$.
  scoringrules' own `weighted_interval_score` (0.11) adds the median value
  instead of its absolute error, so it is not used.
- **Bias** as scoringutils' `bias_quantile`: below the median,
  $1 - 2\max\{\tau : q_\tau \le y\}$; above it,
  $1 - 2\min\{\tau : q_\tau \ge y\}$; zero at the median.

Sample forecasts (`metric` is `crps`):

- **CRPS** from scoringrules' `crps_ensemble` with the `nrg` estimator,
  $E|X - y| - E|X - X'|/2$ over the empirical distribution, which is
  scoringRules' `crps_sample` that scoringutils calls.
- **Bias** as scoringutils' `bias_sample`. For counts (integer samples and
  truth) it is $1 - (P(X \le y) + P(X \le y - 1))$, otherwise
  $1 - 2P(X \le y)$.

Both forms:

- **Log score**: the same score after `log1p` of the forecast and truth
  (scoringutils' `log_shift` with offset 1; Bosse et al. 2023).
- **Coverage** of the central 50% and 90% intervals, bounds included. A
  sample forecast's bounds are its linear (type 7) sample quantiles, as
  scoringutils makes them; a quantile forecast without the levels gets a
  missing coverage.

Positive bias means overprediction. `relative_scores` divides a model's
mean score by the study baseline's mean score on the forecasts both made.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import pandas as pd
import scoringrules as sr

from paperfigs.schema import FORECAST_KEYS

__all__ = [
    "SCORE_COLUMNS",
    "STUDY_BASELINES",
    "metric_of",
    "relative_scores",
    "score_forecasts",
    "summarise_scores",
]

STUDY_BASELINES: Final[dict[str, str]] = {
    "synthetic": "randomwalknowcast",
    "germany": "hub_baseline",
    "ebola": "BDBVHub-baseline",
}
"""Reference model for relative scores in each study."""

SCORE_COLUMNS: Final = (
    "score",
    "log_score",
    "bias",
    "coverage_50",
    "coverage_90",
)
"""Score columns written by `score_forecasts`, beside `metric`."""

_MATCH: Final = ("location", "origin_date", "target_date", "horizon")


# --------------------------------------------------------------------------
# Quantile forecasts
# --------------------------------------------------------------------------


def _wis(q: np.ndarray, levels: np.ndarray, y: np.ndarray) -> np.ndarray:
    lower = levels < 0.5
    is_median = np.isclose(levels, 0.5)
    lo = q[:, lower]
    hi = q[:, ~lower & ~is_median][:, ::-1]
    med = q[:, is_median][:, 0]
    alpha = 2 * levels[lower]
    if alpha.size == 0:
        return np.abs(y - med)
    interval = np.asarray(
        sr.interval_score(y, lo, hi, alpha, backend="numpy"), dtype=float
    ).reshape(lo.shape)
    total = (interval * alpha / 2).sum(axis=1) + 0.5 * np.abs(y - med)
    return total / (alpha.size + 0.5)


def _bias_quantile(
    q: np.ndarray, levels: np.ndarray, y: np.ndarray
) -> np.ndarray:
    med = q[:, np.isclose(levels, 0.5)][:, 0]
    below = np.where(q <= y[:, None], levels, 0.0).max(axis=1)
    above = np.where(q >= y[:, None], levels, 1.0).min(axis=1)
    out = np.where(y < med, 1 - 2 * below, 1 - 2 * above)
    return np.where(y == med, 0.0, out)


def _coverage_quantile(
    q: np.ndarray, levels: np.ndarray, y: np.ndarray, width: float
) -> np.ndarray:
    lo = np.isclose(levels, (1 - width) / 2)
    hi = np.isclose(levels, (1 + width) / 2)
    if not (lo.any() and hi.any()):
        return np.full(y.shape, np.nan)
    return ((q[:, lo][:, 0] <= y) & (y <= q[:, hi][:, 0])).astype(float)


def _score_quantile_block(block: pd.DataFrame) -> pd.DataFrame:
    """Score quantile forecasts that share one set of levels."""
    wide = block.pivot_table(
        index=[*FORECAST_KEYS, "truth"],
        columns="quantile_level",
        values="value",
    )
    levels = wide.columns.to_numpy(dtype=float)
    q = wide.to_numpy(dtype=float)
    out = wide.index.to_frame(index=False)
    y = out["truth"].to_numpy(dtype=float)
    out["metric"] = "wis"
    out["score"] = _wis(q, levels, y)
    out["log_score"] = _wis(np.log1p(q), levels, np.log1p(y))
    out["bias"] = _bias_quantile(q, levels, y)
    out["coverage_50"] = _coverage_quantile(q, levels, y, 0.5)
    out["coverage_90"] = _coverage_quantile(q, levels, y, 0.9)
    return out


# --------------------------------------------------------------------------
# Sample forecasts
# --------------------------------------------------------------------------


def _crps(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.asarray(
        sr.crps_ensemble(y, x, estimator="nrg", backend="numpy"), dtype=float
    ).reshape(y.shape)


def _bias_sample(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    below = (x < y[:, None]).mean(axis=1)
    equal = (x == y[:, None]).mean(axis=1)
    counts = np.all(x == np.round(x)) and np.all(y == np.round(y))
    if counts:
        # P(X <= y) + P(X <= y - 1) for integers.
        return 1 - (2 * below + equal)
    return 1 - 2 * (below + equal)


def _coverage_sample(x: np.ndarray, y: np.ndarray, width: float) -> np.ndarray:
    lo, hi = np.quantile(x, [(1 - width) / 2, (1 + width) / 2], axis=1)
    return ((lo <= y) & (y <= hi)).astype(float)


def _score_sample_block(block: pd.DataFrame) -> pd.DataFrame:
    """Score sample forecasts that share one number of samples."""
    keys = list(FORECAST_KEYS)
    ordered = block.sort_values([*keys, "sample_id"])
    first = ordered.drop_duplicates(keys)
    out = first.loc[:, [*keys, "truth"]].reset_index(drop=True)
    x = ordered["value"].to_numpy(dtype=float).reshape(len(out), -1)
    y = out["truth"].to_numpy(dtype=float)
    out["metric"] = "crps"
    out["score"] = _crps(x, y)
    out["log_score"] = _crps(np.log1p(x), np.log1p(y))
    out["bias"] = _bias_sample(x, y)
    out["coverage_50"] = _coverage_sample(x, y, 0.5)
    out["coverage_90"] = _coverage_sample(x, y, 0.9)
    return out


# --------------------------------------------------------------------------
# Scoring and summaries
# --------------------------------------------------------------------------


def _blocks(df: pd.DataFrame, column: str) -> list[pd.DataFrame]:
    """Split forecasts into blocks with the same levels or sample count."""
    keys = list(FORECAST_KEYS)
    v = df[column].astype(float)
    g = df.assign(_v=v, _v2=v**2).groupby(keys, sort=False)
    sig = (
        g["_v"].transform("size").astype(str)
        + "|"
        + g["_v"].transform("sum").round(6).astype(str)
        + "|"
        + g["_v2"].transform("sum").round(6).astype(str)
    )
    return [b for _, b in df.groupby(sig, sort=False)]


def score_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Score each forecast with a known truth.

    Quantile forecasts get the WIS and sample forecasts the CRPS.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Forecasts in the schema of `paper/schema.md`.

    Returns
    -------
    pandas.DataFrame
        One row per forecast: the key columns, `truth`, `metric` (`wis`
        or `crps`) and `SCORE_COLUMNS`. Forecasts with missing truth are
        dropped.
    """
    keys = list(FORECAST_KEYS)
    columns = [*keys, "truth", "metric", *SCORE_COLUMNS]
    df = forecasts.loc[forecasts["truth"].notna()]
    is_q = df["quantile_level"].notna()
    parts = [
        _score_quantile_block(b) for b in _blocks(df[is_q], "quantile_level")
    ]
    parts += [_score_sample_block(b) for b in _blocks(df[~is_q], "sample_id")]
    if not parts:
        return pd.DataFrame(columns=columns)
    out = pd.concat(parts, ignore_index=True).loc[:, columns]
    return out.sort_values(keys, ignore_index=True)


def summarise_scores(
    scores: pd.DataFrame, by: Sequence[str] = ()
) -> pd.DataFrame:
    """Return the mean of each score by study, model and `by`.

    Parameters
    ----------
    scores : pandas.DataFrame
        Output of `score_forecasts`.
    by : sequence of str
        Further columns to group by, such as `horizon`.

    Returns
    -------
    pandas.DataFrame
        Mean scores, the `metric` and the number of forecasts, `n`.
    """
    group = ["study", "model", "metric", *by]
    g = scores.groupby(group, observed=True)
    out = g[list(SCORE_COLUMNS)].mean()
    out["n"] = g.size()
    return out.reset_index()


def metric_of(scores: pd.DataFrame) -> str:
    """Return the one metric in `scores`, `wis` or `crps`.

    Raises
    ------
    ValueError
        If the scores mix metrics.
    """
    metrics = scores["metric"].unique()
    if len(metrics) != 1:
        raise ValueError(f"Scores mix metrics: {', '.join(metrics)}.")
    return str(metrics[0])


def relative_scores(
    scores: pd.DataFrame,
    by: Sequence[str] = (),
    baseline: str | None = None,
) -> pd.DataFrame:
    """Divide each model's mean score by the baseline's on matched rows.

    Parameters
    ----------
    scores : pandas.DataFrame
        Output of `score_forecasts`.
    by : sequence of str
        Columns to group by beyond study and model, such as `location`.
    baseline : str, optional
        The reference model. Defaults to `STUDY_BASELINES` for each study.

    Returns
    -------
    pandas.DataFrame
        `study`, `model`, `by`, `rel_score`, `rel_log_score` and `n`, the
        number of matched forecasts. The baseline scores 1.

    Raises
    ------
    ValueError
        If a study's baseline has no scores.
    """
    parts = []
    for study, sc in scores.groupby("study", sort=False):
        ref = baseline or STUDY_BASELINES[str(study)]
        parts.append(_relative_one(sc, by, ref))
    return pd.concat(parts, ignore_index=True)


def _relative_one(
    scores: pd.DataFrame, by: Sequence[str], baseline: str
) -> pd.DataFrame:
    base = scores.loc[scores["model"] == baseline]
    if base.empty:
        raise ValueError(f"No scores for the baseline {baseline}.")
    match = ["study", *_MATCH]
    base = base.loc[:, [*match, "score", "log_score"]].rename(
        columns={"score": "base", "log_score": "base_log"}
    )
    paired = scores.merge(base, on=match, how="inner")
    group = ["study", "model", *by]
    g = paired.groupby(group, observed=True)
    sums = g[["score", "log_score", "base", "base_log"]].sum()
    out = pd.DataFrame(
        {
            "rel_score": sums["score"] / sums["base"],
            "rel_log_score": sums["log_score"] / sums["base_log"],
            "n": g.size(),
        }
    )
    return out.reset_index()
