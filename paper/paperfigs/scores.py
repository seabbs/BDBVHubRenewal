r"""Score quantile forecasts in the paper schema.

- **WIS** from scoringrules' `interval_score` on the central
  intervals the forecast's levels make, plus the median (Bracher et al.
  2021). It equals the summed pinball loss over $K + 1/2$.
- **Log WIS**: the same after `log1p` of quantiles and truth, so it scores
  relative error (Bosse et al. 2023).
- **Bias** as in scoringutils: below the median,
  $1 - 2\max\{\tau : q_\tau \le y\}$; above it,
  $1 - 2\min\{\tau : q_\tau \ge y\}$. Positive means overprediction.
- **Coverage** of the central 50% and 90% intervals, missing when the
  forecast lacks the levels.

`relative_wis` divides a model's mean WIS by the baseline's mean WIS on
the forecasts both made.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import pandas as pd
import scoringrules as sr

from paperfigs.schema import FORECAST_KEYS

__all__ = [
    "BASELINE",
    "SCORE_COLUMNS",
    "relative_wis",
    "score_forecasts",
    "summarise_scores",
]

BASELINE: Final = "randomwalknowcast"
"""Default reference model for relative scores."""

SCORE_COLUMNS: Final = ("wis", "log_wis", "bias", "coverage_50", "coverage_90")
"""Score columns written by `score_forecasts`."""

_MATCH: Final = ("location", "origin_date", "target_date", "horizon")


def _wis(q: np.ndarray, levels: np.ndarray, y: np.ndarray) -> np.ndarray:
    lower = levels < 0.5
    lo = q[:, lower]
    hi = q[:, ~lower & ~np.isclose(levels, 0.5)][:, ::-1]
    med = q[:, np.isclose(levels, 0.5)][:, 0]
    alpha = 2 * levels[lower]
    if alpha.size == 0:
        return np.abs(y - med)
    # scoringrules' `weighted_interval_score` adds the median itself, not
    # its absolute error, so the WIS is built from its interval scores.
    interval = np.asarray(
        sr.interval_score(y, lo, hi, alpha, backend="numpy"),
        dtype=float,
    ).reshape(lo.shape)
    total = (interval * alpha / 2).sum(axis=1) + 0.5 * np.abs(y - med)
    return total / (alpha.size + 0.5)


def _bias(q: np.ndarray, levels: np.ndarray, y: np.ndarray) -> np.ndarray:
    med = q[:, np.isclose(levels, 0.5)][:, 0]
    below = np.where(q <= y[:, None], levels, 0.0).max(axis=1)
    above = np.where(q >= y[:, None], levels, 1.0).min(axis=1)
    out = np.where(y < med, 1 - 2 * below, 1 - 2 * above)
    return np.where(y == med, 0.0, out)


def _coverage(
    q: np.ndarray, levels: np.ndarray, y: np.ndarray, width: float
) -> np.ndarray:
    lo_l, hi_l = (1 - width) / 2, (1 + width) / 2
    lo = np.isclose(levels, lo_l)
    hi = np.isclose(levels, hi_l)
    if not (lo.any() and hi.any()):
        return np.full(y.shape, np.nan)
    inside = (q[:, lo][:, 0] <= y) & (y <= q[:, hi][:, 0])
    return inside.astype(float)


def _score_block(block: pd.DataFrame) -> pd.DataFrame:
    """Score forecasts that share one set of quantile levels."""
    keys = list(FORECAST_KEYS)
    wide = block.pivot_table(
        index=[*keys, "truth"], columns="quantile_level", values="value"
    )
    levels = wide.columns.to_numpy(dtype=float)
    q = wide.to_numpy(dtype=float)
    idx = wide.index.to_frame(index=False)
    y = idx["truth"].to_numpy(dtype=float)
    idx["wis"] = _wis(q, levels, y)
    idx["log_wis"] = _wis(np.log1p(q), levels, np.log1p(y))
    idx["bias"] = _bias(q, levels, y)
    idx["coverage_50"] = _coverage(q, levels, y, 0.5)
    idx["coverage_90"] = _coverage(q, levels, y, 0.9)
    return idx


def score_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Score each forecast with a known truth.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Forecasts in the schema of `paper/schema.md`.

    Returns
    -------
    pandas.DataFrame
        One row per forecast: the key columns, `truth` and
        `SCORE_COLUMNS`. Forecasts with missing truth are dropped.
    """
    keys = list(FORECAST_KEYS)
    df = forecasts.loc[forecasts["truth"].notna()]
    if df.empty:
        return pd.DataFrame(columns=[*keys, "truth", *SCORE_COLUMNS])
    # Forecasts are scored in blocks that share their quantile levels, so
    # a model with fewer levels is still scored on what it gave.
    sig = df.groupby(keys, sort=False)["quantile_level"].transform(
        lambda s: ",".join(f"{v:.4f}" for v in sorted(s))
    )
    parts = [_score_block(b) for _, b in df.groupby(sig, sort=False)]
    out = pd.concat(parts, ignore_index=True)
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
        Mean scores and the number of forecasts, `n`.
    """
    group = ["study", "model", *by]
    g = scores.groupby(group, observed=True)
    out = g[list(SCORE_COLUMNS)].mean()
    out["n"] = g.size()
    return out.reset_index()


def relative_wis(
    scores: pd.DataFrame,
    by: Sequence[str] = (),
    baseline: str = BASELINE,
) -> pd.DataFrame:
    """Divide each model's mean WIS by the baseline's on matched rows.

    Parameters
    ----------
    scores : pandas.DataFrame
        Output of `score_forecasts`.
    by : sequence of str
        Columns to group by beyond study and model, such as `location`.
    baseline : str
        The reference model.

    Returns
    -------
    pandas.DataFrame
        `study`, `model`, `by`, `rel_wis`, `rel_log_wis` and `n`, the
        number of matched forecasts. The baseline scores 1.

    Raises
    ------
    ValueError
        If the baseline has no scores.
    """
    base = scores.loc[scores["model"] == baseline]
    if base.empty:
        raise ValueError(f"No scores for the baseline {baseline}.")
    match = ["study", *_MATCH]
    base = base.loc[:, [*match, "wis", "log_wis"]].rename(
        columns={"wis": "base_wis", "log_wis": "base_log_wis"}
    )
    paired = scores.merge(base, on=match, how="inner")
    group = ["study", "model", *by]
    sums = paired.groupby(group, observed=True)[
        ["wis", "log_wis", "base_wis", "base_log_wis"]
    ].sum()
    out = pd.DataFrame(
        {
            "rel_wis": sums["wis"] / sums["base_wis"],
            "rel_log_wis": sums["log_wis"] / sums["base_log_wis"],
            "n": paired.groupby(group, observed=True).size(),
        }
    )
    return out.reset_index()
