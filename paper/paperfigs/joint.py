"""Joint scores of sample trajectories across a forecast's target dates.

A sample forecast's draws with one `sample_id` form a trajectory over its
target dates. `joint_scores` scores each forecast (model, location,
stratum and origin) over its targets with known truth, at least two, with
the energy score on log(x + 1) counts, in its fair form (Ferro 2014),
whose expectation does not depend on the number of draws (scoringrules).
Unlike the per-target scores, it sees whether the draws are paired into
trajectories.
`relative_joint_scores` divides each model's mean by the study
baseline's on the forecasts both made.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
import scoringrules as sr

from paperfigs.scores import STUDY_BASELINES

__all__ = ["JOINT", "joint_scores", "relative_joint_scores"]

JOINT: Final = ("energy_log",)
"""The joint score, the energy score on log(x + 1) counts."""

_KEYS: Final = ["study", "model", "location", "stratum", "origin_date"]


def joint_scores(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Score each sample forecast's trajectories jointly.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Sample forecasts in the schema, with `truth` attached.

    Returns
    -------
    pandas.DataFrame
        `_KEYS`, `n_targets` and `JOINT`, one row per forecast with at
        least two targets with known truth.
    """
    fc = forecasts.loc[
        forecasts["sample_id"].notna() & forecasts["truth"].notna()
    ]
    if "stratum" not in fc.columns:
        fc = fc.assign(stratum="all")
    rows = []
    for key, one in fc.groupby(_KEYS, sort=True):
        if one["target_date"].nunique() < 2:
            continue
        wide = one.pivot_table(
            index="sample_id", columns="target_date", values="value"
        ).dropna()
        truth = one.groupby("target_date")["truth"].first().loc[wide.columns]
        x = np.log1p(wide.to_numpy(float))
        y = np.log1p(truth.to_numpy(float))
        rows.append(
            (
                *key,
                x.shape[1],
                float(sr.es_ensemble(y, x, estimator="fair")),
            )
        )
    return pd.DataFrame(rows, columns=[*_KEYS, "n_targets", *JOINT])


def relative_joint_scores(
    scores: pd.DataFrame,
    by: list[str] | None = None,
    baseline: str | None = None,
) -> pd.DataFrame:
    """Return each model's mean joint scores and their ratio to the baseline.

    Parameters
    ----------
    scores : pandas.DataFrame
        `joint_scores` output for one study.
    by : list of str, optional
        Columns to group by beyond the model.
    baseline : str, optional
        The reference model. Defaults to the study's `STUDY_BASELINES`.

    Returns
    -------
    pandas.DataFrame
        `model`, `by`, `n`, `JOINT` and `rel_<score>` for each, on the
        forecasts each model shares with the baseline.
    """
    by = by or []
    study = str(scores["study"].iloc[0])
    base = scores.loc[scores["model"] == (baseline or STUDY_BASELINES[study])]
    keys = [k for k in _KEYS if k != "model"]
    paired = scores.merge(
        base[[*keys, *JOINT]], on=keys, suffixes=("", "_base")
    )
    g = paired.groupby(["model", *by])
    out = g[list(JOINT)].mean()
    for s in JOINT:
        out[f"rel_{s}"] = g[s].sum() / g[f"{s}_base"].sum()
    out.insert(0, "n", g.size())
    return out.reset_index()
