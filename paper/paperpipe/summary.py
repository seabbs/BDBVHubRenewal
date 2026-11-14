"""The numbers the paper quotes, as Quarto metadata.

`study_summary` reads each study's run notes and main-text score table and
returns one entry per study: where its data came from, its baseline, and
the best model with its relative scores. `write_summary` writes them as
YAML under `results`, so `main.qmd` and `si.qmd` quote them with
`{{< meta results.<study>.<field> >}}`. Where a study has a split, the
numbers are from its held-out half.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from paperfigs.periods import PERIODS
from paperfigs.scores import STUDY_BASELINES
from paperpipe.meta import DUMMY_LABEL

__all__ = ["study_summary", "write_summary"]


def study_summary(meta: Mapping[str, Any], scores: pd.DataFrame) -> dict:
    """Return the quoted numbers of one study.

    Parameters
    ----------
    meta : mapping
        The study's run notes (`paperpipe.meta`).
    scores : pandas.DataFrame
        The study's main-text score table (`paperpipe.outputs`).

    Returns
    -------
    dict
        `source`, `label` (empty for real data), `baseline`, `n_models`,
        `best_model`, `best_rel_score` and `best_rel_log_score`.
    """
    study = meta["study"]
    baseline = STUDY_BASELINES[study]
    if "period" in scores.columns:
        scores = scores.loc[scores["period"] == PERIODS[-1]]
    others = scores.loc[scores["model"] != baseline].dropna(
        subset=["rel_score"]
    )
    best = others.loc[others["rel_score"].idxmin()] if len(others) else None
    return {
        "source": meta["source"],
        "label": DUMMY_LABEL if meta["source"] == "dummy" else "",
        "baseline": baseline,
        "n_models": int(scores["model"].nunique()),
        "best_model": "" if best is None else str(best["model"]),
        "best_rel_score": None
        if best is None
        else round(float(best["rel_score"]), 2),
        "best_rel_log_score": None
        if best is None
        else round(float(best["rel_log_score"]), 2),
    }


def write_summary(path: Path, mode: str, studies: Mapping[str, dict]) -> None:
    """Write every study's numbers as Quarto metadata under `results`."""
    doc = {"results": {"mode": mode, **studies}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
