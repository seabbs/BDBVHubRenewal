# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""The live fine-tuning run's newest checkpoints, as preliminary results.

The checkpoint scorer lands each checkpoint's scores in
`evidence/checkpoint-scores.csv` on the `checkpoint-scoring` branch: the
German development half (WIS), the BDBV release backtest (CRPS) and the
trainer's held-out loss (`held_out`). The paper and the docs read that
table at the branch head, never a pinned checkpoint, and take each
study's newest checkpoint of the live run (`run`), labelled as
preliminary with the number of simulations it was fine-tuned on.

Only development data are shown. `tidy` refuses BDBV rows from the
scored hub rounds (`SCORED_FROM`); the scorer never reads the German
second half.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

__all__ = [
    "MODELS",
    "PLANNED",
    "RULE",
    "SCORED_FROM",
    "checkpoint_table",
    "label",
    "newest",
    "summary_table",
    "tidy",
]

RULE: Final = "own-quantile-wis_crps-1000-common"
"""The scoring rule shown: WIS on each model's own quantiles for Germany,
CRPS on 1,000 draws with common random numbers for BDBV."""

PLANNED: Final = 8000
"""Simulations the live run is fine-tuned on at its last checkpoint."""

SCORED_FROM: Final = "2026-11-21"
"""The first scored BDBV hub round, kept out of every fine-tuned result."""

HUB_TARGET: Final = "7day"
"""The target scored: the hub's own."""

MODELS: Final = ("fm_chronos_daily_ragged", "fm_chronos_weekly_ragged")
"""The fine-tuned models, shown daily then weekly in each cell."""

STUDIES: Final = ("germany", "ebola", "held_out")


def tidy(scores: pd.DataFrame) -> pd.DataFrame:
    """Return the shown rows, with one relative score column per scale.

    Keeps the case studies' rows of `RULE` and the hub target, and the
    trainer's held-out loss. `rel` and `rel_log` hold Germany's WIS and
    BDBV's CRPS relative to `randomwalknowcast`.

    Raises
    ------
    ValueError
        If a BDBV row reaches the scored hub rounds.
    """
    out = scores.loc[scores["study"].isin(STUDIES)].copy()
    if "rule" in out:
        case = out["study"] != "held_out"
        out = out.loc[~case | (out["rule"] == RULE)]
    if "target" in out:
        out = out.loc[out["target"].fillna(HUB_TARGET) == HUB_TARGET]
    if "last_origin" in out:
        late = (out["study"] == "ebola") & (
            out["last_origin"].astype(str) >= SCORED_FROM
        )
        if late.any():
            raise ValueError(
                f"BDBV rows reach the scored hub rounds (from {SCORED_FROM}), "
                "which stay out of fine-tuned results."
            )

    def col(name: str) -> pd.Series:
        return out[name] if name in out else pd.Series(float("nan"), out.index)

    out["rel"] = col("rel_wis").fillna(col("rel_crps"))
    out["rel_log"] = col("rel_log_wis").fillna(col("rel_log_crps"))
    return out.reset_index(drop=True)


def newest(scores: pd.DataFrame, run: str) -> dict[str, tuple[str, int]]:
    """Return each study's newest scored checkpoint of the live run.

    Studies the run has no rows for yet are left out.
    """
    rows = tidy(scores)
    rows = rows.loc[rows["run"] == run]
    out = {}
    for study, g in rows.groupby("study"):
        top = g.sort_values("outbreaks").iloc[-1]
        out[str(study)] = (str(top["checkpoint"]), int(top["outbreaks"]))
    return out


def label(outbreaks: int) -> str:
    """Return the preliminary label of a checkpoint."""
    return (
        f"Preliminary: fine-tuned on {outbreaks:,} of {PLANNED:,} simulations."
    )


def _pair(rows: pd.DataFrame, study: str, column: str) -> str:
    """Return the national daily and weekly scores as `d / w`, or ''."""
    one = rows.loc[(rows["study"] == study) & (rows["level"] == "national")]
    values = []
    for model in MODELS:
        v = one.loc[one["model"] == model, column].dropna()
        values.append(f"{v.iloc[0]:.2f}" if len(v) else "")
    return " / ".join(values) if any(values) else ""


def _loss(rows: pd.DataFrame, zero_shot: bool = False) -> str:
    one = rows.loc[(rows["study"] == "held_out") & (rows["level"] == "all")]
    v = one["loss_zero_shot" if zero_shot else "loss"].dropna()
    return f"{v.iloc[0]:.2f}" if len(v) else ""


COLUMNS: Final = (
    "German WIS",
    "BDBV CRPS",
    "BDBV log CRPS",
    "Held-out loss",
)


def _row(rows: pd.DataFrame, zero_shot: bool = False) -> dict[str, str]:
    return {
        "German WIS": _pair(rows, "germany", "rel"),
        "BDBV CRPS": _pair(rows, "ebola", "rel"),
        "BDBV log CRPS": _pair(rows, "ebola", "rel_log"),
        "Held-out loss": _loss(rows, zero_shot),
    }


def checkpoint_table(scores: pd.DataFrame, run: str) -> pd.DataFrame:
    """Return zero-shot then each checkpoint of the live run, one row each.

    Columns are the national relative scores on development data (daily /
    weekly model) and the trainer's held-out loss; a checkpoint not yet
    scored on a study is blank.
    """
    rows = tidy(scores)
    live = rows.loc[rows["run"] == run]
    out = [
        {
            "Simulations": "Zero-shot",
            **_row(rows.loc[rows["checkpoint"] == "zero-shot"]),
        }
    ]
    # Zero-shot has no held-out row of its own; each row carries it.
    out[0]["Held-out loss"] = _loss(live, zero_shot=True)
    for n in sorted(live["outbreaks"].unique()):
        g = live.loc[live["outbreaks"] == n]
        out.append({"Simulations": f"{int(n):,}", **_row(g)})
    return pd.DataFrame(out, columns=["Simulations", *COLUMNS])


def summary_table(scores: pd.DataFrame, run: str) -> pd.DataFrame:
    """Return zero-shot and each study's newest checkpoint of the live run.

    Each fine-tuned cell gives its checkpoint's simulations in brackets,
    as studies are scored at different checkpoints.
    """
    rows = tidy(scores)
    full = checkpoint_table(scores, run)
    top = newest(scores, run)
    live = rows.loc[rows["run"] == run]
    study_of = {
        "German WIS": "germany",
        "BDBV CRPS": "ebola",
        "BDBV log CRPS": "ebola",
        "Held-out loss": "held_out",
    }
    cells = {"Simulations": "Newest checkpoint"}
    for name, study in study_of.items():
        if study not in top:
            cells[name] = ""
            continue
        ck, n = top[study]
        value = _row(live.loc[live["checkpoint"] == ck])[name]
        cells[name] = f"{value} ({n:,})" if value else ""
    return pd.DataFrame(
        [full.iloc[0].to_dict(), cells], columns=list(full.columns)
    )
