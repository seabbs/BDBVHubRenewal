# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Rolling held-out evaluation under the model choice record.

The record (`results/model-choice.yml`) logs the model chosen for
submission and every later change, each with a date:

```yaml
choices:
  - date: 2026-10-10
    model: randomwalk
    summary: Why this model.
```

Every real vintage after the first choice date is held out.
A held-out vintage is forecast by the model chosen at the time, the
latest choice dated before the vintage, and by the baseline.
Each forecast is scored on the count each later real vintage reports
(`bdbvhubrenewal.evaluation.nowcast.release_backtest`), so a forecast
gets its first score when the next vintage is published and another as
each newer one arrives.
The development backtest only uses vintages up to the latest choice date
(`bdbvhubrenewal.evaluation.backtest.BacktestConfig.development_until`).

`run_heldout` refuses to run unless asked explicitly and needs the record
to exist first.
It writes the forecasts, the scores and a run log with the hash of the
record.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from bdbvhubrenewal.data.dates import as_date
from bdbvhubrenewal.evaluation.backtest import (
    BacktestConfig,
    make_tasks,
    merge_by_model,
    real_vintages,
    run_forecasts,
)
from bdbvhubrenewal.evaluation.nowcast import release_backtest
from bdbvhubrenewal.forecasters import get_model, list_models

__all__ = [
    "HeldOutError",
    "ModelChoice",
    "heldout_pairs",
    "load_model_choices",
    "model_for_vintage",
    "run_heldout",
]

log = logging.getLogger(__name__)

_STORED = ["model", "vintage", "location", "target_end_date", "sample", "value"]
"""Columns of the final-count samples the release scores read."""


class HeldOutError(RuntimeError):
    """Raised when held-out vintages would be scored without permission."""


@dataclass(frozen=True)
class ModelChoice:
    """One entry of the model choice record.

    Attributes
    ----------
    date : str
        ISO date of the choice. Vintages after it use this model until
        the next choice.
    model : str
        Registered model name.
    summary : str
        Why this model was chosen.
    """

    date: str
    model: str
    summary: str = ""


def load_model_choices(path: Path | str) -> tuple[ModelChoice, ...]:
    """Read the model choice record.

    Parameters
    ----------
    path : pathlib.Path or str
        The YAML record, with a `choices` list.

    Returns
    -------
    tuple of ModelChoice
        Choices ordered by date. Empty if the file does not exist.

    Raises
    ------
    ValueError
        If two choices share a date.
    """
    path = Path(path)
    if not path.exists():
        return ()
    record = yaml.safe_load(path.read_text()) or {}
    choices = sorted(
        (
            ModelChoice(
                date=as_date(str(c["date"])).isoformat(),
                model=str(c["model"]),
                summary=str(c.get("summary", "")).strip(),
            )
            for c in record.get("choices", [])
        ),
        key=lambda c: c.date,
    )
    dates = [c.date for c in choices]
    if len(set(dates)) != len(dates):
        raise ValueError(f"Each model choice needs its own date: {dates}.")
    return tuple(choices)


def model_for_vintage(
    choices: Sequence[ModelChoice], vintage: dt.date | str
) -> str | None:
    """Return the model chosen at the time of a vintage.

    Parameters
    ----------
    choices : sequence of ModelChoice
        The record, in date order.
    vintage : datetime.date or str
        The vintage date.

    Returns
    -------
    str or None
        The model of the latest choice dated before the vintage, or None
        if the vintage is not held out.
    """
    day = as_date(vintage)
    model = None
    for c in choices:
        if as_date(c.date) < day:
            model = c.model
    return model


def heldout_pairs(
    choices: Sequence[ModelChoice], vintages: Sequence[str]
) -> pd.DataFrame:
    """List the held-out vintages, their model and their latest truth.

    Parameters
    ----------
    choices : sequence of ModelChoice
        The record, in date order.
    vintages : sequence of str
        Real vintages.

    Returns
    -------
    pandas.DataFrame
        Columns `vintage`, `model` and `truth_vintage`, the latest later
        vintage, or None while there is none.
    """
    ordered = sorted(vintages)
    latest = ordered[-1] if ordered else None
    rows = []
    for v in ordered:
        model = model_for_vintage(choices, v)
        if model is None:
            continue
        rows.append(
            {
                "vintage": v,
                "model": model,
                "truth_vintage": latest if latest != v else None,
            }
        )
    return pd.DataFrame(rows, columns=["vintage", "model", "truth_vintage"])


def _baseline() -> str:
    return next(s.name for s in list_models() if s.baseline)


def _baselines() -> list[str]:
    """Every registered model with no fit: cheap, so always scored."""
    return [s.name for s in list_models() if not s.fitted]


def run_heldout(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    config: BacktestConfig,
    *,
    evaluate_holdout: bool = False,
    out_dir: Path | str = "results",
    models: Sequence[str] = (),
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """Forecast and score every held-out vintage under its chosen model.

    Each held-out vintage is forecast by its chosen model, every model
    with no fit (random walk and randomwalknowcast) and any other `models`,
    and scored on the count each later real vintage reports.

    Parameters
    ----------
    target : pandas.DataFrame
        The tidy target table.
    delays : pandas.DataFrame
        The tidy delay table.
    config : BacktestConfig
        Settings, including the path of the model choice record.
    evaluate_holdout : bool
        Must be True. Guards against scoring held-out vintages by accident.
    out_dir : pathlib.Path or str
        Where to write `heldout-forecasts.csv.gz`, `heldout-scores.csv`
        and `heldout-run.yml`.
    models : sequence of str
        Other registered models to forecast and score alongside, such as
        the variants not chosen. Their scores keep `chosen_model`, so
        they compare with the choice but are not mistaken for it.
    daily : pandas.DataFrame
        Daily onsets by release, as for `make_tasks`.

    Returns
    -------
    pandas.DataFrame
        `ReleaseBacktest.scores` of the held-out origins, with
        `chosen_model`.

    Raises
    ------
    HeldOutError
        If `evaluate_holdout` is False or the model choice record is
        missing or empty.
    """
    if not evaluate_holdout:
        raise HeldOutError(
            "Held-out vintages are only scored with evaluate_holdout=True "
            "(`--evaluate-holdout`)."
        )
    choices = load_model_choices(config.model_choice)
    if not choices:
        raise HeldOutError(
            f"Write and commit the model choice record {config.model_choice} "
            "before the held-out evaluation."
        )
    real = target.loc[target["kind"] == "real"]
    pairs = heldout_pairs(choices, real_vintages(real))
    log.warning(
        "Running the held-out evaluation for vintages %s.",
        ", ".join(pairs["vintage"]),
    )
    baseline = _baseline()
    fitted = {s.name for s in list_models() if s.fitted}
    frames = []
    tasks = []
    for row in pairs.itertuples(index=False):
        names = list(dict.fromkeys([str(row.model), *_baselines(), *models]))
        # Fitted models are never run here, even when chosen: their
        # forecasts come from the fit store (`evaluation.store.load`).
        names = [n for n in names if n not in fitted]
        rows = real.loc[real["vintage"] == row.vintage]
        task = make_tasks(
            rows, delays, config, real_vintages(real), real, daily=daily
        )
        fc = run_forecasts([get_model(n) for n in names], task)
        fc.insert(1, "chosen_model", str(row.model))
        frames.append(fc)
        tasks += task
    out = Path(out_dir)
    # Forecasts of models not run here, the fitted ones from the fit
    # store, are kept and scored again.
    chosen = dict(zip(pairs["vintage"], pairs["model"], strict=True))
    if frames:
        forecasts = merge_by_model(
            out / "heldout-forecasts.csv.gz",
            pd.concat(frames, ignore_index=True),
        )
        final = forecasts.loc[forecasts["as_of"].isna(), _STORED]
        scores = release_backtest(
            tasks, daily, target, models=(), stored=final
        ).scores
    else:
        # No vintage after the choice yet, so nothing is held out.
        forecasts, scores = pd.DataFrame(), pd.DataFrame()
    if not scores.empty:
        scores.insert(1, "chosen_model", scores["origin"].map(chosen))
    out.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(out / "heldout-forecasts.csv.gz", index=False)
    scores.to_csv(out / "heldout-scores.csv", index=False)
    scored = set(scores["origin"]) if not scores.empty else set()
    record = {
        "run_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "model_choice": str(config.model_choice),
        "model_choice_sha256": hashlib.sha256(
            config.model_choice.read_bytes()
        ).hexdigest(),
        "choices": [vars(c) for c in choices],
        "baseline": baseline,
        "latest_vintage": real_vintages(real)[-1],
        "scored": [{"vintage": v, "model": chosen[v]} for v in sorted(scored)],
        "pending": [
            {"vintage": v, "model": chosen[v]}
            for v in pairs["vintage"]
            if v not in scored
        ],
    }
    (out / "heldout-run.yml").write_text(
        yaml.safe_dump(record, sort_keys=False)
    )
    return scores
