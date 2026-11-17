"""German 1 to 4 week forecasts for the checkpoint scores (SI only).

The `germany_forecast` unit of scripts/checkpoint_scores.py. The German
hub scored nowcasts only; this asks how the models forecast its data.
From each development-half nowcast date (before `split_date`,
config/germany.toml) each model forecasts the hub's 7-day sums ending 7,
14, 21 and 28 days after the nowcast date (`HORIZON_WEEKS`), from the
same data and tasks as the nowcasts (`forecast_tasks`: the nowcast tasks
with their weeks moved 1 to 4 weeks past the nowcast's last week).
A date on or after the split is refused (`check_dates`), and so is any
target week ending on or after it (Sam: never the second half), so the
horizons have 73, 66, 59 and 52 origins (`origins_per_horizon`). Set
`KEEP_SECOND_HALF_TARGETS` to True (and rerun the units) to keep them.

No model is refitted or changed for this. `randomwalknowcast` walks on
from each sample's completed cut-off week (`RandomWalkNowcast._paths`),
and Chronos-2 forecasts every step of its grid to the last target week
(`FoundationModel.forecast_both`), so a task's later weeks are forecasts.
Each forecast is kept as quantiles at the hub's levels: the model's own
where it gives them (the weekly Chronos-2 models and the daily ones with
weekly total rows), else those of 1,000 draws
(`bdbvhubrenewal.evaluation.germany.germany_forecasts`).

Truth is the nowcasts' rule (`with_truth`): each sum's count in the data
`truth_delay` (40) days after it ends. The hub's triangle runs to
2022-08-08, past the last forecast's truth (2022-02-09 + 28 + 40 days),
so every forecast has one.

The models: `randomwalknowcast` and zero-shot Chronos-2 (`reference`),
and the `_ft` models of one checkpoint: the live run's final checkpoint
(`cs.FINAL_OUTBREAKS`, the reported model; none is picked on scores),
once its German nowcasts are done (`final_checkpoint`). Until then only
the reference forecasts. The label is pinned in
`<cache>/germany_forecast/label.txt` (`PIN`); remove it by hand to
forecast another. The poller's gate (`units`) waits for every
live German nowcast it can still finish: labels held, superseded or
failed too often (`skip`) do not block it. Forecasts are cached in
`<cache>/germany_forecast/<label>/<date>.csv.gz`, one file per date as
it finishes.

Rows: study `germany_forecast`, so readers of the other studies pass
them by, with `horizon_weeks` (1 to 4). WIS and log WIS by level
(national, states, age) and horizon, on the cells every model and
checkpoint made, relative to `randomwalknowcast` and to the same model
zero-shot (`rel_to`).
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import time
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import checkpoint_scores as cs

STUDY = "germany_forecast"
HORIZON_WEEKS = (1, 2, 3, 4)
"""Weeks from the nowcast date's hub week to each forecast week."""
REFERENCE_MODELS = (cs.BASELINE, *cs.ZERO_SHOT)
"""The reference run's models: no complete-data models, to save time."""
KEEP_SECOND_HALF_TARGETS = False
"""Keep target weeks ending on or after the split (False: dropped)."""
PIN = "label.txt"
"""In `<cache>/germany_forecast/`: the checkpoint the job forecasts."""
ZERO_SHOT = "zero-shot"
"""`rel_to` of rows divided by the same model zero-shot."""
CONFIG = "config/germany.toml"
SCORES = ["wis", "log_wis", "coverage_50", "coverage_95"]


def _config() -> Any:
    from bdbvhubrenewal.evaluation import germany as de

    return de.load_germany_config(CONFIG)


def dev_dates() -> list[str]:
    """Return the development half's nowcast dates."""
    from bdbvhubrenewal.evaluation import germany as de

    return de.nowcast_dates(_config(), held_out=False)


def check_dates(dates: Sequence[str], split_date: str) -> None:
    """Refuse any date of the held-out half, on or after `split_date`."""
    held = sorted(d for d in dates if d >= split_date)
    if held:
        raise ValueError(
            f"held-out nowcast dates (from {split_date}) are never "
            f"forecast here: {held[:3]}"
        )


def label_models(label: str) -> tuple[str, ...]:
    """Return the models a label runs."""
    return REFERENCE_MODELS if label == cs.REFERENCE else cs.FINETUNED


def forecast_tasks(
    triangle: pd.DataFrame,
    date: str,
    n_samples: int,
    seed: int,
    split_date: str,
) -> list[Any]:
    """Return a nowcast date's tasks with weeks 1 to 4 weeks past its own.

    The nowcast tasks (`nowcast_tasks`), with `target_end_dates` the
    `HORIZON_WEEKS` weeks after the nowcast's last week.

    Raises
    ------
    ValueError
        If `date` is on or after `split_date`.
    """
    import dataclasses

    from bdbvhubrenewal.data.germany import nowcast_tasks

    check_dates([date], split_date)
    out = []
    for task in nowcast_tasks(triangle, date, n_samples, seed):
        last = dt.date.fromisoformat(max(task.target_end_dates))
        weeks = tuple(
            (last + dt.timedelta(weeks=k)).isoformat() for k in HORIZON_WEEKS
        )
        out.append(dataclasses.replace(task, target_end_dates=weeks))
    return out


def forecasts(
    models: Sequence[str],
    triangle: pd.DataFrame,
    date: str,
    n_samples: int,
    seed: int,
    split_date: str,
) -> pd.DataFrame:
    """Forecast one date with each model, as quantiles at the hub's levels.

    Returns
    -------
    pandas.DataFrame
        `germany_forecasts` columns (`horizon` 7 to 28 days) and
        `horizon_weeks`.
    """
    from bdbvhubrenewal.evaluation import germany as de

    check_dates([date], split_date)

    def tasks(tri: pd.DataFrame, d: str, n: int, s: int) -> list[Any]:
        return forecast_tasks(tri, d, n, s, split_date)

    fc = de.germany_forecasts(
        models, triangle, [date], n_samples, seed, make_tasks=tasks
    )
    fc = first_half(fc, split_date)
    return fc.assign(horizon_weeks=fc["horizon"] // 7)


def first_half(fc: pd.DataFrame, split_date: str) -> pd.DataFrame:
    """Drop target weeks ending on or after the split, unless kept."""
    if KEEP_SECOND_HALF_TARGETS:
        return fc
    return fc.loc[fc["target_end_date"] < split_date].reset_index(drop=True)


def origins_per_horizon(
    dates: Sequence[str], split_date: str
) -> dict[int, int]:
    """Return the origins each horizon keeps (`first_half`)."""
    out = {}
    for h in HORIZON_WEEKS:
        ends = [
            (dt.date.fromisoformat(d) + dt.timedelta(weeks=h)).isoformat()
            for d in dates
        ]
        out[h] = sum(KEEP_SECOND_HALF_TARGETS or e < split_date for e in ends)
    return out


def with_truth(
    forecasts: pd.DataFrame, triangle: pd.DataFrame, truth_delay: int
) -> pd.DataFrame:
    """Add each sum's truth by the nowcasts' rule (`de.with_truth`)."""
    from bdbvhubrenewal.evaluation import germany as de

    return de.with_truth(forecasts, triangle, truth_delay)


# The unit ----------------------------------------------------------------


def unit_done(
    cache: Path, label: str, dates: Sequence[str] | None = None
) -> bool:
    """Whether every date has its file."""
    want = dev_dates() if dates is None else dates
    root = cache / STUDY / label
    return all((root / f"{d}.csv.gz").exists() for d in want)


def unit(cache: Path, label: str, dates: Sequence[str] | None = None) -> None:
    """Forecast each development date not yet cached, with truth."""
    import resource

    from bdbvhubrenewal.data.germany import hub_triangle

    config = _config()
    dates = dev_dates() if dates is None else list(dates)
    check_dates(dates, config.split_date)
    cs._pin(label)
    triangle = hub_triangle(
        Path("data/raw/germany") / Path(config.triangle).name
    )
    models = list(label_models(label))
    root = cs.unit_root(cache, STUDY, label)
    for i, date in enumerate(dates, 1):
        path = root / f"{date}.csv.gz"
        if path.exists():
            continue
        start = time.monotonic()
        fc = forecasts(
            models,
            triangle,
            date,
            config.n_samples,
            config.seed,
            config.split_date,
        )
        fc = with_truth(fc, triangle, config.truth_delay)
        cs._write(fc.assign(target="7day"), path)
        cs.free_caches()
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        print(
            f"{STUDY} {label} {date} ({i}/{len(dates)}): "
            f"{time.monotonic() - start:.0f}s, peak RSS so far {peak:.2f} GB",
            flush=True,
        )
    print(f"{STUDY} {label} done", flush=True)


# Which labels ------------------------------------------------------------


def _outbreaks(cache: Path, label: str) -> int:
    settings = cache / "heldout" / f"{label}.json"
    if not settings.exists():
        return 0
    return int(json.loads(settings.read_text()).get("n_outbreaks") or 0)


def live_labels(runs: Mapping[str, str]) -> list[str]:
    """Return the live run's checkpoints (`cs.CURRENT_RUNS`)."""
    return sorted(lb for lb, run in runs.items() if run in cs.CURRENT_RUNS)


def final_checkpoint(
    cache: Path, runs: Mapping[str, str], done: Any = None
) -> str | None:
    """Return the live run's final checkpoint if its nowcasts are done.

    The final checkpoint (`cs.FINAL_OUTBREAKS` outbreaks) is the reported
    model; none is picked on scores. None until it exists and its German
    nowcasts are done. `done` is `cs.unit_done` by default.
    """
    done = done or cs.unit_done
    final = [
        lb
        for lb in live_labels(runs)
        if _outbreaks(cache, lb) == cs.FINAL_OUTBREAKS
        and done(cache, "germany", lb)
    ]
    return min(final) if final else None


def pinned(cache: Path) -> str | None:
    """Return the pinned checkpoint (`PIN`), or None."""
    pin = cache / STUDY / PIN
    return pin.read_text().strip() or None if pin.exists() else None


def units(
    cache: Path,
    runs: Mapping[str, str],
    done: Any = None,
    skip: Collection[str] = (),
) -> list[tuple[str, str]]:
    """Return the forecast units still to run, pinning the label.

    With a pinned label (`PIN`), the reference's and its units. Else none
    while a live German nowcast not in `skip` (labels the poller cannot
    finish) is not done; then the reference's, and the final checkpoint's
    (`final_checkpoint`) once it exists, which is pinned here. `done` is `cs.unit_done` by
    default.
    """
    done = done or cs.unit_done
    label = pinned(cache)
    if label is None:
        live = [lb for lb in live_labels(runs) if lb not in set(skip)]
        if any(not done(cache, "germany", lb) for lb in live):
            return []
        label = final_checkpoint(cache, runs, done)
        if label is not None:
            root = cache / STUDY
            root.mkdir(parents=True, exist_ok=True)
            (root / PIN).write_text(label + "\n")
    labels = [cs.REFERENCE, *([label] if label else [])]
    return [(STUDY, lb) for lb in labels if not done(cache, STUDY, lb)]


# Rows --------------------------------------------------------------------


def scores(fc: pd.DataFrame, truth_delay: int) -> pd.DataFrame:
    """Score each forecast by WIS and log WIS, with its horizon."""
    from germany_chronos_dev import scores as germany_scores

    sc = germany_scores(fc.drop(columns="horizon_weeks"), truth_delay)
    return sc.assign(horizon_weeks=sc["horizon"] // 7)


def _means(part: pd.DataFrame) -> pd.DataFrame:
    by = part.groupby(["model", "checkpoint"])
    mean = by[SCORES].mean()
    mean["n"] = by.size()
    mean["n_origins"] = by["origin"].nunique()
    mean["first_origin"] = by["origin"].min()
    mean["last_origin"] = by["origin"].max()
    return mean


def relative_rows(sc: pd.DataFrame) -> pd.DataFrame:
    """Return mean scores by level and horizon, relative two ways.

    `sc` holds each label's scores with `checkpoint`, `level` and
    `horizon_weeks`. In each level only the cells every model and
    checkpoint made are kept (`cs.matched`). Each model is divided by
    `randomwalknowcast` (`rel_to` `randomwalknowcast`) and by the same
    model zero-shot (`rel_to` `zero-shot`), where it has one.
    """
    sc = cs.matched(sc, cs.GERMANY_CELL)
    out = []
    for (target, level, h), part in sc.groupby(
        ["target", "level", "horizon_weeks"]
    ):
        mean = _means(part)
        keys = {"target": target, "level": level, "horizon_weeks": h}
        if (cs.BASELINE, ZERO_SHOT) in mean.index:
            base = mean.loc[(cs.BASELINE, ZERO_SHOT)]
            out.append(
                mean.assign(
                    rel_wis=mean["wis"] / base["wis"],
                    rel_log_wis=mean["log_wis"] / base["log_wis"],
                    rel_to=cs.BASELINE,
                    **keys,
                ).reset_index()
            )
        for (model, ckpt), row in mean.iterrows():
            if model == cs.BASELINE or (model, ZERO_SHOT) not in mean.index:
                continue
            zero = mean.loc[(model, ZERO_SHOT)]
            out.append(
                pd.DataFrame([row])
                .assign(
                    model=model,
                    checkpoint=ckpt,
                    rel_wis=row["wis"] / zero["wis"],
                    rel_log_wis=row["log_wis"] / zero["log_wis"],
                    rel_to=ZERO_SHOT,
                    **keys,
                )
                .reset_index(drop=True)
            )
    if not out:
        return pd.DataFrame(columns=cs.COLUMNS)
    rows = pd.concat(out, ignore_index=True).assign(study=STUDY, rule=cs.RULE)
    rows["horizon_weeks"] = rows["horizon_weeks"].astype(int)
    return rows.reindex(columns=cs.COLUMNS)


def rows(
    cache: Path, labels: Sequence[str], dates: Sequence[str] | None = None
) -> pd.DataFrame:
    """Return the rows of the labels with every date forecast.

    None until the reference has: every row divides by its models.
    """
    want = dev_dates() if dates is None else list(dates)
    ready = [lb for lb in labels if unit_done(cache, lb, want)]
    if cs.REFERENCE not in ready:
        return pd.DataFrame(columns=cs.COLUMNS)
    config = _config()
    parts = []
    for label in ready:
        files = [cache / STUDY / label / f"{d}.csv.gz" for d in want]
        fc = pd.concat([cs.read_cache(f) for f in files], ignore_index=True)
        fc = first_half(
            fc.loc[fc["origin"] < config.split_date], config.split_date
        )
        sc = scores(fc.drop(columns="target"), config.truth_delay)
        parts.append(cs._tag(sc, label).assign(target="7day"))
    sc = pd.concat(parts, ignore_index=True).rename(columns={"pool": "level"})
    sc["horizon_weeks"] = sc["horizon_weeks"].astype(np.int64)
    return relative_rows(sc)


def main(argv: Sequence[str]) -> None:
    """Run `python scripts/germany_forecast.py CACHE LABEL [N_DATES]`.

    With N_DATES, only the first that many development dates (a dry run).
    """
    dates = dev_dates()[: int(argv[2])] if len(argv) > 2 else None
    unit(Path(argv[0]), argv[1], dates)


if __name__ == "__main__":
    main(sys.argv[1:])
