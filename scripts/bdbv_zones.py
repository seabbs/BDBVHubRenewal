"""BDBV health-zone draws and rows for the checkpoint scores.

The `bdbv_zones` unit of scripts/checkpoint_scores.py. For each origin
whose province draws a label's `bdbv` unit cached, the task is rebuilt
as `checkpoint_scores.bdbv_unit` builds it (`store.origin_tasks`, the
backtest's seed, 1,000 draws), and each model's 7-day zone draws are
taken as in a hub submission:

- `own`: a model whose cached draws hold zone rows (main's province
  models return their zones from their own forecast, #299, and
  Chronos-2 forecasts them, #234) is scored on those rows as they are.
- `native`: `randomwalknowcast` with no cached zone rows is run on the
  zones' own release snapshots (`zones_for_model`), from the same task,
  seed and stream as in a submission. Since #263 these zones are its own
  forecasts and need not sum to its provinces.
- `split`: any other model with no zone rows has its province draws
  split across the zones by `ZoneAllocator`, checked to sum to them.
  The backtest tasks carry no zone data (`store.origin_tasks` without
  `zones`), as the old poller scored, so this is how a Chronos-2
  checkpoint's zones are made here, not as its submission makes them.

An origin where no model has zones stops the unit. Only the zones
with a case at the origin (`active_zones`) are kept, in
`<cache>/bdbv_zones/<label>/<origin>.csv.gz`, one file per origin as it
finishes. Zone codes come from the hub data (`hub-config/tasks.json`
and `daily_zones.csv.gz`).

Rows: sample CRPS and log CRPS of each zone and week (`cell_scores`)
against the zone truth (`zone_truth`), at level `zones` of the `ebola`
study, on the cells every model and checkpoint made, relative to
`randomwalknowcast`, with `zone_method` saying how a model's zones were
made.
"""

from __future__ import annotations

import dataclasses
import sys
import time
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import checkpoint_scores as cs

STUDY = "bdbv_zones"
LEVEL = "zones"
NATIVE = "native"
SPLIT = "split"
OWN = "own"
NATIVE_MODELS = (cs.BASELINE,)
"""Models run on the zones (`zones_for_model`'s random walk with no fit)
when their cached draws hold no zone rows."""
SAMPLE = ["location", "target_end_date", "sample", "value"]


def zone_method(model: str) -> str:
    """Return how a model's zones are made if its cache has none.

    `native` (run on the zones) or `split` (from its province draws).
    """
    return NATIVE if model in NATIVE_MODELS else SPLIT


def active_zones(counts: pd.DataFrame) -> set[str]:
    """Return the zones with a case at the origin (`weekly_zone_counts`)."""
    return set(counts["location"])


def province_draws(draws: pd.DataFrame, model: str, task: Any) -> pd.DataFrame:
    """Return a model's cached 7-day province draws, checked against `task`.

    The national rows and the daily target are dropped. The draws must
    cover the task's samples and weeks, or the split would read gaps.
    """
    rows = draws.loc[
        (draws["model"] == model)
        & (cs.with_target(draws)["target"] == "7day")
        & (draws["location"].str.count("-") == 1)
    ]
    rows = rows[SAMPLE].reset_index(drop=True)
    weeks = set(rows["target_end_date"])
    samples = set(rows["sample"].astype(int))
    if weeks != set(task.target_end_dates) or samples != set(
        range(1, task.n_samples + 1)
    ):
        raise ValueError(
            f"{model} at {task.vintage}: cached draws cover "
            f"{len(weeks)} weeks and {len(samples)} samples, the task "
            f"{len(task.target_end_dates)} weeks and {task.n_samples}"
        )
    return rows


def origin_zones(
    task: Any,
    draws: pd.DataFrame,
    counts: pd.DataFrame,
    zones: tuple[str, ...],
    daily_zones: pd.DataFrame,
    active: Collection[str] | None = None,
) -> pd.DataFrame:
    """Return each cached model's zone draws at one origin.

    Parameters
    ----------
    task : ForecastTask
        The origin's province task, as the `bdbv` unit ran it.
    draws : pandas.DataFrame
        The origin's cache file (`checkpoint_scores.read_cache`).
    counts : pandas.DataFrame
        Weekly zone counts at the origin (`weekly_zone_counts`).
    zones : tuple of str
        The hub's zone codes (`hub_zones`).
    daily_zones : pandas.DataFrame
        The daily zone table (`load_daily_zones`).
    active : collection of str, optional
        Keep only these zones. All if None.

    Returns
    -------
    pandas.DataFrame
        `location`, `target_end_date`, `sample`, `value`, `model` and
        `zone_method`.
    """
    from bdbvhubrenewal.data.zones import is_zone
    from bdbvhubrenewal.forecasters.zones import (
        ZoneAllocator,
        check_zone_coherence,
        zones_for_model,
    )

    seven = draws.loc[cs.with_target(draws)["target"] == "7day"]
    frames = []
    for model in sorted(set(seven["model"])):
        rows = seven.loc[seven["model"] == model]
        own = rows.loc[rows["location"].map(is_zone), SAMPLE]
        if not own.empty:
            got, method = own.reset_index(drop=True), OWN
        elif zone_method(model) == NATIVE:
            prov = province_draws(seven, model, task)
            got = zones_for_model(model, task, prov, counts, zones, daily_zones)
            method = NATIVE
        else:
            prov = province_draws(seven, model, task)
            got = ZoneAllocator().allocate(prov, task, counts, zones)
            bad = check_zone_coherence(prov, got)
            if bad:
                raise ValueError(f"{model} at {task.vintage}: {bad[0]}")
            method = SPLIT
        if got.empty:
            continue
        if active is not None:
            got = got.loc[got["location"].isin(set(active))]
        frames.append(got.assign(model=model, zone_method=method))
    if not frames:
        raise ValueError(f"no zone draws of any model at {task.vintage}")
    out = pd.concat(frames, ignore_index=True)
    out["value"] = out["value"].astype("float32")
    return out


# The unit ----------------------------------------------------------------


def _fresh(out: Path, src: Path) -> bool:
    """Whether `out` was written after its province file `src`."""
    return out.exists() and out.stat().st_mtime >= src.stat().st_mtime


def unit_done(
    cache: Path, label: str, origins: Sequence[str] | None = None
) -> bool:
    """Whether every origin's province draws have newer zone draws.

    False until the label's `bdbv` unit has every origin (`origins`, the
    backtest's by default). A province file rewritten, with new models
    say, makes its zone file stale.
    """
    origins = cs.bdbv_origins() if origins is None else origins
    src, out = cache / "bdbv" / label, cache / STUDY / label
    for origin in origins:
        name = f"{origin}.csv.gz"
        if not (src / name).exists() or not _fresh(out / name, src / name):
            return False
    return True


def has_zones(
    cache: Path, label: str, origins: Sequence[str] | None = None
) -> bool:
    """Whether every origin has a zone file, fresh or not.

    The rows read these, so a province file rewritten (the daily target
    added, say) keeps the label's zone rows until its zones are redone.
    """
    origins = cs.bdbv_origins() if origins is None else origins
    out = cache / STUDY / label
    return all((out / f"{o}.csv.gz").exists() for o in origins)


def unit(cache: Path, label: str) -> None:
    """Write zone draws for each origin of a label's cached draws."""
    import resource

    from bdbvhubrenewal.data import load_daily_zones
    from bdbvhubrenewal.data.zones import hub_zones, weekly_zone_counts
    from bdbvhubrenewal.evaluation import backtest as bt

    config = bt.load_config()
    data = Path("data")
    tasks = cs.bdbv_tasks(data, config)
    daily_zones = load_daily_zones(data)
    zones = hub_zones(data / "hub-config" / "tasks.json")
    src = cache / "bdbv" / label
    root = cs.unit_root(cache, STUDY, label)
    for origin in cs.bdbv_origins():
        path, out = src / f"{origin}.csv.gz", root / f"{origin}.csv.gz"
        if not path.exists() or _fresh(out, path):
            continue
        start = time.monotonic()
        task = dataclasses.replace(
            tasks[origin], n_samples=cs.N_DRAWS, seed=config.seed
        )
        counts = weekly_zone_counts(daily_zones, origin)
        made = origin_zones(
            task,
            cs.read_cache(path),
            counts,
            zones,
            daily_zones,
            active=active_zones(counts),
        )
        cs._write(made.assign(origin=origin), out)
        cs.free_caches()
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        print(
            f"{STUDY} {label} {origin}: {made['model'].nunique()} models, "
            f"{made['location'].nunique()} zones, "
            f"{time.monotonic() - start:.0f}s, peak RSS so far {peak:.2f} GB",
            flush=True,
        )
    print(f"{STUDY} {label} done", flush=True)


# Rows --------------------------------------------------------------------


def zone_truth(
    daily_zones: pd.DataFrame, cutoffs: Mapping[str, str], delay: int
) -> pd.DataFrame:
    """Return each zone-week's count in the first release `delay` days on.

    The provinces' rule (`rolling_truth`) on the zone counts. Each release
    covers every zone any release has a case in, and every week from the
    first onset week to its cut-off week, 0 where it has no onset.

    Parameters
    ----------
    daily_zones : pandas.DataFrame
        The daily zone table (`load_daily_zones`).
    cutoffs : mapping
        Each release's cut-off, as in the province target, so a zone-week
        takes the release its province-week takes.
    delay : int
        $D$, the days from the end of a week to its truth's cut-off.

    Returns
    -------
    pandas.DataFrame
        `location`, `target_end_date`, `truth` and `truth_vintage`.
    """
    from bdbvhubrenewal.data.zones import weekly_zone_counts
    from bdbvhubrenewal.evaluation.nowcast import rolling_truth

    places = sorted(set(daily_zones["location"]))
    onset = pd.to_datetime(daily_zones["onset_date"]).min()
    first = onset + pd.Timedelta(days=(5 - onset.weekday()) % 7)
    frames = []
    for vintage in sorted(set(daily_zones["vintage"].astype(str))):
        if vintage not in cutoffs:
            continue
        cutoff = pd.Timestamp(cutoffs[vintage])
        last = cutoff + pd.Timedelta(days=(5 - cutoff.weekday()) % 7)
        weeks = pd.date_range(first, last, freq="7D").strftime("%Y-%m-%d")
        grid = pd.MultiIndex.from_product(
            [places, weeks], names=["location", "target_end_date"]
        )
        counts = weekly_zone_counts(daily_zones, vintage)
        frames.append(
            counts.set_index(["location", "target_end_date"])["total"]
            .reindex(grid, fill_value=0)
            .reset_index()
            .assign(vintage=vintage, cutoff=cutoffs[vintage])
        )
    truth = rolling_truth(pd.concat(frames, ignore_index=True), delay)
    return truth.astype({"target_end_date": str, "truth_vintage": str})


def score_rows(
    draws: Mapping[str, Iterable[pd.DataFrame]], truth: pd.DataFrame
) -> pd.DataFrame:
    """Return the zone rows of each label's zone draws.

    Each frame is scored as it comes and only its cell scores are kept,
    so frames read lazily are held one at a time.

    Parameters
    ----------
    draws : mapping
        Each label's zone draws, a frame per origin, with `model` and
        `origin`. The reference's must be there.
    truth : pandas.DataFrame
        `zone_truth`.

    Returns
    -------
    pandas.DataFrame
        Rows with `checkpoint_scores.COLUMNS`, level `zones`.
    """
    parts = []
    for label, frames in draws.items():
        for frame in frames:
            sc = cs.cell_scores(
                frame.drop(columns="zone_method", errors="ignore"), truth
            )
            parts.append(cs._tag(sc, label))
    sc = pd.concat(parts, ignore_index=True).assign(level=LEVEL, target="7day")
    out = cs.relative(
        cs.matched(sc, cs.BDBV_CELL), "ebola", ("crps", "log_crps")
    )
    out["zone_method"] = out["model"].map(zone_method)
    return out.reindex(columns=cs.COLUMNS)


def rows(
    cache: Path, labels: Sequence[str], origins: Sequence[str] | None = None
) -> pd.DataFrame:
    """Return the zone rows of the labels with zones at every origin.

    None until the reference has: every row divides by its
    `randomwalknowcast`. Each file is read and scored in turn.
    """
    ready = [lb for lb in labels if has_zones(cache, lb, origins)]
    if cs.REFERENCE not in ready:
        return pd.DataFrame(columns=cs.COLUMNS)
    from bdbvhubrenewal.data import load_daily_zones
    from bdbvhubrenewal.evaluation import backtest as bt
    from bdbvhubrenewal.evaluation.nowcast import resolve_truth_delay

    config = bt.load_config()
    target = pd.read_csv("data/target.csv", dtype={"kind": str})
    real = target.loc[target["kind"] == "real"]
    cutoffs = real.groupby("vintage")["cutoff"].first().astype(str).to_dict()
    delay = resolve_truth_delay(config, target)
    truth = zone_truth(load_daily_zones(Path("data")), cutoffs, delay)
    want = set(cs.bdbv_origins() if origins is None else origins)

    def read(label: str) -> Iterator[pd.DataFrame]:
        for f in sorted((cache / STUDY / label).glob("*.csv.gz")):
            if f.name.removesuffix(".csv.gz") not in want:
                continue
            yield pd.read_csv(
                f,
                usecols=[*SAMPLE, "model", "origin"],
                dtype={
                    "origin": str,
                    "target_end_date": str,
                    "location": "category",
                    "model": "category",
                    "value": "float32",
                },
            ).astype({"location": str, "model": str})

    return score_rows({lb: read(lb) for lb in ready}, truth)


def main(argv: Sequence[str]) -> None:
    """Run `python scripts/bdbv_zones.py CACHE LABEL`."""
    unit(Path(argv[0]), argv[1])


if __name__ == "__main__":
    main(sys.argv[1:])
