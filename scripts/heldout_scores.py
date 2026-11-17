"""Proper scores of each checkpoint on the held-out simulations.

Sam's request of 2026-11-17: besides the trainer's held-out loss, score
the held-out synthetic set with proper scores, for every checkpoint, with
zero-shot Chronos-2 and randomwalknowcast as references.

Two held-out studies sit in evidence/checkpoint-scores.csv. Rows of
study `held_out` are the trainer's own held-out loss, read from each
release's settings.json (`checkpoint_scores.held_out_rows`). Rows of
study `heldout` are this module's: WIS and sample CRPS on a fixed sample
of the same held-out blocks, by layout and target.

The blocks
----------
The held-out simulations are the trainer's: the first `HOLDOUT`
simulations of seeds `HELD_SEEDS` in `SIM_DIR` (checked against the
releases' `held_out_sha256`), cut as `finetune_stream` cuts them, with
the trainer's seed (`CUT_SEED`, its `--seed`) and settings
(`cut_settings`) and the one-model run's five layouts (`LAYOUTS`, in
`finetune_chronos.SOURCE_SHARES` order). Scoring every block takes about
8 hours a label on one thread, so `SAMPLE` blocks of each layout are
drawn with `SAMPLE_SEED`, the same for every label. The cut and sample
are cached once (`prepare`).

Targets: a layout's count rows are its own target (daily, 7day for the
weekly layout, 3day). A mixed layout (daily+weekly, daily+3day) has
daily count rows (target daily) and each place's 7-day or 3-day totals
on period-end days (target 7day or 3day). A row and step is scored where
the block has its truth; the totals rows have none on other days.

Each step is in the nowcast window if its first day is on or before the
block's cut-off, else the forecast window (`windows`). Levels are the
blocks' (national, intermediate, leaf).

The models
----------
- `fm_chronos`: Chronos-2 on the block as the trainer's held-out loss
  runs it (`finetune_stream.ByLength` and `forward`, so the series bias
  is assigned as there), zero-shot for the reference label and with the
  checkpoint's adapter otherwise. Its native quantiles are read at the
  hub levels (`scoring.QUANTILE_LEVELS`), linear between them as
  `foundation.native_sampler` does, and scored by WIS and log WIS.
- `randomwalknowcast`, reference label only, on the weekly layout's 7day
  target alone (`BASELINE_CELLS`): it nowcasts hub weeks and has no
  daily or 3-day output on this code. Its task is built from the block's
  own releases (`walk_task`). It gives 1,000 draws, scored by sample CRPS
  and, to compare with Chronos-2, by WIS of their quantiles at the same
  levels. On blocks whose country is not the sum of its places
  (`unallocated`, `late_parent`) it forecasts the country from the
  country's own releases, as the block holds them.

Rows are means over the cells every model of a (layout, target) made,
relative to randomwalknowcast where it forecast and to zero-shot Chronos-2
elsewhere (`rel_to`).
"""

from __future__ import annotations

import copy
import datetime as dt
import gc
import hashlib
import inspect
import json
import pickle
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import finetune_chronos as ft
import finetune_stream as fs

from bdbvhubrenewal.evaluation.scoring import (
    QUANTILE_LEVELS,
    crps_sample,
    wis_quantile,
)
from bdbvhubrenewal.forecasters.finetune import AHEAD_WEEKS, MIXED

STUDY = "heldout_scores"
"""The unit's name and cache folder."""
SIM_DIR = Path("~/bdbv-rounds/rounds/r1/sims")
HELD_SEEDS = (30140, 30143)
HOLDOUT = 150
HELD_SHA256 = "79ad959990d3e36f5620327c32be8fe099e276ec2091ae3272e17df18ea095da"
"""Every r4 checkpoint's `held_out_sha256`: its 100 held-out simulations."""
CUT_SEED = 1
"""The trainer's `--seed`, which seeds each simulation's cut."""
PER_SIMULATION = 32
"""The trainer's `--blocks-per-simulation`."""
DRAWS, SUBSET_SEED = 2, 1
"""The trainer's `--draws-per-simulation` and `--subset-seed`."""
UNIVARIATE = 0.2
MEMORY_GB = 36.0
LAYOUTS = tuple(ft.SOURCE_SHARES)
SAMPLE = {
    "daily": 40,
    "weekly": 100,
    "3day": 80,
    "daily+weekly": 40,
    "daily+3day": 30,
}
"""Blocks scored per layout. Chronos-2 takes about 0.14 s a series on one
thread, so these take about 45 minutes a label: more of the small weekly
and 3-day blocks, fewer of the larger daily ones."""
SAMPLE_SEED = 20261117
CHUNK = 10
"""Blocks per cache file."""
N_DRAWS = 1000
WALK_SEED = 1
BASELINE = "randomwalknowcast"
CHRONOS = "fm_chronos"
ZERO_SHOT = "zero-shot"
BASELINE_CELLS = {("weekly", "7day")}
"""The layout and target cells randomwalknowcast forecasts."""
PERIOD_TARGET = {"weekly": "7day", "3day": "3day"}
OWN_TARGET = {"daily": "daily", "weekly": "7day", "3day": "3day"}
STEP_DAYS = {"weekly": 7, "3day": 3}
SCORES = ("wis", "log_wis", "crps", "coverage_50", "coverage_90", "coverage_95")
FRAME = [
    "block",
    "layout",
    "target",
    "level",
    "window",
    "row",
    "step",
    "truth",
    "model",
    *SCORES,
]
ROWS = [
    "study",
    "level",
    "model",
    "checkpoint",
    "n",
    "n_blocks",
    "layout",
    "target",
    "window",
    "rel_to",
    "wis",
    "rel_wis",
    "rel_log_wis",
    "crps",
    "coverage_50",
    "coverage_90",
    "coverage_95",
]


# The blocks ------------------------------------------------------------


def cut_settings() -> tuple:
    """Return `finetune_stream.init_worker`'s arguments for the cut."""
    return (
        min(ft.MAX_SERIES, ft.max_series(MEMORY_GB)),
        ft.MAX_LAGS,
        LAYOUTS,
        UNIVARIATE,
        DRAWS,
        SUBSET_SEED,
    )


def cut_blocks(
    sim_dir: Path,
    ids: Sequence[str],
    per_simulation: int = PER_SIMULATION,
    seed: int = CUT_SEED,
) -> list[dict]:
    """Return the held-out blocks as the trainer cuts them, with metadata.

    The blocks are `finetune_stream.cut`'s. Each also has its `outbreak`
    and `meta`: the cut-offs, places, `training_block` options and the
    generator its country draws used, its start and horizon, and for
    weekly blocks the counts, so randomwalknowcast's task can be built
    from the releases the block was cut from.
    """
    fs.init_worker(*cut_settings())
    block_of, pad_of = ft.training_block, ft._pad

    def training_block(counts, source, places, cutoffs, **kw):
        meta = {
            "source": source,
            "places": tuple(places),
            "cutoffs": list(cutoffs),
            "options": {k: v for k, v in kw.items() if k != "rng"},
            "rng": copy.deepcopy(kw.get("rng")),
            "counts": counts if source == "weekly" else None,
        }
        out = block_of(counts, source, places, cutoffs, **kw)
        return None if out is None else {**out, "_meta": meta}

    def pad(block, context):
        meta = {
            **block["_meta"],
            "start": block["start"],
            "horizon": block["horizon"],
        }
        return {**pad_of(block, context), "meta": meta}

    ft.training_block, ft._pad = training_block, pad
    try:
        pairs = fs.cut(Path(sim_dir), list(ids), seed, per_simulation)
    finally:
        ft.training_block, ft._pad = block_of, pad_of
    return [{**b, "outbreak": o} for o, b in pairs]


def sample_index(
    blocks: Sequence[Mapping], sizes: Mapping[str, int], seed: int
) -> list[int]:
    """Return a fixed sample of `sizes[layout]` blocks of each layout."""
    rng = np.random.default_rng(seed)
    out: list[int] = []
    for layout in LAYOUTS:
        mine = [i for i, b in enumerate(blocks) if b["source"] == layout]
        k = min(sizes.get(layout, 0), len(mine))
        out += sorted(int(mine[j]) for j in rng.choice(len(mine), k, False))
    return out


def _key(sha: str, sizes: Mapping[str, int], per_simulation: int) -> str:
    """Return the cut and sample's key: the held-out set and settings."""
    text = json.dumps(
        {
            "held_out": sha,
            "seed": CUT_SEED,
            "per_simulation": per_simulation,
            "layouts": LAYOUTS,
            "sizes": dict(sizes),
            "sample_seed": SAMPLE_SEED,
            "settings": [DRAWS, SUBSET_SEED, UNIVARIATE, MEMORY_GB],
        },
        sort_keys=True,
    )
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _held_ids(sim_dir: Path) -> list[str]:
    seeds = range(HELD_SEEDS[0], HELD_SEEDS[1] + 1)
    return fs.ready_outbreaks(sim_dir, seeds)[:HOLDOUT]


def _atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.rename(path)


def prepare(
    cache: Path,
    sim_dir: Path = SIM_DIR,
    ids: Sequence[str] | None = None,
    sha: str = HELD_SHA256,
    sizes: Mapping[str, int] = SAMPLE,
    per_simulation: int = PER_SIMULATION,
) -> list[dict]:
    """Return the sampled held-out blocks, cutting them once.

    The blocks are written to `<cache>/heldout_scores/blocks-<key>.pkl`
    and the key, sample and block counts to `sample.json` there. Each
    block gains its `index` in the full cut.

    Raises
    ------
    ValueError
        If the simulations' ids do not hash to `sha`, the held-out set.
    """
    root = Path(cache) / STUDY
    root.mkdir(parents=True, exist_ok=True)
    key = _key(sha, sizes, per_simulation)
    path = root / f"blocks-{key}.pkl"
    if path.exists():
        with path.open("rb") as f:
            return list(pickle.load(f))
    sim_dir = Path(sim_dir).expanduser()
    ids = list(_held_ids(sim_dir) if ids is None else ids)
    found = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
    if found != sha:
        raise ValueError(
            f"the held-out simulations hash to {found}, not the trainer's "
            f"held-out set {sha}"
        )
    blocks = cut_blocks(sim_dir, ids, per_simulation)
    index = sample_index(blocks, sizes, SAMPLE_SEED)
    sampled = [{**blocks[i], "index": i} for i in index]
    for b in sampled:
        if b["source"] != "weekly":
            b["meta"] = {**b["meta"], "counts": None}
    _atomic(path, pickle.dumps(sampled))
    counts = pd.Series([b["source"] for b in blocks]).value_counts()
    note = {
        "key": key,
        "held_out_sha256": sha,
        "n_blocks": len(sampled),
        "cut": {str(k): int(v) for k, v in counts.items()},
        "n_cut": len(blocks),
        "sample": {s: int(sizes.get(s, 0)) for s in LAYOUTS},
        "index": index,
    }
    _atomic(root / "sample.json", json.dumps(note, indent=1).encode())
    return sampled


def _sample(cache: Path) -> dict | None:
    path = Path(cache) / STUDY / "sample.json"
    return json.loads(path.read_text()) if path.exists() else None


# Rows, steps and windows -----------------------------------------------


def row_targets(block: Mapping) -> list[str]:
    """Return each target row's target: daily, 7day or 3day."""
    n, source = int(block["n_targets"]), block["source"]
    if source in MIXED:
        half = n // 2
        return ["daily"] * half + [PERIOD_TARGET[MIXED[source]]] * half
    return [OWN_TARGET[source]] * n


def step_ends(block: Mapping) -> list[dt.date]:
    """Return the last day of each step of a block's horizon.

    The horizon ends `AHEAD_WEEKS` after the Saturday ending the cut-off's
    week (`finetune.training_block`); 3-day bins end on the cut-off and
    every 3 days from it (`finetune.bin_ends`).
    """
    meta = block["meta"]
    cut = dt.date.fromisoformat(meta["cutoffs"][-1])
    end = cut + dt.timedelta(days=(5 - cut.weekday()) % 7)
    end += dt.timedelta(weeks=AHEAD_WEEKS)
    h, width = int(meta["horizon"]), STEP_DAYS.get(block["source"], 1)
    if block["source"] == "3day":
        end = cut + dt.timedelta(days=3 * ((end - cut).days // 3))
    return [end - dt.timedelta(days=width * (h - 1 - i)) for i in range(h)]


def windows(block: Mapping) -> np.ndarray:
    """Return each step's window: nowcast if it starts by the cut-off."""
    cut = dt.date.fromisoformat(block["meta"]["cutoffs"][-1])
    width = STEP_DAYS.get(block["source"], 1)
    return np.array(
        [
            "nowcast" if e - dt.timedelta(days=width - 1) <= cut else "forecast"
            for e in step_ends(block)
        ]
    )


def truth(block: Mapping) -> np.ndarray:
    """Return the target rows' final counts over the horizon, `[n, h]`."""
    n, h = int(block["n_targets"]), int(block["meta"]["horizon"])
    return np.asarray(block["context"][:n, ft.CONTEXT : ft.CONTEXT + h], float)


# Scores ----------------------------------------------------------------


def at_levels(
    q: np.ndarray, native: np.ndarray, levels: np.ndarray = QUANTILE_LEVELS
) -> np.ndarray:
    """Read quantiles `[..., Q]` at `levels`, after sorting them.

    Linear between the native levels and flat beyond them, as
    `foundation.native_sampler` reads them.
    """
    q = np.sort(np.asarray(q, float), axis=-1)
    native = np.asarray(native, float)
    hi = np.clip(np.searchsorted(native, levels), 1, len(native) - 1)
    lo = hi - 1
    w = np.clip((levels - native[lo]) / (native[hi] - native[lo]), 0.0, 1.0)
    return np.asarray(q[..., lo] * (1 - w) + q[..., hi] * w)


def _covers(y: np.ndarray, q: np.ndarray, levels: np.ndarray, p: int):
    a = (1 - p / 100) / 2
    lo = q[..., int(np.flatnonzero(np.isclose(levels, a))[0])]
    hi = q[..., int(np.flatnonzero(np.isclose(levels, 1 - a))[0])]
    out = ((y >= lo) & (y <= hi)).astype(float)
    return np.where(np.isnan(y), np.nan, out)


def quantile_scores(
    y: np.ndarray, q: np.ndarray, levels: np.ndarray = QUANTILE_LEVELS
) -> dict[str, np.ndarray]:
    """Return WIS, log WIS (on log1p, quantiles floored at 0) and coverage.

    `y` is `[n, h]` and `q` `[n, h, K]` at `levels`.
    """
    y = np.asarray(y, float)
    shape = y.shape
    flat = q.reshape(-1, q.shape[-1])

    def wis(obs, quant):
        return wis_quantile(obs.ravel(), quant, levels)["wis"].reshape(shape)

    return {
        "wis": wis(y, flat),
        "log_wis": wis(np.log1p(y), np.log1p(np.clip(flat, 0, None))),
        "crps": np.full(shape, np.nan),
        **{f"coverage_{p}": _covers(y, q, levels, p) for p in (50, 90, 95)},
    }


def sample_scores(
    y: np.ndarray, draws: np.ndarray, levels: np.ndarray = QUANTILE_LEVELS
) -> dict[str, np.ndarray]:
    """Return sample CRPS, and WIS of the draws' quantiles, `[n, h]`.

    `draws` is `[S, n, h]`.
    """
    y = np.asarray(y, float)
    q = np.moveaxis(np.quantile(draws, levels, axis=0), 0, -1)
    out = quantile_scores(y, q, levels)
    crps = np.full(y.shape, np.nan)
    for i, j in zip(*np.nonzero(~np.isnan(y)), strict=True):
        x = draws[:, i, j]
        if np.isfinite(x).all():
            crps[i, j] = crps_sample(float(y[i, j]), x)
    out["crps"] = crps
    return out


def block_frame(
    block: Mapping, model: str, scores: Mapping[str, np.ndarray]
) -> pd.DataFrame:
    """Return a block's scores, one row per target row and step it has."""
    y = truth(block)
    rows, steps = np.nonzero(~np.isnan(y) & ~np.isnan(scores["wis"]))
    targets = np.array(row_targets(block))
    levels = np.array(block["levels"])
    out = pd.DataFrame(
        {
            "block": int(block["index"]),
            "layout": block["source"],
            "target": targets[rows],
            "level": levels[rows],
            "window": windows(block)[steps],
            "row": rows,
            "step": steps,
            "truth": y[rows, steps],
            "model": model,
            **{s: np.asarray(scores[s])[rows, steps] for s in SCORES},
        }
    )
    return out.reindex(columns=FRAME)


# Relative rows ---------------------------------------------------------


def reference_of(pairs: set[tuple[str, str]]) -> tuple[str, str]:
    """Return the (model, checkpoint) a cell is relative to."""
    if (BASELINE, ZERO_SHOT) in pairs:
        return (BASELINE, ZERO_SHOT)
    return (CHRONOS, ZERO_SHOT)


def _matched(cell: pd.DataFrame, n_pairs: int) -> pd.DataFrame:
    keys = ["block", "row", "step"]
    k = cell["model"] + "@" + cell["checkpoint"]
    n = cell.assign(_k=k).groupby(keys)["_k"].nunique()
    full = n.index[n == n_pairs]
    return cell.set_index(keys).loc[full].reset_index()


def relative_rows(scores: pd.DataFrame) -> pd.DataFrame:
    """Return mean scores by layout, target, level and window.

    `scores` holds every label's `block_frame`s with a `checkpoint`. In
    each (layout, target) cell only the target rows and steps every model
    and checkpoint scored are kept. Means are relative to randomwalknowcast
    where it forecast the cell, else to zero-shot Chronos-2 (`rel_to`).
    """
    out = []
    for (layout, target), cell in scores.groupby(["layout", "target"]):
        pairs = set(zip(cell["model"], cell["checkpoint"], strict=True))
        ref = reference_of(pairs)
        cell = _matched(cell, len(pairs))
        for level in ["all", *sorted(set(cell["level"]))]:
            for window in ("all", "nowcast", "forecast"):
                part = cell
                if level != "all":
                    part = part.loc[part["level"] == level]
                if window != "all":
                    part = part.loc[part["window"] == window]
                if part.empty or ref not in set(
                    zip(part["model"], part["checkpoint"], strict=True)
                ):
                    continue
                by = part.groupby(["model", "checkpoint"])
                mean = by[list(SCORES)].mean()
                mean["n"] = by.size()
                mean["n_blocks"] = by["block"].nunique()
                base = mean.loc[ref]
                mean["rel_wis"] = mean["wis"] / base["wis"]
                mean["rel_log_wis"] = mean["log_wis"] / base["log_wis"]
                out.append(
                    mean.reset_index().assign(
                        layout=layout,
                        target=target,
                        level=level,
                        window=window,
                        rel_to=ref[0] if ref[0] == BASELINE else ZERO_SHOT,
                    )
                )
    if not out:
        return pd.DataFrame(columns=ROWS)
    rows = pd.concat(out, ignore_index=True).assign(study="heldout")
    return rows.reindex(columns=ROWS)


# The unit --------------------------------------------------------------


Predictor = Callable[[dict], Any]


def chronos_predictor(model: Any) -> Predictor:
    """Return a block's native quantile levels and quantiles `[n, h, Q]`.

    The block is one batch, built and run as the trainer's held-out loss
    builds and runs it, so a series bias sees the rows' places.
    """
    import torch
    from chronos.chronos2.dataset import DatasetMode
    from chronos.chronos2.model import Chronos2Model

    from bdbvhubrenewal.forecasters import series_bias

    accepted = inspect.signature(Chronos2Model.forward).parameters
    native = np.asarray(model.chronos_config.quantiles, float)
    bias = series_bias.has_bias(model)

    def predict(block: dict) -> tuple[np.ndarray, np.ndarray]:
        plain = {k: v for k, v in block.items() if k not in ("meta",)}
        data = fs.ByLength([plain], 10**6, DatasetMode.VALIDATION, bias)
        batch = data._build_batch([0])
        with torch.no_grad():
            out = fs.forward(model, batch, accepted)
        n, h = int(block["n_targets"]), int(block["meta"]["horizon"])
        q = out.quantile_preds[:n, :, :h].float().cpu().numpy()
        return native, np.moveaxis(q, 1, -1)

    return predict


def hub_code(place: str) -> str:
    """Return a place's code in the hub's hierarchy (`CD-<place>`).

    The random walks read a place's parent from its first five characters
    (`data.snapshots.parent_code`), so `P01` is `CD-01`, `R1-P01` (a
    place in a region) `CD-R1-P01`, `P01-A1` `CD-01-A1` and the age total
    `A1` `CD-A1`; the country stays `CD`.
    """
    from bdbvhubrenewal.forecasters.base import NATIONAL

    if place == NATIONAL:
        return place
    return "CD-" + re.sub(r"^P(\d\d)", r"\1", place)


def walk_task(block: Mapping, n_samples: int = N_DRAWS) -> Any:
    """Return randomwalknowcast's task for a weekly block.

    The weekly and daily releases at the block's cut-offs, made from its
    counts as `finetune.training_block` made them (report batching and
    the country's counts replayed with the same generator), for the
    block's places in hub codes, over the block's horizon weeks.
    """
    from bdbvhubrenewal.forecasters import finetune as fin
    from bdbvhubrenewal.forecasters.base import ForecastTask

    meta = block["meta"]
    opts = meta["options"]
    counts = meta["counts"]
    if opts.get("batch_weekday") is not None:
        counts = fin.batch_reports(counts, opts["batch_weekday"])
    counts, country = fin.country_counts(
        counts, opts.get("parent", "exact"), copy.deepcopy(meta["rng"])
    )
    frames = {}
    for source in ("weekly", "daily"):
        rows = fin.releases(
            counts, source, meta["cutoffs"], opts.get("totals", False), country
        )
        rows = rows.assign(location=rows["location"].map(hub_code))
        rows.insert(0, "kind", "simulated")
        frames[source] = rows.reset_index(drop=True)
    cut = meta["cutoffs"][-1]
    ref = dt.date.fromisoformat(cut)
    ref += dt.timedelta(days=(5 - ref.weekday()) % 7)
    weekly = frames["weekly"]
    return ForecastTask(
        vintage=cut,
        reference_date=ref,
        kind="simulated",
        cutoff=cut,
        target=weekly.loc[weekly["vintage"] == cut].reset_index(drop=True),
        delays=None,
        locations=tuple(hub_code(p) for p in meta["places"]),
        target_end_dates=tuple(e.isoformat() for e in step_ends(block)),
        n_samples=n_samples,
        seed=WALK_SEED,
        target_history=weekly,
        daily_history=frames["daily"],
    )


def block_draws(block: Mapping, drawn: pd.DataFrame) -> np.ndarray:
    """Return draws `[S, n, h]` for the block's target rows and steps.

    A row or step the model did not forecast is NaN.
    """
    places = [hub_code(block["places"][i]) for i in block["target_places"]]
    weeks = [e.isoformat() for e in step_ends(block)]
    samples = sorted(set(drawn["sample"]))
    full = pd.MultiIndex.from_product(
        [samples, places, weeks], names=["sample", "location", "week"]
    )
    values = (
        drawn.assign(week=drawn["target_end_date"].astype(str))
        .groupby(["sample", "location", "week"])["value"]
        .first()
        .reindex(full)
        .to_numpy(float)
    )
    return values.reshape(len(samples), len(places), len(weeks))


def walk_predictor() -> Predictor:
    """Return randomwalknowcast's draws `[S, n, h]` for a weekly block."""
    from bdbvhubrenewal.forecasters import get_model

    model = get_model(BASELINE)

    def predict(block: dict) -> np.ndarray:
        drawn = model.forecast(walk_task(block))
        if "as_of" in drawn:
            drawn = drawn.loc[drawn["as_of"].isna()]
        return block_draws(block, drawn)

    return predict


def load_models(label: str, reference: bool) -> dict[str, Predictor]:
    """Return a label's models: the reference's, or the checkpoint's."""
    from bdbvhubrenewal.forecasters import foundation as fd

    checkpoint = None if reference else fd.resolve_checkpoint(label)
    model = fd._chronos(checkpoint).model.eval()
    models = {CHRONOS: chronos_predictor(model)}
    if reference:
        models[BASELINE] = walk_predictor()
    return models


def _chunks(blocks: Sequence[dict]) -> list[Sequence[dict]]:
    return [blocks[i : i + CHUNK] for i in range(0, len(blocks), CHUNK)]


def _chunk_path(cache: Path, label: str, key: str, k: int) -> Path:
    return Path(cache) / STUDY / label / f"{key}-{k:03d}.csv.gz"


def free_caches() -> None:
    """Free JAX's compiled functions and collect garbage."""
    if "jax" in sys.modules:
        sys.modules["jax"].clear_caches()
    gc.collect()


def score_block(
    block: dict, models: Mapping[str, Predictor]
) -> list[pd.DataFrame]:
    """Return each model's scores of one block."""
    from bdbvhubrenewal.forecasters.base import ModelUnavailableError

    out = []
    y = truth(block)
    for name, predict in models.items():
        if name == BASELINE and block["source"] not in {
            layout for layout, _ in BASELINE_CELLS
        }:
            continue
        try:
            got = predict(block)
        except ModelUnavailableError as err:
            print(f"{name} block {block['index']}: {err}", flush=True)
            continue
        if isinstance(got, tuple):
            native, q = got
            scores = quantile_scores(y, at_levels(q, native))
        else:
            scores = sample_scores(y, got)
        out.append(block_frame(block, name, scores))
    return out


def unit(
    cache: Path,
    label: str,
    models: Mapping[str, Predictor] | None = None,
    blocks: Sequence[dict] | None = None,
    reference: bool | None = None,
) -> None:
    """Score a label's models on the sampled blocks, a chunk a file.

    Finished chunks are read from the cache, so a rerun resumes.
    """
    blocks = prepare(cache) if blocks is None else blocks
    note = _sample(cache)
    assert note is not None
    key = note["key"]
    (Path(cache) / STUDY / label).mkdir(parents=True, exist_ok=True)
    chunks = _chunks(blocks)
    for k, chunk in enumerate(chunks):
        path = _chunk_path(cache, label, key, k)
        if path.exists():
            continue
        if models is None:
            models = load_models(label, bool(reference))
        frames = [f for b in chunk for f in score_block(b, models)]
        frame = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=FRAME)
        )
        tmp = path.with_name(path.name + ".tmp")
        frame.to_csv(tmp, index=False, compression="gzip")
        tmp.rename(path)
        free_caches()
        print(f"{STUDY} {label} chunk {k + 1}/{len(chunks)}", flush=True)
    print(f"{STUDY} {label} done", flush=True)


def unit_done(cache: Path, label: str) -> bool:
    """Whether every chunk of a label's unit is in the cache."""
    note = _sample(cache)
    if note is None:
        return False
    n = -(-int(note["n_blocks"]) // CHUNK)
    return all(
        _chunk_path(cache, label, note["key"], k).exists() for k in range(n)
    )


def label_scores(cache: Path, label: str) -> pd.DataFrame:
    """Return a label's cached scores."""
    note = _sample(cache)
    assert note is not None
    files = sorted(
        (Path(cache) / STUDY / label).glob(f"{note['key']}-*.csv.gz")
    )
    return pd.concat(
        [pd.read_csv(f, dtype={"model": str}) for f in files],
        ignore_index=True,
    )


def rows(cache: Path, labels: Sequence[str], reference: str) -> pd.DataFrame:
    """Return the finished labels' relative rows, none without the reference."""
    if reference not in labels:
        return pd.DataFrame(columns=ROWS)
    parts = [
        label_scores(cache, lb).assign(
            checkpoint=ZERO_SHOT if lb == reference else lb
        )
        for lb in labels
    ]
    return relative_rows(pd.concat(parts, ignore_index=True))
