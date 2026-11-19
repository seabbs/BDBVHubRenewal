# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Fine-tune Chronos-2 in one pass while the outbreaks are simulated.

The training outbreaks are taken in order of seed and index, in chunks of
`CHUNK` outbreaks. A chunk is trained on as soon as every simulation
batch it needs has finished (`scripts/finetune_simulate.R`), and the
trainer waits for it otherwise. Within a chunk the blocks
(`finetune_chronos.outbreak_blocks`, seeded per outbreak) are shuffled
and packed by size (`finetune_chronos.plan_batches`). The learning rate
falls linearly to zero over the run's `--outbreaks` simulations
(`sims_done`), so the checkpoint after the first N outbreaks is
Chronos-2 fine-tuned on those N in one pass of that schedule. Each is
saved in `OUT/n<N>` with its consumed order and settings hash, and
published as the release `chronos-ft-<hash>` when `--publish` is set.
Held-out loss is on the first `--holdout` outbreaks of the held-out
seeds, simulated first.

    uv run --group chronos python scripts/finetune_stream.py SIM_DIR OUT \
        --grid r3 --train-seeds FIRST LAST --held-out-seeds FIRST LAST \
        --publish

`scripts/finetune_driver.sh` runs it with the seeds of a config such as
`config/finetune/r3.env`.

`OUT/stream-status.json` holds the trainer's state, and while
`OUT/../PAUSE` exists the trainer waits between steps.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import multiprocessing
import os
import pickle
import shutil
import subprocess
import sys
import time
import zlib
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from bdbvhubrenewal.forecasters import series_bias
from bdbvhubrenewal.forecasters.foundation import (
    CHRONOS_MODEL,
    CHRONOS_REVISION,
    SLOTS_PREFIX,
    release_tag,
    torch_device,
)

sys.path.insert(0, str(Path(__file__).parent))
import finetune_chronos as ft
import finetune_verify

CHUNK = 500
"""Outbreaks per chunk."""

LEVEL_BLOCKS = 300
"""Held-out blocks scored by level at each checkpoint."""

SAVE_EVERY = 50
"""Steps between saves of the trainer's state, which a restarted trainer
resumes from."""
BY_SOURCE_EVERY = 200
"""Steps between held-out losses by source, which cost three monitor
evals (the default of `--by-source-every`)."""

EVAL_MODES = ("local", "offload")
"""Where held-out evaluation runs (`--eval`)."""
HELD_OUT_BLOCKS = "held-out-blocks.pkl"
"""The held-out blocks an offloaded run saves for its evaluator."""

POLL = 60
"""Seconds between checks for a chunk's simulations."""


REPO = "seabbs/bdbv-hub-modelling"
"""Repository of the checkpoint releases."""


def log(message: str) -> None:
    """Print a timestamped line."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{stamp} [train] {message}", flush=True)


def read_ids(path: Path) -> pd.Series:
    """Return a batch file's outbreak ids; none if every draw failed."""
    if path.stat().st_size == 0:
        return pd.Series([], dtype=str)
    try:
        return pd.read_csv(path, usecols=["outbreak"])["outbreak"]
    except pd.errors.EmptyDataError:
        return pd.Series([], dtype=str)


def real_share(real: list, chunk: int, chunks: int) -> list:
    """Return the real-data blocks trained with chunk `chunk` of `chunks`.

    The blocks are spread evenly and in order over the round's chunks, so
    real data is seen throughout training, not only at its end.
    """
    return real[chunk::chunks]


BLOCK_KEYS = ("context", "n_targets", "prediction_length", "levels")
"""Keys every block the trainer reads must have."""


def load_blocks(path: Path) -> list[dict]:
    """Return the blocks pickled at `path`, refusing stale ones.

    A block cut by older code may lack a key the trainer reads only at a
    checkpoint, hours later; this fails at once instead.

    Raises
    ------
    ValueError
        If a block lacks one of `BLOCK_KEYS`.
    """
    with path.open("rb") as f:
        blocks = pickle.load(f)
    for i, block in enumerate(blocks):
        missing = [k for k in BLOCK_KEYS if k not in block]
        if missing:
            raise ValueError(
                f"{path} block {i} has no {', '.join(missing)}: cut it again"
            )
    return blocks


def read_seed_list(path: Path) -> list[int]:
    """Return the training seeds a file lists, one a line, in its order."""
    return [int(x) for x in path.read_text().split()]


LORA_TARGETS = (
    "self_attention.q",
    "self_attention.v",
    "self_attention.k",
    "self_attention.o",
    "output_patch_embedding.output_layer",
)
"""Modules the LoRA adapter trains."""


BIAS_CODE = ("src/bdbvhubrenewal/forecasters/series_bias.py",)
"""The series bias's code and its parent rule, hashed into its settings."""


def lora_config(bias: bool):
    """Return the LoRA config, saving the series bias if `bias`.

    The bias (`bdbvhubrenewal.forecasters.series_bias`) trains beside the
    adapter as one of PEFT's `modules_to_save`, so it is saved in the
    adapter's files and published and loaded with it.
    """
    from peft import LoraConfig

    return LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=list(LORA_TARGETS),
        modules_to_save=[series_bias.BIAS_MODULE] if bias else None,
    )


def bias_settings(bias: bool, lr: float = 0.0) -> dict:
    """Return the series bias's entry of a checkpoint's settings."""
    if not bias:
        return {}
    return {
        "series_bias": {
            "pairs": "same place (count and lag series); parent and child "
            "either way round",
            "per": "encoder layer and head, added to the group attention "
            "scores",
            "init": "zero",
            "learning_rate": lr,
            "parent_rule": "nearest '-' prefix in the block, else the "
            f"country ({', '.join(series_bias.COUNTRIES)})",
            "module": series_bias.BIAS_MODULE,
        }
    }


def param_groups(model, lr: float, bias_lr: float = 0.0) -> list[dict]:
    """Return the optimiser's groups: the adapter at `lr`, the bias at its own.

    Each keeps its starting rate as `base_lr`, which `set_learning_rates`
    decays. A few hundred bias scalars moving at Adam's step of about `lr`
    would reach only about 0.2 by the end at 1e-5, small against the
    attention scores, so they have their own rate.
    """
    adapter, bias = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (bias if series_bias.BIAS_MODULE in name else adapter).append(p)
    groups = [{"params": adapter, "lr": lr, "base_lr": lr, "name": "lora"}]
    if bias:
        groups.append(
            {"params": bias, "lr": bias_lr, "base_lr": bias_lr, "name": "bias"}
        )
    return groups


def make_optimiser(model, lr: float, bias_lr: float = 0.0):
    """Return AdamW over `param_groups`, with no weight decay in any group.

    As Chronos-2's own fit (the HF Trainer's default weight decay of 0,
    a linear schedule and no warmup).
    """
    import torch

    return torch.optim.AdamW(param_groups(model, lr, bias_lr), weight_decay=0.0)


def group_rates(optimiser) -> dict:
    """Return each optimiser group's current rate by its name."""
    return {g["name"]: g["lr"] for g in optimiser.param_groups}


def set_learning_rates(optimiser, sims: float, outbreaks: int) -> None:
    """Set each group's rate at `sims` simulations of `outbreaks`."""
    for group in optimiser.param_groups:
        group["lr"] = learning_rate(sims, group["base_lr"], outbreaks)


def lora_model(base, config, init: Path | None):
    """Return `base` with a LoRA adapter to train.

    A fresh adapter from `config`, or, to continue a round, the adapter
    saved at `init`, still trainable. A base with the series bias
    continues only from an adapter that holds it.
    """
    from peft import PeftModel, get_peft_model

    if init is None:
        model = get_peft_model(base, config)
    else:
        if series_bias.has_bias(base) and not series_bias.adapter_has_bias(
            init
        ):
            raise ValueError(f"{init} holds no series bias to continue")
        model = PeftModel.from_pretrained(base, init, is_trainable=True)
    return model


def learning_rate(sims: float, base: float, outbreaks: int) -> float:
    """Return the learning rate once `sims` simulations are trained.

    It falls linearly from `base` to zero over the run's `outbreaks`
    simulations, so it reaches zero as the last simulation's last batch
    is trained, however the blocks are packed into steps.
    """
    return base * max(0.0, 1 - sims / outbreaks)


def sims_done(done: int, size: int, position: int, steps: int) -> float:
    """Return the simulations trained before step `position` of a chunk.

    `done` simulations before the chunk of `size`, planned as `steps`
    steps: within the chunk, its share of steps taken, so the rate falls
    smoothly.
    """
    return done + size * position / steps


def should_stop(
    losses: list[float], done: int, stop_from: int, min_gain: float
) -> bool:
    """Return whether the last doubling cut held-out loss by too little."""
    if done < stop_from or len(losses) < 2:
        return False
    return losses[-2] - losses[-1] < min_gain * losses[-2]


def simulation_ids(outbreaks: pd.Series) -> set[str]:
    """Return the simulations of outbreak draws `<seed>-<index>[-<draw>]`."""
    parts = outbreaks.astype(str).str.split("-")
    return set(parts.str[0] + "-" + parts.str[1])


def quarantine(sim_dir: Path, seed: int, reason: str) -> None:
    """Move a bad batch's files aside, so the driver runs it again."""
    where = sim_dir / "bad" / f"{seed}-{int(time.time())}"
    where.mkdir(parents=True, exist_ok=True)
    for path in (
        sim_dir / f"counts-{seed}.csv.gz",
        sim_dir / f"settings-{seed}.csv",
        sim_dir / f"failures-{seed}.csv",
        sim_dir / "logs" / f"{seed}.log",
    ):
        if path.exists():
            path.rename(where / path.name)
    log(f"batch {seed} {reason}: moved to {where} to be run again")


# Outbreaks of each finished batch, by simulation folder and seed.
_READY: dict[tuple[Path, int], list[str]] = {}


def seed_done(sim_dir: Path, seed: int) -> bool:
    """Return whether a simulation batch has finished and is whole.

    A batch whose counts and settings hold different simulations is moved
    aside (`quarantine`), so it counts as not done and is run again.

    Raises
    ------
    RuntimeError
        If the batch failed.
    """
    if (sim_dir, seed) in _READY:
        return True
    counts = sim_dir / f"counts-{seed}.csv.gz"
    settings = sim_dir / f"settings-{seed}.csv"
    if not (counts.exists() and settings.exists()):
        log_file = sim_dir / "logs" / f"{seed}.log"
        if log_file.exists() and "Execution halted" in log_file.read_text():
            raise RuntimeError(f"simulation batch {seed} failed: {log_file}")
        return False
    a = simulation_ids(read_ids(settings))
    b = simulation_ids(read_ids(counts))
    if a != b:
        quarantine(sim_dir, seed, "has counts and settings that disagree")
        return False
    return True


EXCLUDE_FILE: Path | None = None
"""Simulations the sim QC excluded (`--exclude`), one a line or a CSV
with a `simulation` column. It is read again before each new batch, as a
continuous run's rounds bring their exclusions with them."""


def excluded() -> set[str]:
    """Return the simulations in `EXCLUDE_FILE`, if any."""
    if EXCLUDE_FILE is None or not EXCLUDE_FILE.exists():
        return set()
    lines = [x.strip() for x in EXCLUDE_FILE.read_text().splitlines()]
    if lines and lines[0].startswith("simulation"):
        lines = [x.split(",")[0] for x in lines[1:]]
    return {x for x in lines if x}


def batch_outbreaks(sim_dir: Path, seed: int) -> list[str] | None:
    """Return a finished batch's outbreaks in order, or None if not done.

    A finished batch is read once and remembered.
    """
    if (sim_dir, seed) not in _READY:
        if not seed_done(sim_dir, seed):
            return None
        ids = read_ids(sim_dir / f"settings-{seed}.csv")
        skip = excluded()
        ids = [i for i in ids if i not in skip]
        _READY[sim_dir, seed] = sorted(ids, key=ft._order)
    return _READY[sim_dir, seed]


def ready_outbreaks(sim_dir: Path, seeds: range) -> list[str]:
    """Return the outbreaks of the finished batches, in order.

    The batches are read in seed order up to the first not yet finished.
    """
    out: list[str] = []
    for seed in seeds:
        ids = batch_outbreaks(sim_dir, seed)
        if ids is None:
            break
        out.extend(ids)
    return out


def next_chunk(
    sim_dir: Path,
    seeds: range,
    trained: list[str],
    size: int,
) -> list[str] | None:
    """Return the next `size` outbreaks to train on, or None if not ready.

    They are the first outbreaks not yet `trained` of the finished
    batches in seed order, up to the first batch not finished, so a
    chunk waits for a missing batch and every run trains in seed order.
    """
    seen = set(trained)
    out: list[str] = []
    for seed in seeds:
        ids = batch_outbreaks(sim_dir, seed)
        if ids is None:
            break
        out.extend(o for o in ids if o not in seen)
        if len(out) >= size:
            return out[:size]
    return None


DRAWS_PER_PASS: int | None = None
"""Observation draws of each simulation a pass trains on; None is all."""

SUBSET_SEED = 0
"""Seed of the draws a pass takes (`snapshots`)."""


def snapshots(
    draws: list[str],
    rng: np.random.Generator,
    n_draws: int | None = None,
    subset_seed: int = 0,
) -> list[tuple]:
    """Return the draws cut at each of `CUTS` snapshots, with a source each.

    Every observation draw is cut once at each snapshot, so a simulation
    of 10 draws gives 160 blocks; each block's source is drawn from
    `SOURCES` at its share (`SOURCE_SHARES`). With `n_draws`, a pass cuts
    only that many draws at each snapshot, sampled across all the draws
    by `subset_seed`, the simulation and the snapshot, so a later pass
    with another seed can take the others.
    """
    simulation = draws[0].rsplit("-", 1)[0] if draws else ""
    shares = np.array([ft.SOURCE_SHARES[s] for s in ft.SOURCES])
    shares = shares / shares.sum()
    out = []
    for j in range(ft.CUTS):
        chosen = range(len(draws))
        if n_draws is not None and n_draws < len(draws):
            chosen = sorted(
                np.random.default_rng(
                    [subset_seed, zlib.crc32(simulation.encode()), j]
                ).choice(len(draws), n_draws, replace=False)
            )
        for i in chosen:
            out.append((draws[i], str(rng.choice(ft.SOURCES, p=shares)), j))
    return out


def cut(
    sim_dir: Path,
    ids: list[str],
    seed: int,
    per_simulation: int,
    empty: list[str] | None = None,
) -> list[tuple[str, dict]]:
    """Return a sample of blocks of each simulation, with its draw.

    A simulation's candidate blocks are each of its observation draws
    (`<id>-<draw>`) cut in each of `CUTS` slots of each source; a sample of
    `per_simulation` of them, seeded by the simulation, is cut. Blocks
    skipped for having no rows are described in `empty`.
    """
    seeds = sorted({ft._order(o)[0] for o in ids})
    counts = pd.concat(
        [pd.read_csv(sim_dir / f"counts-{s}.csv.gz") for s in seeds]
    )
    parts = counts["outbreak"].str.split("-")
    counts["simulation"] = parts.str[0] + "-" + parts.str[1]
    settings = pd.concat(
        [pd.read_csv(sim_dir / f"settings-{s}.csv") for s in seeds]
    ).set_index("outbreak")
    by = dict(tuple(counts.groupby("simulation")))
    out = []
    for simulation in ids:
        row = settings.loc[simulation]
        days = row.get("max_delay", np.nan)
        weeks = ft.SIM_DELAY_WEEKS if pd.isna(days) else int(days) // 7
        draws = dict(tuple(by[simulation].groupby("outbreak")))
        rng = np.random.default_rng([seed, zlib.crc32(simulation.encode())])
        candidates = snapshots(sorted(draws), rng, DRAWS_PER_PASS, SUBSET_SEED)
        k = min(per_simulation, len(candidates))
        chosen = [candidates[i] for i in rng.choice(len(candidates), k, False)]
        for outbreak in sorted(draws):
            picks = {(s, j) for o, s, j in chosen if o == outbreak}
            if not picks:
                continue
            draw_rng = np.random.default_rng(
                [seed, zlib.crc32(outbreak.encode())]
            )
            blocks = ft.outbreak_blocks(
                draws[outbreak],
                ft.strata(row),
                draw_rng,
                weeks,
                picks=picks,
                empty=empty,
            )
            out.extend((outbreak, block) for block in blocks)
    return out


def init_worker(
    max_series: int,
    lags: dict[str, tuple[int, ...]],
    sources: tuple[str, ...],
    univariate: float = ft.UNIVARIATE_SHARE,
    draws: int | None = None,
    subset_seed: int = 0,
    grid: str = ft.DESIGN,
) -> None:
    """Give a cutting worker the trainer's block settings."""
    global DRAWS_PER_PASS, SUBSET_SEED
    ft.use_design(grid)
    ft.MAX_SERIES, ft.MAX_LAGS, ft.SOURCES = max_series, lags, sources
    ft.UNIVARIATE_SHARE = univariate
    DRAWS_PER_PASS, SUBSET_SEED = draws, subset_seed


def _cut_part(
    sim_dir: Path, ids: list[str], seed: int, per_simulation: int
) -> tuple[list[tuple[str, dict]], list[str]]:
    """Return `cut` of some simulations and the empty blocks skipped."""
    empty: list[str] = []
    return cut(sim_dir, ids, seed, per_simulation, empty), empty


def cut_in_workers(
    pool: Executor | None,
    sim_dir: Path,
    ids: list[str],
    seed: int,
    per_simulation: int,
    parts: int,
) -> tuple[list[tuple[str, dict]], list[str]]:
    """Return `cut` of the simulations `ids`, in `parts` run by `pool`.

    Each simulation's blocks are seeded by it alone, so the parts, joined
    in order, are the blocks `cut` gives for all of them at once. Without
    a pool the parts run here.
    """
    return _join(submit_cut(pool, sim_dir, ids, seed, per_simulation, parts))


def submit_cut(
    pool: Executor | None,
    sim_dir: Path,
    ids: list[str],
    seed: int,
    per_simulation: int,
    parts: int,
) -> list[Future]:
    """Start cutting the simulations `ids` in `parts` (`cut_in_workers`)."""
    split = np.array_split(np.arange(len(ids)), max(1, min(parts, len(ids))))
    futures = []
    for part in split:
        args = (sim_dir, [ids[i] for i in part], seed, per_simulation)
        if pool is None:
            future: Future = Future()
            future.set_result(_cut_part(*args))
        else:
            future = pool.submit(_cut_part, *args)
        futures.append(future)
    return futures


def _join(
    futures: list[Future],
) -> tuple[list[tuple[str, dict]], list[str]]:
    """Return the blocks and empty blocks of cut parts, in order."""
    pairs: list[tuple[str, dict]] = []
    empty: list[str] = []
    for future in futures:
        part, skipped = future.result()
        pairs.extend(part)
        empty.extend(skipped)
    return pairs, empty


def forward(model, batch: dict, accepted, device=None):
    """Run the model on a batch, its rows given their places if any."""
    import torch

    inputs = {
        k: v.to(device or model.device) if torch.is_tensor(v) else v
        for k, v in batch.items()
        if k in accepted
    }
    with series_bias.assigned(model, batch.get("series_rows")):
        return model(**inputs)


def train_step(model, batch: dict, accepted, device, share: float) -> float:
    """Accumulate a batch's gradients, scaled by `share`; return its loss.

    Forward and backward both run with the rows' places assigned, so a
    block recomputed in backward (`checkpoint_large_batches`) gets the
    same series bias.
    """
    with series_bias.assigned(model, batch.get("series_rows")):
        loss = forward(model, batch, accepted, device).loss
        (loss * share).backward()
    return float(loss.detach())


def bias_summary(model) -> dict:
    """Return each series bias pair's values, by layer and head, and range.

    Empty for a model without the bias.
    """
    out = {}
    for name, p in model.named_parameters():
        pair = name.rsplit(".", 1)[-1]
        if series_bias.BIAS_MODULE not in name or pair not in series_bias.PAIRS:
            continue
        if not p.requires_grad:
            continue
        values = p.detach().float().cpu()
        out[pair] = {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
            "values": values.tolist(),
        }
    return out


def _bias_range(summary: dict) -> str:
    """Describe the bias pairs' means and ranges for the log."""
    return ", ".join(
        f"{k} {v['mean']:+.3f} [{v['min']:+.3f}, {v['max']:+.3f}]"
        for k, v in summary.items()
    )


def block_rows(blocks: list[dict]) -> np.ndarray:
    """Return the places and parents of the rows of blocks batched together.

    A block's targets are at its `target_places` (a place's daily counts
    and its period totals alike), then each place's lags
    (`series_bias.block_rows`); each block's places are its own
    (`series_bias.batch_rows`).
    """
    return series_bias.batch_rows(
        [
            series_bias.block_rows(
                b.get("target_places", range(int(b["n_targets"]))),
                int(b["n_covariates"]),
                block_parents(b),
            )
            for b in blocks
        ]
    )


def block_parents(block: dict) -> list[int]:
    """Return each place's parent in a block (`series_bias.parents_of`).

    A block without place names links every place to the country, when
    it holds it, by its levels.
    """
    if "places" in block:
        return series_bias.parents_of(block["places"])
    levels = list(block["levels"])
    root = levels.index("national") if "national" in levels else -1
    return [-1 if i == root else root for i in range(len(levels))]


def tensors(block: dict) -> dict:
    """Return a block with its arrays as the tensors Chronos-2 reads."""
    import torch

    context = torch.from_numpy(block["context"])
    return {
        **block,
        "context": context,
        "future_covariates": context[:, ft.CONTEXT :],
    }


def checkpoint_large_batches(above: int) -> None:
    """Recompute encoder activations in backward for large batches.

    Chronos-2's group attention runs over every series of a batch, so its
    memory grows with their square. Above `above` series each encoder block
    is checkpointed (`torch.utils.checkpoint`).
    """
    import functools

    import torch
    from chronos.chronos2.model import Chronos2EncoderBlock
    from torch.utils.checkpoint import checkpoint

    forward = Chronos2EncoderBlock.forward

    def checkpointed(self, hidden_states, **kwargs):
        large = hidden_states.shape[0] > above
        if large and self.training and torch.is_grad_enabled():
            return checkpoint(
                functools.partial(forward, self, **kwargs),
                hidden_states,
                use_reentrant=False,
            )
        return forward(self, hidden_states, **kwargs)

    Chronos2EncoderBlock.forward = checkpointed


class ByLength:
    """Chronos-2 datasets of blocks, one for each prediction length.

    `_build_batch` takes indices of blocks that share a length
    (`finetune_chronos.plan_by_length`), as `Chronos2Dataset` does.
    """

    def __init__(
        self,
        blocks: list[dict],
        batch_size: int,
        mode,
        bias: bool = False,
    ) -> None:
        from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode

        # With the series bias, each batch carries its rows' places and
        # parents (`block_rows`).
        self.bias = bias
        self.batch_size = batch_size
        self.inputs = [tensors(b) for b in blocks]
        self.lengths = [b["prediction_length"] for b in blocks]
        self.local: dict[int, int] = {}
        self.data = {}
        for length in sorted(set(self.lengths)):
            idx = [i for i, n in enumerate(self.lengths) if n == length]
            self.local |= {i: j for j, i in enumerate(idx)}
            self.data[length] = Chronos2Dataset(
                inputs=[self.inputs[i] for i in idx],
                context_length=ft.CONTEXT,
                prediction_length=length,
                batch_size=batch_size,
                output_patch_size=ft.OUTPUT_PATCH,
                mode=mode,
                **(
                    {"min_past": ft.CONTEXT}
                    if mode == DatasetMode.TRAIN
                    else {}
                ),
            )

    def _build_batch(self, indices: list[int]) -> dict:
        """Return the batch of blocks `indices`, all of one length."""
        length = self.lengths[indices[0]]
        batch = self.data[length]._build_batch([self.local[i] for i in indices])
        if self.bias:
            batch["series_rows"] = block_rows([self.inputs[i] for i in indices])
        return batch

    def indices(self):
        """Yield every block's index once, in batches of one length.

        Batches are those Chronos-2's sequential batches make: blocks in
        order until a batch holds `batch_size` series.
        """
        for length in self.data:
            idx = [i for i, n in enumerate(self.lengths) if n == length]
            batch, rows = [], 0
            for i in idx:
                batch.append(i)
                rows += int(self.inputs[i]["context"].shape[0])
                if rows >= self.batch_size:
                    yield batch
                    batch, rows = [], 0
            if batch:
                yield batch

    def batches(self):
        """Yield every block once, in batches of one length."""
        for indices in self.indices():
            yield self._build_batch(indices)


def held_out_loss(model, blocks: list[dict], batch_size: int) -> float:
    """Return the mean quantile loss of a Chronos-2 model on blocks."""
    import torch
    from chronos.chronos2.dataset import DatasetMode
    from chronos.chronos2.model import Chronos2Model

    data = ByLength(
        blocks, batch_size, DatasetMode.VALIDATION, series_bias.has_bias(model)
    )
    accepted = inspect.signature(Chronos2Model.forward).parameters
    training = model.training
    model.eval()
    losses, weights = [], []
    with torch.no_grad():
        for indices in data.indices():
            batch = data._build_batch(indices)
            losses.append(float(forward(model, batch, accepted).loss))
            weights.append(len(batch["context"]))
    model.train(training)
    return float(np.average(losses, weights=weights))


def free_memory() -> None:
    """Return the GPU memory PyTorch caches but does not use."""
    import torch

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def settle_heap() -> None:
    """Collect garbage, then leave what survives out of later collections.

    `free_memory` collects every step, and each collection walks every
    object Python tracks: a chunk's blocks, the held-out blocks and the
    model's modules. Frozen (`gc.freeze`), they are skipped, while the
    step's own garbage is still collected and reference counting still
    frees them. Called once a chunk, it first unfreezes, so cycles left by
    the chunk before are collected then.
    """
    gc.unfreeze()
    gc.collect()
    gc.freeze()


def memory() -> str:
    """Describe the GPU memory in use."""
    import torch

    if torch.backends.mps.is_available():
        used = torch.mps.current_allocated_memory() / 1e9
        driver = torch.mps.driver_allocated_memory() / 1e9
        return f"MPS allocated {used:.1f} GB, driver {driver:.1f} GB"
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        return f"CUDA allocated {used:.1f} GB"
    return "CPU"


def publish(where: Path, key: str, run=subprocess.run) -> None:
    """Publish a checkpoint as the release `chronos-ft-<key>`.

    Safe to run again: a release whose sha256sums match is left as it is,
    one missing assets gets them uploaded, and one whose sums differ is an
    error.
    """
    tag = release_tag(key)
    files = [
        where / "finetuned-ckpt" / "adapter_config.json",
        where / "finetuned-ckpt" / "adapter_model.safetensors",
        where / "settings.json",
        where / "outbreaks.txt",
    ]
    sums = where / "sha256sums.txt"
    sums.write_text("".join(f"{ft._sha256(f)}  {f.name}\n" for f in files))
    files.append(sums)

    def gh(*command: str) -> subprocess.CompletedProcess:
        return run(
            ["gh", "release", *command, "-R", REPO],
            capture_output=True,
            text=True,
        )

    view = gh("view", tag, "--json", "assets")
    if view.returncode != 0:
        n = json.loads((where / "settings.json").read_text())["n_outbreaks"]
        result = gh(
            "create",
            tag,
            "--latest=false",
            "--title",
            f"Chronos-2 fine-tuned checkpoint {key} ({n} outbreaks)",
            "--notes",
            f"LoRA adapter of Chronos-2 for the _ft models, fine-tuned in one "
            f"pass on {n} simulated outbreaks (scripts/finetune_stream.py). "
            f"Check the files with sha256sums.txt.\n\nThis was opened by a "
            f"bot. Please ping @seabbs for any questions.",
            *map(str, files),
        )
        if result.returncode != 0:
            raise RuntimeError(f"could not publish {tag}: {result.stderr}")
        log(f"published {tag}: {result.stdout.strip()}")
        return
    present = {a["name"] for a in json.loads(view.stdout)["assets"]}
    if sums.name in present:
        theirs = gh("download", tag, "-p", sums.name, "-O", "-").stdout
        if theirs != sums.read_text():
            raise RuntimeError(f"{tag} exists and its sha256sums differ")
    missing = [f for f in files if f.name not in present]
    if not missing:
        log(f"{tag} already published")
        return
    result = gh("upload", tag, *map(str, missing))
    if result.returncode != 0:
        raise RuntimeError(f"could not upload to {tag}: {result.stderr}")
    log(f"uploaded {[f.name for f in missing]} to {tag}")


def accumulate(indices: list[int], backward, zero, skip) -> float:
    """Return a batch's loss, accumulating its gradients by `backward`.

    The gradients are clear when it starts (`finite_step`). If memory runs
    out, they are cleared (`zero`) and the batch is run again in halves,
    and so on; a single block that still does not fit is left out
    (`skip`). Every other block is used once.
    """
    parts = [indices]
    retry = False
    while True:
        if retry:
            zero()
        retry = True
        values, sizes = [], []
        try:
            for part in parts:
                values.append(backward(part, len(part) / len(indices)))
                sizes.append(len(part))
            return (
                float(np.average(values, weights=sizes)) if values else np.nan
            )
        except RuntimeError as error:
            if "out of memory" not in str(error):
                raise
            if len(part) == 1:
                skip(part[0])
                parts = [p for p in parts if p is not part]
            else:
                half = len(part) // 2
                at = parts.index(part)
                parts[at : at + 1] = [part[:half], part[half:]]


def finite_step(params: list, optimiser, value: float) -> bool:
    """Clip and apply the gradients unless any is not finite.

    The step is skipped if the loss or a gradient is not finite, and the
    gradients are cleared either way.

    Returns
    -------
    bool
        Whether the step was applied.
    """
    import torch

    grads = [p.grad for p in params if p.grad is not None]
    # One check on the device, read back once, not once a tensor.
    finite = (
        np.isfinite(value)
        and bool(grads)
        and bool(torch.stack([torch.isfinite(g).all() for g in grads]).all())
    )
    if finite:
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimiser.step()
    optimiser.zero_grad()
    return finite


def held_out_by_source(
    model, blocks: list[dict], batch_size: int, n: int | None = None
) -> dict:
    """Return held-out loss by source (weekly, daily, 3day).

    Each source's loss is on its first `n` blocks, or all of them, so a
    source's curve compares across runs that cut it alike.
    """
    by: dict[str, list[dict]] = {}
    for block in blocks:
        by.setdefault(block["source"], []).append(block)
    return {
        source: held_out_loss(model, group[:n], batch_size)
        for source, group in sorted(by.items())
    }


def _by_source(losses: dict) -> str:
    """Describe losses by source for the log."""
    return ", ".join(f"{s} {v:.4f}" for s, v in losses.items())


def held_out_by_level(model, blocks: list[dict], n: int) -> dict:
    """Return held-out loss by level (national, intermediate, leaf).

    Each of the first `n` blocks is scored once per level it holds, with
    the other rows' targets masked, so each level's loss is on its own
    series in the full block.
    """
    import torch
    from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode
    from chronos.chronos2.model import Chronos2Model

    accepted = inspect.signature(Chronos2Model.forward).parameters
    bias = series_bias.has_bias(model)
    training = model.training
    model.eval()
    losses: dict[str, list[float]] = {}
    with torch.no_grad():
        for block in blocks[:n]:
            data = Chronos2Dataset(
                inputs=[tensors(block)],
                context_length=ft.CONTEXT,
                prediction_length=block["prediction_length"],
                batch_size=1,
                output_patch_size=ft.OUTPUT_PATCH,
                mode=DatasetMode.VALIDATION,
            )
            batch = next(iter(data))
            if bias:
                batch["series_rows"] = block_rows([block])
            levels = np.array(block["levels"])
            for level in sorted(set(levels)):
                target = batch["future_target"].clone()
                keep = torch.zeros(len(target), dtype=torch.bool)
                keep[: len(levels)] = torch.from_numpy(levels == level)
                target[~keep] = float("nan")
                loss = float(
                    forward(
                        model, {**batch, "future_target": target}, accepted
                    ).loss
                )
                losses.setdefault(level, []).append(loss)
    model.train(training)
    return {k: float(np.mean(v)) for k, v in losses.items()}


def skipped_lines(skipped: list[dict]) -> list[str]:
    """Return the lines listing skipped blocks in a release's outbreaks."""
    return [
        f"# skipped {b['outbreak']} at step {b['step']}, {b['series']} series"
        for b in skipped
    ]


def check_until(until: int | None, prefixes: list[int]) -> None:
    """Raise if `--until` is not one of the run's prefixes."""
    if until is not None and until not in prefixes:
        raise ValueError(f"--until {until} is not a prefix of {prefixes}")


def until_start(done: int, until: int | None, have_prefix: bool) -> str:
    """Return what a segment does with the state it resumes from.

    "train" while the state is short of `until`; "kept" if it is at
    `until` and prefix `until` is saved, so only the state is kept with
    it. A state past `until`, or at it without the prefix, cannot make
    that prefix again, so the segment stops with an error rather than
    train past it.
    """
    if until is None or done < until:
        return "train"
    if done > until:
        raise SystemExit(
            f"the state is at {done} outbreaks, past n{until}: that prefix "
            "cannot be made from it"
        )
    if not have_prefix:
        raise SystemExit(
            f"the state is at n{until} but n{until} is missing; retrain it "
            "from the state of the prefix before"
        )
    return "kept"


def keep_state(state: Path, where: Path) -> None:
    """Copy the trainer's state to `where/state`, replacing any there."""
    kept = where / "state"
    shutil.rmtree(kept, ignore_errors=True)
    shutil.copytree(state, kept)


def save_rng(where: Path) -> None:
    """Save torch's random state (CPU and any GPU) to `where/rng.pt`.

    Chronos-2 trains with dropout, so a resumed run matches the run that
    did not stop only with the random state it stopped at.
    """
    import torch

    rng = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        rng["mps"] = torch.mps.get_rng_state()
    torch.save(rng, where / "rng.pt")


def load_rng(where: Path) -> bool:
    """Restore torch's random state from `where/rng.pt`, if saved."""
    import torch

    path = where / "rng.pt"
    if not path.exists():
        return False
    rng = torch.load(path, weights_only=True)
    torch.set_rng_state(rng["cpu"])
    if "cuda" in rng and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    if "mps" in rng and torch.backends.mps.is_available():
        torch.mps.set_rng_state(rng["mps"])
    return True


def save_state(where: Path, model, optimiser, state: dict) -> None:
    """Save the trainer's state: adapter, optimiser and position."""
    import torch
    from peft import get_peft_model_state_dict

    tmp = where.with_suffix(".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    torch.save(get_peft_model_state_dict(model), tmp / "adapter.pt")
    torch.save(optimiser.state_dict(), tmp / "optimiser.pt")
    (tmp / "state.json").write_text(json.dumps(state))
    save_rng(tmp)
    # The last state is kept until the new one is in place.
    old = where.with_suffix(".old")
    if where.exists():
        shutil.rmtree(old, ignore_errors=True)
        where.rename(old)
    tmp.rename(where)
    shutil.rmtree(old, ignore_errors=True)


def load_state(where: Path, model, optimiser) -> dict | None:
    """Load the trainer's saved state into the model and optimiser.

    If a save stopped between its renames, the last state is in `.old`.
    """
    import torch
    from peft import set_peft_model_state_dict

    if not (where / "state.json").exists():
        where = where.with_suffix(".old")
    if not (where / "state.json").exists():
        return None
    set_peft_model_state_dict(
        model, torch.load(where / "adapter.pt", weights_only=True)
    )
    optimiser.load_state_dict(
        torch.load(where / "optimiser.pt", weights_only=True)
    )
    # A state saved without the random state resumes without it.
    load_rng(where)
    return json.loads((where / "state.json").read_text())


def code_commit() -> str:
    """Return the commit of the repository holding this script."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


BIAS_LR = 1e-3
"""The series bias's starting learning rate, decaying to zero with the
adapter's."""


def build_parser() -> argparse.ArgumentParser:
    """Return the trainer's command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sim_dir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--train-seeds", type=int, nargs=2)
    # A round: its training seeds, one a line, in place of --train-seeds,
    # and the adapter it continues from.
    parser.add_argument("--seed-list", type=Path)
    parser.add_argument("--init-adapter", type=Path)
    # The seed list grows as rounds land: wait for more seeds rather than
    # stop when every listed seed is done.
    parser.add_argument("--wait-for-seeds", action="store_true")
    # Outbreaks per chunk: training starts once this many have landed.
    parser.add_argument("--chunk", type=int, default=CHUNK)
    # Real-data training blocks (for example the German data's first 40%,
    # cut with `finetune_chronos` functions), spread over the chunks.
    parser.add_argument("--real-training", type=Path)
    parser.add_argument("--held-out-seeds", type=int, nargs=2, required=True)
    parser.add_argument("--outbreaks", type=int, default=ft.PREFIXES[-1])
    parser.add_argument("--holdout", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    # Sources to train on: weekly, daily, and 3-day blocks summed from the
    # daily triangle; one of them trains an adapter for it alone.
    # Every layout of the grid's design (`ft.DESIGNS`) by default.
    parser.add_argument("--sources", nargs="+")
    # Draws of each simulation this pass trains on, all 16 snapshots
    # each, picked by --subset-seed; all draws if not given.
    parser.add_argument("--draws-per-simulation", type=int)
    parser.add_argument("--subset-seed", type=int, default=0)
    # Share of blocks that are one series (the country or one leaf).
    parser.add_argument(
        "--univariate-share", type=float, default=ft.UNIVARIATE_SHARE
    )
    # Real validation blocks (scripts/finetune_real_validation.py), scored
    # at each checkpoint to choose among them; not trained on.
    parser.add_argument("--real-validation", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=os.environ.get("DEVICE", "auto"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--note", default="")
    parser.add_argument("--grid", required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--blocks-per-simulation", type=int, default=12)
    # Sim QC exclusions (scripts/sim_qc.py), read before each new batch.
    parser.add_argument("--exclude", type=Path)
    parser.add_argument(
        "--prefixes", type=int, nargs="+", default=list(ft.PREFIXES)
    )
    parser.add_argument("--min-gain", type=float, default=0.01)
    parser.add_argument("--stop-from", type=int, default=8000)
    parser.add_argument("--memory-gb", type=float, default=ft.MEMORY_GB)
    # Processes cutting blocks a chunk ahead; they do not change the blocks.
    parser.add_argument("--cut-workers", type=int, default=3)
    # The series bias (bdbvhubrenewal.forecasters.series_bias): learned
    # group attention bias for same-place and parent-child pairs.
    parser.add_argument("--series-bias", action="store_true")
    parser.add_argument("--bias-lr", type=float, default=BIAS_LR)
    # One segment of a continuous run: stop once prefix N is saved, with
    # the trainer's state beside it (`n<N>/state`), which the next
    # segment resumes from (workflow/rules/finetune.smk).
    parser.add_argument("--until", type=int)
    # Steps between held-out evaluations: the monitor, and by source on
    # a multiple of its steps. They only score, so the steps trained and
    # the settings hash are the same whatever they are.
    # Evaluate in the trainer (local) or leave adapter snapshots for
    # scripts/finetune_evaluate.py to score elsewhere (offload). Training
    # and the settings hash are the same either way.
    parser.add_argument("--eval", choices=EVAL_MODES, default="local")
    parser.add_argument("--eval-every", type=int, default=ft.EVAL_EVERY)
    parser.add_argument("--by-source-every", type=int, default=BY_SOURCE_EVERY)
    return parser


def check_offload(mode: str, stop_from: int, outbreaks: int) -> None:
    """Raise if evaluation is offloaded while early stopping can act.

    Stopping reads each checkpoint's held-out loss, which an offloaded
    run does not compute.
    """
    if mode == "offload" and stop_from <= outbreaks:
        raise ValueError(
            f"--eval offload needs --stop-from above --outbreaks "
            f"({stop_from} <= {outbreaks}): stopping reads held-out loss"
        )


def save_snapshot(model, where: Path, request: dict) -> None:
    """Save the adapter (and series bias) and what to evaluate on it.

    For `scripts/finetune_evaluate.py`. Written beside `where` and then
    moved into place, so an evaluator never reads part of one. Reading
    the weights changes nothing the training uses.
    """
    tmp = where.with_name(where.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp / "adapter")
    (tmp / "request.json").write_text(json.dumps(request))
    shutil.rmtree(where, ignore_errors=True)
    tmp.rename(where)


def save_blocks(blocks: list[dict], path: Path) -> None:
    """Save the held-out blocks, so an evaluator scores exactly these."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(blocks, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.rename(path)


def check_cadence(eval_every: int, by_source_every: int) -> None:
    """Raise unless both are positive and by source is on monitor steps."""
    if eval_every <= 0 or by_source_every <= 0:
        raise ValueError("evaluation steps must be positive")
    if by_source_every % eval_every:
        raise ValueError(
            f"--by-source-every {by_source_every} is not a multiple of "
            f"--eval-every {eval_every}"
        )


def main() -> None:
    """Train on the chunks as they are simulated."""
    global CHUNK, DRAWS_PER_PASS, SUBSET_SEED
    import torch
    from chronos import Chronos2Pipeline
    from chronos.chronos2.dataset import DatasetMode
    from chronos.chronos2.model import Chronos2Model

    parser = build_parser()
    args = parser.parse_args()
    CHUNK = args.chunk
    # The grid picks the block design; grids before r6 keep r5's blocks.
    ft.use_design(args.grid)
    ft.SOURCES = tuple(args.sources or ft.SOURCE_SHARES)
    ft.check_sources(ft.SOURCES)
    ft.check_package(Path(__file__).parents[1])
    ft.UNIVARIATE_SHARE = args.univariate_share
    global EXCLUDE_FILE, DRAWS_PER_PASS, SUBSET_SEED
    EXCLUDE_FILE = args.exclude
    DRAWS_PER_PASS, SUBSET_SEED = args.draws_per_simulation, args.subset_seed
    out, sim_dir = args.out, args.sim_dir
    # Read once at start: the worktree may move on while training.
    commit = code_commit()
    out.mkdir(parents=True, exist_ok=True)
    pause = out.parent / "PAUSE"
    status_file = out / "stream-status.json"
    status = {"stage": "starting", "step": 0, "outbreaks": 0}

    def report(**kwargs) -> None:
        status.update(kwargs, time=time.strftime("%Y-%m-%d %H:%M:%S"))
        status_file.write_text(json.dumps(status))

    extended = out / "train-last.txt"

    if (args.seed_list is None) == (args.train_seeds is None):
        parser.error("give one of --train-seeds and --seed-list")
    try:
        check_cadence(args.eval_every, args.by_source_every)
        check_offload(args.eval, args.stop_from, args.outbreaks)
    except ValueError as error:
        parser.error(str(error))
    # A release carries the held-out losses, which an offloaded run's
    # checkpoint gets later (the workflow's publish rule).
    if args.eval == "offload" and args.publish:
        parser.error("--eval offload publishes through the workflow")
    offload = args.eval == "offload"

    def training_seeds() -> range | list[int]:
        if args.seed_list is not None:
            return read_seed_list(args.seed_list)
        # The driver adds seeds by writing the last one to `extended`.
        last = args.train_seeds[1]
        if extended.exists():
            last = max(last, int(extended.read_text()))
        return range(args.train_seeds[0], last + 1)

    held_seeds = range(args.held_out_seeds[0], args.held_out_seeds[1] + 1)
    prefixes = [p for p in args.prefixes if p < args.outbreaks]
    prefixes.append(args.outbreaks)
    check_until(args.until, prefixes)
    k = args.blocks_per_simulation

    # Held-out outbreaks first.
    report(stage="waiting for held-out simulations")
    while len(held := ready_outbreaks(sim_dir, held_seeds)) < args.holdout:
        if all(seed_done(sim_dir, s) for s in held_seeds):
            raise RuntimeError(
                f"{len(held)} held-out outbreaks, fewer than {args.holdout}"
            )
        time.sleep(POLL)
    held = held[: args.holdout]
    (out / "outbreaks-held-out.txt").write_text("\n".join(held) + "\n")
    # Checkpointing and the largest block follow from the memory budget,
    # so held-out blocks are cut to the same size as training blocks.
    checkpoint_large_batches(ft.checkpoint_above(args.memory_gb))
    ft.MAX_SERIES = min(ft.MAX_SERIES, ft.max_series(args.memory_gb))
    pool = ProcessPoolExecutor(
        args.cut_workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=init_worker,
        initargs=(
            ft.MAX_SERIES,
            ft.MAX_LAGS,
            ft.SOURCES,
            ft.UNIVARIATE_SHARE,
            DRAWS_PER_PASS,
            SUBSET_SEED,
            args.grid,
        ),
    )
    held_pairs, empty = cut_in_workers(
        pool, sim_dir, held, args.seed, k, args.cut_workers
    )
    val = [b for _, b in held_pairs]
    if offload:
        save_blocks(val, out / HELD_OUT_BLOCKS)
    log(
        f"{len(held)} held-out outbreaks, {len(val)} blocks, "
        f"{len(empty)} skipped with no rows"
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  # noqa: NPY002
    device = torch_device(args.device)
    pipeline = Chronos2Pipeline.from_pretrained(
        CHRONOS_MODEL, revision=CHRONOS_REVISION, device_map=device
    )
    if args.series_bias:
        series_bias.attach(pipeline.model)
    lora = lora_config(args.series_bias)
    model = lora_model(pipeline.model, lora, args.init_adapter)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = make_optimiser(model, args.learning_rate, args.bias_lr)
    accepted = inspect.signature(Chronos2Model.forward).parameters
    base = {
        "grid": args.grid,
        "chronos_revision": CHRONOS_REVISION,
        "code_sha256": {
            **{p: ft._sha256(Path(p)) for p in ft.CODE},
            "scripts/finetune_stream.py": ft._sha256(Path(__file__)),
            # The bias's code shapes a bias checkpoint, so it is hashed.
            **(
                {c: ft._sha256(Path(c)) for c in BIAS_CODE}
                if args.series_bias
                else {}
            ),
            **{
                f"foundation.{f.__name__}": hashlib.sha256(
                    inspect.getsource(f).encode()
                ).hexdigest()
                for f in ft.FOUNDATION_CODE
            },
        },
        "max_lags": {k: [min(v), max(v)] for k, v in ft.MAX_LAGS.items()},
        "held_out_sha256": ft._sha256(out / "outbreaks-held-out.txt"),
        "n_validation": len(val),
        "chunk": CHUNK,
        "blocks_per_simulation": k,
        "univariate_share": ft.UNIVARIATE_SHARE,
        "parent_mix": list(ft.PARENT_MIX),
        "source_shares": ft.SOURCE_SHARES,
        **({"release_cadences": ft.CADENCES} if ft.DESIGN != "r5" else {}),
        "aggregate_end": ft.AGGREGATE_END,
        **(
            {
                "draws_per_simulation": DRAWS_PER_PASS,
                "subset_seed": SUBSET_SEED,
            }
            if DRAWS_PER_PASS
            else {}
        ),
        "block_sampling": (
            f"{k} of each simulation's draws x {ft.CUTS} cut-off slots x "
            "2 sources, seeded by the simulation"
        ),
        "lora": {
            "r": 8,
            "alpha": 16,
            **(
                {"modules_to_save": [series_bias.BIAS_MODULE]}
                if args.series_bias
                else {}
            ),
        },
        **bias_settings(args.series_bias, args.bias_lr),
        "learning_rate": args.learning_rate,
        **(
            {
                "init_adapter_sha256": ft._sha256(
                    args.init_adapter / "adapter_model.safetensors"
                )
            }
            if args.init_adapter
            else {}
        ),
        **(
            {"seed_list_sha256": ft._sha256(args.seed_list)}
            if args.seed_list
            else {}
        ),
        **(
            {"real_training_sha256": ft._sha256(args.real_training)}
            if args.real_training
            else {}
        ),
        "lr_decay": "linear to zero over the outbreaks",
        **(
            {"sources": list(ft.SOURCES)}
            if ft.SOURCES != ("weekly", "daily")
            else {}
        ),
        "batch_size": args.batch_size,
        "context": ft.CONTEXT,
        "prediction_length": (
            f"each block's horizon in whole patches of {ft.OUTPUT_PATCH} steps"
        ),
        "seed": args.seed,
        "note": args.note,
    }
    consumed = out / "consumed.txt"
    state_dir = out / "state"
    state = load_state(state_dir, model, optimiser)
    # The rates are this run's, as hashed, not the saved state's.
    for group, base_lr in zip(
        optimiser.param_groups,
        (args.learning_rate, args.bias_lr),
        strict=False,
    ):
        group["base_lr"] = base_lr
    step, done, position = 0, 0, 0
    # The outbreaks trained on, in order, and those of the current chunk.
    trained: list[str] = []
    chunk: list[str] = []
    seen = {"blocks": 0, "series": 0}
    losses: list[float] = []
    skipped: list[dict] = []

    def request(kinds: list[str], **extra) -> dict:
        """Return what an offloaded evaluation of the model now scores."""
        return {
            "step": step,
            "kinds": kinds,
            "batch_size": args.batch_size,
            "n_validation": len(val),
            **extra,
        }

    if state is None and offload:
        consumed.write_text("")
        step = 0
        start_loss = None
        history = [{"step": 0}]
        save_snapshot(model, out / "evals" / "step-0000000", request(["start"]))
        log("held-out loss at the start left to the evaluator")
    elif state is None:
        consumed.write_text("")
        start_loss = held_out_loss(model, val, args.batch_size)
        by_level = held_out_by_level(model, val, LEVEL_BLOCKS)
        by_source = held_out_by_source(
            model, val, args.batch_size, ft.EVAL_BLOCKS
        )
        history = [
            {
                "step": 0,
                "held_out_loss": start_loss,
                "by_level": by_level,
                "by_source": by_source,
            }
        ]
        log(
            f"held-out loss at the start {start_loss:.4f}; first "
            f"{ft.EVAL_BLOCKS} blocks by source {_by_source(by_source)}"
        )
    else:
        step, done, position = state["step"], state["done"], state["position"]
        start_loss, history = state["start_loss"], state["history"]
        seen, losses = state["seen"], state["losses"]
        skipped = state.get("skipped", [])
        trained = (
            state.get("trained")
            or ready_outbreaks(sim_dir, training_seeds())[:done]
        )
        chunk = state.get("chunk", []) if position else []
        lines = consumed.read_text().splitlines()[:step]
        consumed.write_text("".join(f"{line}\n" for line in lines))
        log(f"resumed at step {step}, outbreaks {done + 1}-, batch {position}")

    def snapshot() -> None:
        save_state(
            state_dir,
            model,
            optimiser,
            {
                "step": step,
                "done": done,
                "position": position,
                "trained": trained,
                "chunk": chunk,
                "start_loss": start_loss,
                "history": history,
                "seen": seen,
                "losses": losses,
                "skipped": skipped,
            },
        )

    def backward(part: list[int], share: float) -> float:
        """Accumulate the gradients of part of a batch."""
        batch = data._build_batch(part)
        value = train_step(model, batch, accepted, device, share)
        del batch
        return value

    def skip(i: int) -> None:
        """Record a block too large for memory."""
        series = int(data.inputs[i]["context"].shape[0])
        skipped.append(
            {"outbreak": pairs[i][0], "step": step + 1, "series": series}
        )
        log(f"skipped {pairs[i][0]}, {series} series, too large: {memory()}")

    def zero() -> None:
        optimiser.zero_grad()
        free_memory()

    # The next chunk's blocks are cut in `pool` while this one trains.
    ahead: dict[tuple[str, ...], list[Future]] = {}

    def cut_ahead(after: list[str]) -> None:
        """Start cutting the chunk after outbreaks `after` if it is ready."""
        size = min(CHUNK, args.outbreaks - len(after))
        if size <= 0:
            return
        ids = next_chunk(sim_dir, training_seeds(), after, size)
        if ids is not None and tuple(ids) not in ahead:
            ahead[tuple(ids)] = submit_cut(
                pool, sim_dir, ids, args.seed, k, args.cut_workers
            )

    real_training: list[dict] = []
    if args.real_training:
        real_training = load_blocks(args.real_training)
        log(f"{len(real_training)} real-data training blocks")
    real_validation = (
        load_blocks(args.real_validation) if args.real_validation else []
    )
    t0 = time.perf_counter()
    # A segment stops at its prefix; one whose state is already there
    # only keeps it, and one past it fails (`until_start`).
    until_start(
        done, args.until, (out / f"n{args.until}" / "settings.json").exists()
    )
    stop = args.outbreaks if args.until is None else args.until
    while done < stop:
        hi = min(done + CHUNK, args.outbreaks)
        split = {"wait_s": 0.0, "cut_s": 0.0, "train_s": 0.0, "eval_s": 0.0}
        tick = time.perf_counter()
        report(
            stage=f"waiting for outbreaks {done + 1}-{hi}",
            waiting_since=time.time(),
        )
        free_memory()
        while not chunk:
            ids = next_chunk(sim_dir, training_seeds(), trained, hi - done)
            if ids is not None:
                chunk = ids
                break
            done_all = all(seed_done(sim_dir, s) for s in training_seeds())
            if done_all and not args.wait_for_seeds:
                raise RuntimeError(f"fewer than {hi} training outbreaks")
            time.sleep(POLL)
        split["wait_s"] = time.perf_counter() - tick
        tick = time.perf_counter()
        report(
            stage=f"cutting blocks of outbreaks {done + 1}-{hi}",
            waiting_since=None,
        )
        if tuple(chunk) not in ahead:
            ahead.clear()
            cut_ahead(trained)
        if tuple(chunk) not in ahead:
            ahead[tuple(chunk)] = submit_cut(
                pool, sim_dir, chunk, args.seed, k, args.cut_workers
            )
        pairs, empty = _join(ahead.pop(tuple(chunk)))
        pairs += [
            (f"real-{i}", b)
            for i, b in real_share(
                list(enumerate(real_training)),
                done // CHUNK,
                -(-args.outbreaks // CHUNK),
            )
        ]
        split["cut_s"] = time.perf_counter() - tick
        tick = time.perf_counter()
        blocks = [b for _, b in pairs]
        sizes = np.array([int(b["context"].shape[0]) for b in blocks])
        rng = np.random.default_rng([args.seed, done])
        lengths = np.array([b["prediction_length"] for b in blocks])
        plan = ft.plan_by_length(
            lengths, sizes, args.batch_size, rng, ft.MAX_SERIES
        )
        data = ByLength(
            blocks,
            args.batch_size,
            DatasetMode.TRAIN,
            args.series_bias,
        )
        settle_heap()
        log(
            f"outbreaks {done + 1}-{hi}: {len(blocks)} blocks, "
            f"{sizes.sum()} series, {len(plan)} steps"
        )
        while position < len(plan):
            _, indices = plan[position]
            while pause.exists():
                report(stage="paused")
                time.sleep(30)
            set_learning_rates(
                optimiser,
                sims_done(done, len(chunk), position, len(plan)),
                args.outbreaks,
            )
            # Cheap without `free_memory`; the step starts from no gradients
            # whatever ran before it.
            optimiser.zero_grad()
            value = accumulate(list(indices), backward, zero, skip)
            if not finite_step(params, optimiser, value):
                log(f"step {step + 1}: no update (loss {value})")
            # Batches differ in shape, so cached buffers pile up unless they
            # are freed every step.
            free_memory()
            step += 1
            position += 1
            with consumed.open("a") as f:
                f.write(
                    f"{step}\t"
                    + ",".join(
                        f"{pairs[i][0]}:{'+'.join(sorted(set(blocks[i]['levels'])))}"
                        for i in indices
                    )
                    + "\n"
                )
            if step % 10 == 0:
                cut_ahead(trained + chunk)
                history.append({"step": step, "loss": value})
                report(
                    stage="training", step=step, outbreaks=done, gpu=memory()
                )
            if step % args.eval_every == 0 and offload:
                since = time.perf_counter()
                kinds = ["monitor"]
                if step % args.by_source_every == 0:
                    kinds.append("by_source")
                save_snapshot(
                    model, out / "evals" / f"step-{step:07d}", request(kinds)
                )
                entry = {"step": step, "lr": group_rates(optimiser)}
                bias = bias_summary(model)
                if bias:
                    entry["bias"] = bias
                split["eval_s"] += time.perf_counter() - since
                history.append(entry)
                log(
                    f"step {step}: loss {value:.4f}, held-out "
                    f"({', '.join(kinds)}) left to the evaluator, "
                    f"{time.perf_counter() - t0:.0f} s"
                )
                (out / "history.json").write_text(json.dumps(history))
            elif step % args.eval_every == 0:
                since = time.perf_counter()
                monitor = held_out_loss(
                    model, val[: ft.EVAL_BLOCKS], args.batch_size
                )
                entry = {"step": step, "eval_loss": monitor}
                described = ""
                if step % args.by_source_every == 0:
                    entry["by_source"] = held_out_by_source(
                        model, val, args.batch_size, ft.EVAL_BLOCKS
                    )
                    described = f"by source {_by_source(entry['by_source'])}, "
                entry["lr"] = group_rates(optimiser)
                bias = bias_summary(model)
                if bias:
                    entry["bias"] = bias
                    described += f"bias {_bias_range(bias)}, "
                # The next batch starts without the evaluation's buffers.
                free_memory()
                split["eval_s"] += time.perf_counter() - since
                history.append(entry)
                log(
                    f"step {step}: loss {value:.4f}, held-out "
                    f"(first {ft.EVAL_BLOCKS} blocks) {monitor:.4f}, "
                    f"{described}{time.perf_counter() - t0:.0f} s; {memory()}"
                )
                (out / "history.json").write_text(json.dumps(history))
            if step % SAVE_EVERY == 0:
                snapshot()
        trained.extend(chunk)
        chunk = []
        done, position = len(trained), 0
        split["train_s"] = time.perf_counter() - tick - split["eval_s"]
        tick = time.perf_counter()
        seen["blocks"] += len(pairs)
        seen["series"] += int(sizes.sum())
        seen["tokens"] = seen.get("tokens", 0) + int(
            (sizes * (ft.CONTEXT + lengths)).sum() // ft.OUTPUT_PATCH
        )
        mix = seen.setdefault("mix", {})
        for block in blocks:
            mix[block["mix"]] = mix.get(block["mix"], 0) + 1
            parents = seen.setdefault("parents", {})
            kind = block.get("parent", "exact")
            parents[kind] = parents.get(kind, 0) + 1
            cadences = seen.setdefault("cadences", {})
            cell = f"{block['source']}:{block['cadence']}"
            cadences[cell] = cadences.get(cell, 0) + 1
        # Blocks of strata with no cases by their cut are not laid out.
        mix["empty"] = mix.get("empty", 0) + len(empty)
        if done in prefixes:
            where = out / f"n{done}"
            where.mkdir(parents=True, exist_ok=True)
            # The release lists its outbreaks in the order trained, and
            # any block skipped.
            lines = [*trained, *skipped_lines(skipped)]
            (where / "outbreaks.txt").write_text("\n".join(lines) + "\n")
            (where / "consumed.txt").write_text(consumed.read_text())
            model.save_pretrained(where / "finetuned-ckpt")
            settings = {
                **base,
                "n_outbreaks": done,
                "outbreaks_sha256": ft._sha256(where / "outbreaks.txt"),
                "consumed_sha256": ft._sha256(where / "consumed.txt"),
                "steps": step,
                "sims_done": done,
                "blocks": seen["blocks"],
                "series": seen["series"],
                "level_mix": seen.get("mix", {}),
                **(
                    {"cadence_mix": seen.get("cadences", {})}
                    if ft.DESIGN != "r5"
                    else {}
                ),
                "tokens": seen.get("tokens", 0),
                "skipped_blocks": skipped,
                "memory_gb": args.memory_gb,
            }
            key = ft._settings_hash(settings)
            settings["hash"] = key
            # The commit the trainer ran from, outside the hash: the
            # checkpoint is rebuilt from it to the same hash.
            settings["code_commit"] = commit
            settings["device"] = device
            if offload:
                # The evaluator scores the checkpoint's adapter.
                if args.real_validation:
                    settings["real_validation_sha256"] = ft._sha256(
                        args.real_validation
                    )
                settings["seconds"] = time.perf_counter() - t0
                (where / "settings.json").write_text(
                    json.dumps(settings, indent=1)
                )
                history.append({"step": step, "n_outbreaks": done, "hash": key})
                (out / "history.json").write_text(json.dumps(history))
                log(
                    f"checkpoint n{done}: hash {key}, held-out loss left to "
                    f"the evaluator; {memory()}"
                )
                report(last_checkpoint=f"n{done} {key}")
                mismatch = finetune_verify.problems(where)
                if mismatch:
                    log(f"n{done} does not verify: {'; '.join(mismatch)}")
            else:
                settings["held_out_loss_start"] = start_loss
                settings["held_out_loss"] = held_out_loss(
                    model, val, args.batch_size
                )
                if args.real_validation:
                    real = real_validation
                    settings["real_validation_sha256"] = ft._sha256(
                        args.real_validation
                    )
                    settings["real_validation_loss"] = held_out_loss(
                        model, real, args.batch_size
                    )
                settings["held_out_loss_by_level"] = held_out_by_level(
                    model, val, LEVEL_BLOCKS
                )
                settings["held_out_loss_by_source"] = held_out_by_source(
                    model, val, args.batch_size
                )
                settings["seconds"] = time.perf_counter() - t0
                (where / "settings.json").write_text(
                    json.dumps(settings, indent=1)
                )
                history.append(
                    {
                        "step": step,
                        "n_outbreaks": done,
                        "held_out_loss": settings["held_out_loss"],
                        "hash": key,
                    }
                )
                (out / "history.json").write_text(json.dumps(history))
                log(
                    f"checkpoint n{done}: hash {key}, held-out loss "
                    f"{settings['held_out_loss']:.4f} "
                    f"(start {start_loss:.4f}); {memory()}"
                )
                report(last_checkpoint=f"n{done} {key}")
                # A checkpoint whose hash does not recompute from its commit
                # is kept but not published.
                mismatch = finetune_verify.problems(where)
                if mismatch:
                    log(f"n{done} not published: {'; '.join(mismatch)}")
                elif args.publish:
                    # Bias checkpoints are released apart (`release_tag`).
                    spec = f"{SLOTS_PREFIX}{key}" if args.series_bias else key
                    publish(where, spec)
                    report(last_release=release_tag(spec))
                # From `stop_from` on, go on only while the last doubling cut
                # held-out loss by more than `min_gain`.
                losses.append(settings["held_out_loss"])
                if should_stop(losses, done, args.stop_from, args.min_gain):
                    log(f"held-out loss gained under {args.min_gain:.0%}: stop")
                    break
        split["eval_s"] += time.perf_counter() - tick
        log(
            "chunk time: "
            + ", ".join(f"{k[:-2]} {v:.0f} s" for k, v in split.items())
        )
        snapshot()
        report(stage="training", step=step, outbreaks=done, split=split)
    pool.shutdown(cancel_futures=True)
    if args.until is not None and done == args.until:
        # A segment: the next one resumes from the state kept with n<N>.
        snapshot()
        keep_state(state_dir, out / f"n{done}")
        report(stage=f"stopped at n{done}", step=step, outbreaks=done)
        log(f"stopped at n{done} (--until)")
        return
    report(stage="finished", step=step, outbreaks=done)
    log("finished")


if __name__ == "__main__":
    main()
