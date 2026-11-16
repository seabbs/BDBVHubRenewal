# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Fine-tune Chronos-2 in one pass while the outbreaks are simulated.

The training outbreaks are taken in order of seed and index, in chunks of
`CHUNK` outbreaks. A chunk is trained on as soon as every simulation
batch it needs has finished (`scripts/finetune_simulate.R`), and the
trainer waits for it otherwise. Within a chunk the blocks
(`finetune_chronos.outbreak_blocks`, seeded per outbreak) are shuffled
and packed by size (`finetune_chronos.plan_batches`). The learning rate
is constant, so the checkpoint after the first N outbreaks
(`finetune_chronos.PREFIXES`) is Chronos-2 fine-tuned on those N in one
pass. Each is saved in `OUT/n<N>` with its consumed order and settings
hash, and published as the release `chronos-ft-<hash>` when `--publish`
is set. Held-out loss is on the first `--holdout` outbreaks of the
held-out seeds, simulated first.

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
import pickle
import subprocess
import sys
import time
import zlib
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from bdbvhubrenewal.forecasters.foundation import (
    CHRONOS_MODEL,
    CHRONOS_REVISION,
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

POLL = 60
"""Seconds between checks for a chunk's simulations."""

PATIENCE = 20 * 60
"""Seconds a chunk waits for a missing batch before later batches train
first; the late batch's outbreaks then join a later chunk."""

REPO = "seabbs/BDBVHubRenewal"
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


def read_seed_list(path: Path) -> list[int]:
    """Return the training seeds a file lists, one a line, in its order."""
    return [int(x) for x in path.read_text().split()]


def lora_model(base, config, init: Path | None):
    """Return `base` with a LoRA adapter to train.

    A fresh adapter from `config`, or, to continue a round, the adapter
    saved at `init`, still trainable.
    """
    from peft import PeftModel, get_peft_model

    if init is None:
        return get_peft_model(base, config)
    return PeftModel.from_pretrained(base, init, is_trainable=True)


def learning_rate(step: int, base: float, steps: int) -> float:
    """Return the learning rate of step `step` (from 0).

    It falls linearly from `base` to zero over `steps` steps, as
    Chronos-2's own fine-tuning does, and stays constant if `steps` is 0.
    """
    if steps <= 0:
        return base
    return base * max(0.0, 1 - step / steps)


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


def batch_outbreaks(sim_dir: Path, seed: int) -> list[str] | None:
    """Return a finished batch's outbreaks in order, or None if not done.

    A finished batch is read once and remembered.
    """
    if (sim_dir, seed) not in _READY:
        if not seed_done(sim_dir, seed):
            return None
        ids = read_ids(sim_dir / f"settings-{seed}.csv")
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
    waited: float,
    patience: float = PATIENCE,
) -> list[str] | None:
    """Return the next `size` outbreaks to train on, or None if not ready.

    They are the first outbreaks not yet `trained` of the finished
    batches in seed order, up to the first batch not finished. Once the
    chunk has `waited` `patience` seconds, later finished batches count
    too, so a straggler's outbreaks join a later chunk instead of holding
    up training.
    """
    seen = set(trained)
    out: list[str] = []
    for seed in seeds:
        ids = batch_outbreaks(sim_dir, seed)
        if ids is None:
            if waited < patience:
                break
            continue
        out.extend(o for o in ids if o not in seen)
        if len(out) >= size:
            return out[:size]
    return None


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
        candidates = [
            (outbreak, source, j)
            for outbreak in sorted(draws)
            for source in ft.SOURCES
            for j in range(ft.CUTS)
        ]
        rng = np.random.default_rng([seed, zlib.crc32(simulation.encode())])
        if ft.ALL_DRAWS:
            # `per_simulation` snapshots, each in its own stretch of the
            # outbreak with a source of its own, cut from every draw with
            # the same cut-off, lag and places.
            sources = rng.choice(ft.SOURCES, per_simulation)
            picks = {(str(s), j) for j, s in enumerate(sources)}
            for outbreak in sorted(draws):
                blocks = ft.outbreak_blocks(
                    draws[outbreak],
                    ft.strata(row),
                    np.random.default_rng(
                        [seed, zlib.crc32(simulation.encode()), 1]
                    ),
                    weeks,
                    cuts=per_simulation,
                    picks=picks,
                    empty=empty,
                )
                out.extend((outbreak, block) for block in blocks)
            continue
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
    all_draws: bool = False,
    univariate_share: float = 0.0,
) -> None:
    """Give a cutting worker the trainer's block settings."""
    ft.MAX_SERIES, ft.MAX_LAGS, ft.SOURCES = max_series, lags, sources
    ft.ALL_DRAWS, ft.UNIVARIATE_SHARE = all_draws, univariate_share


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

    def __init__(self, blocks: list[dict], batch_size: int, mode) -> None:
        from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode

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
        return self.data[length]._build_batch([self.local[i] for i in indices])

    def batches(self):
        """Yield every block once, in batches of one length."""
        for data in self.data.values():
            yield from data


def held_out_loss(model, blocks: list[dict], batch_size: int) -> float:
    """Return the mean quantile loss of a Chronos-2 model on blocks."""
    import torch
    from chronos.chronos2.dataset import DatasetMode
    from chronos.chronos2.model import Chronos2Model

    data = ByLength(blocks, batch_size, DatasetMode.VALIDATION)
    accepted = inspect.signature(Chronos2Model.forward).parameters
    training = model.training
    model.eval()
    losses, weights = [], []
    with torch.no_grad():
        for batch in data.batches():
            batch = {
                k: v.to(model.device) if torch.is_tensor(v) else v
                for k, v in batch.items()
                if k in accepted
            }
            losses.append(float(model(**batch).loss))
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
    tag = f"chronos-ft-{key}"
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

    If memory runs out, the gradients are cleared (`zero`) and the batch
    is run again in halves, and so on; a single block that still does not
    fit is left out (`skip`). Every other block is used once.
    """
    parts = [indices]
    while True:
        zero()
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
    finite = (
        np.isfinite(value)
        and bool(grads)
        and all(torch.isfinite(g).all() for g in grads)
    )
    if finite:
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimiser.step()
    optimiser.zero_grad()
    return finite


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
            levels = np.array(block["levels"])
            for level in sorted(set(levels)):
                target = batch["future_target"].clone()
                keep = torch.zeros(len(target), dtype=torch.bool)
                keep[: len(levels)] = torch.from_numpy(levels == level)
                target[~keep] = float("nan")
                inputs = {
                    k: v.to(model.device) if torch.is_tensor(v) else v
                    for k, v in {**batch, "future_target": target}.items()
                    if k in accepted
                }
                loss = float(model(**inputs).loss)
                losses.setdefault(level, []).append(loss)
    model.train(training)
    return {k: float(np.mean(v)) for k, v in losses.items()}


def skipped_lines(skipped: list[dict]) -> list[str]:
    """Return the lines listing skipped blocks in a release's outbreaks."""
    return [
        f"# skipped {b['outbreak']} at step {b['step']}, {b['series']} series"
        for b in skipped
    ]


def save_state(where: Path, model, optimiser, state: dict) -> None:
    """Save the trainer's state: adapter, optimiser and position."""
    import torch
    from peft import get_peft_model_state_dict

    tmp = where.with_suffix(".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    torch.save(get_peft_model_state_dict(model), tmp / "adapter.pt")
    torch.save(optimiser.state_dict(), tmp / "optimiser.pt")
    (tmp / "state.json").write_text(json.dumps(state))
    if where.exists():
        for f in where.iterdir():
            f.unlink()
        where.rmdir()
    tmp.rename(where)


def load_state(where: Path, model, optimiser) -> dict | None:
    """Load the trainer's saved state into the model and optimiser."""
    import torch
    from peft import set_peft_model_state_dict

    if not (where / "state.json").exists():
        return None
    set_peft_model_state_dict(
        model, torch.load(where / "adapter.pt", weights_only=True)
    )
    optimiser.load_state_dict(
        torch.load(where / "optimiser.pt", weights_only=True)
    )
    return json.loads((where / "state.json").read_text())


def main() -> None:
    """Train on the chunks as they are simulated."""
    import torch
    from chronos import Chronos2Pipeline
    from chronos.chronos2.dataset import DatasetMode
    from chronos.chronos2.model import Chronos2Model
    from peft import LoraConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sim_dir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--train-seeds", type=int, nargs=2)
    # A round: its training seeds, one a line, in place of --train-seeds,
    # and the adapter it continues from.
    parser.add_argument("--seed-list", type=Path)
    parser.add_argument("--init-adapter", type=Path)
    # Real-data training blocks (for example the German data's first 40%,
    # cut with `finetune_chronos` functions), spread over the chunks.
    parser.add_argument("--real-training", type=Path)
    parser.add_argument("--held-out-seeds", type=int, nargs=2, required=True)
    parser.add_argument("--outbreaks", type=int, default=ft.PREFIXES[-1])
    parser.add_argument("--holdout", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    # Steps over which the learning rate falls linearly to zero; 0 keeps
    # it constant.
    parser.add_argument("--lr-steps", type=int, default=0)
    # Sources to train on: weekly, daily, and 3-day blocks summed from the
    # daily triangle; one of them trains an adapter for it alone.
    parser.add_argument("--sources", nargs="+", default=list(ft.SOURCES))
    # Real validation blocks (scripts/finetune_real_validation.py), scored
    # at each checkpoint to choose among them; not trained on.
    parser.add_argument("--real-validation", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--note", default="")
    parser.add_argument("--grid", required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--blocks-per-simulation", type=int, default=12)
    # Cut each simulation's snapshots alike from every draw: with 16
    # snapshots and 10 draws, 160 blocks a simulation.
    parser.add_argument("--all-draws", action="store_true")
    # Share of blocks that are one series (`ft.UNIVARIATE_SHARE`).
    parser.add_argument("--univariate-share", type=float, default=0.0)
    parser.add_argument(
        "--prefixes", type=int, nargs="+", default=list(ft.PREFIXES)
    )
    parser.add_argument("--min-gain", type=float, default=0.01)
    parser.add_argument("--stop-from", type=int, default=8000)
    parser.add_argument("--memory-gb", type=float, default=ft.MEMORY_GB)
    # Processes cutting blocks a chunk ahead; they do not change the blocks.
    parser.add_argument("--cut-workers", type=int, default=3)
    args = parser.parse_args()
    ft.SOURCES = tuple(args.sources)
    out, sim_dir = args.out, args.sim_dir
    # The commit this trainer runs, read once: the worktree may move on.
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
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
    k = args.blocks_per_simulation
    ft.ALL_DRAWS, ft.UNIVARIATE_SHARE = args.all_draws, args.univariate_share

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
            ft.ALL_DRAWS,
            ft.UNIVARIATE_SHARE,
        ),
    )
    held_pairs, empty = cut_in_workers(
        pool, sim_dir, held, args.seed, k, args.cut_workers
    )
    val = [b for _, b in held_pairs]
    log(
        f"{len(held)} held-out outbreaks, {len(val)} blocks, "
        f"{len(empty)} skipped with no rows"
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  # noqa: NPY002
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    pipeline = Chronos2Pipeline.from_pretrained(
        CHRONOS_MODEL, revision=CHRONOS_REVISION, device_map=device
    )
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "self_attention.q",
            "self_attention.v",
            "self_attention.k",
            "self_attention.o",
            "output_patch_embedding.output_layer",
        ],
    )
    model = lora_model(pipeline.model, lora, args.init_adapter)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(params, lr=args.learning_rate)
    accepted = inspect.signature(Chronos2Model.forward).parameters
    base = {
        "grid": args.grid,
        "chronos_revision": CHRONOS_REVISION,
        "code_sha256": {
            **{p: ft._sha256(Path(p)) for p in ft.CODE},
            "scripts/finetune_stream.py": ft._sha256(Path(__file__)),
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
        "block_sampling": (
            f"{k} snapshots, each with one source, cut alike from every "
            f"draw; {ft.UNIVARIATE_SHARE:g} of blocks one series; seeded "
            "by the simulation"
            if ft.ALL_DRAWS
            else f"{k} of each simulation's draws x {ft.CUTS} cut-off slots"
            " x 2 sources, seeded by the simulation"
        ),
        "all_draws": ft.ALL_DRAWS,
        "univariate_share": ft.UNIVARIATE_SHARE,
        "lora": {"r": 8, "alpha": 16},
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
        **({"lr_steps": args.lr_steps} if args.lr_steps else {}),
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
    step, done, position = 0, 0, 0
    # The outbreaks trained on, in order, and those of the current chunk.
    trained: list[str] = []
    chunk: list[str] = []
    seen = {"blocks": 0, "series": 0}
    losses: list[float] = []
    skipped: list[dict] = []
    if state is None:
        consumed.write_text("")
        start_loss = held_out_loss(model, val, args.batch_size)
        by_level = held_out_by_level(model, val, LEVEL_BLOCKS)
        history = [
            {"step": 0, "held_out_loss": start_loss, "by_level": by_level}
        ]
        log(f"held-out loss at the start {start_loss:.4f}")
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
        batch = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in data._build_batch(part).items()
            if k in accepted
        }
        loss = model(**batch).loss
        (loss * share).backward()
        value = float(loss.detach())
        del batch, loss
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
        ids = next_chunk(sim_dir, training_seeds(), after, size, 0)
        if ids is not None and tuple(ids) not in ahead:
            ahead[tuple(ids)] = submit_cut(
                pool, sim_dir, ids, args.seed, k, args.cut_workers
            )

    real_training: list[dict] = []
    if args.real_training:
        with args.real_training.open("rb") as f:
            real_training = pickle.load(f)
        log(f"{len(real_training)} real-data training blocks")
    t0 = time.perf_counter()
    while done < args.outbreaks:
        hi = min(done + CHUNK, args.outbreaks)
        split = {"wait_s": 0.0, "cut_s": 0.0, "train_s": 0.0, "eval_s": 0.0}
        tick = time.perf_counter()
        report(
            stage=f"waiting for outbreaks {done + 1}-{hi}",
            waiting_since=time.time(),
        )
        free_memory()
        while not chunk:
            waited = time.perf_counter() - tick
            ids = next_chunk(
                sim_dir, training_seeds(), trained, hi - done, waited
            )
            if ids is not None:
                chunk = ids
                break
            if all(seed_done(sim_dir, s) for s in training_seeds()):
                raise RuntimeError(f"fewer than {hi} training outbreaks")
            time.sleep(POLL)
        contiguous = ready_outbreaks(sim_dir, training_seeds())[done:hi]
        if chunk != contiguous:
            log(
                f"outbreaks {done + 1}-{hi}: batches still running after "
                f"{PATIENCE // 60} min, so later batches train first"
            )
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
        data = ByLength(blocks, args.batch_size, DatasetMode.TRAIN)
        log(
            f"outbreaks {done + 1}-{hi}: {len(blocks)} blocks, "
            f"{sizes.sum()} series, {len(plan)} steps"
        )
        while position < len(plan):
            _, indices = plan[position]
            while pause.exists():
                report(stage="paused")
                time.sleep(30)
            for group in optimiser.param_groups:
                group["lr"] = learning_rate(
                    step, args.learning_rate, args.lr_steps
                )
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
            if step % ft.EVAL_EVERY == 0:
                since = time.perf_counter()
                monitor = held_out_loss(
                    model, val[: ft.EVAL_BLOCKS], args.batch_size
                )
                split["eval_s"] += time.perf_counter() - since
                history.append({"step": step, "eval_loss": monitor})
                log(
                    f"step {step}: loss {value:.4f}, held-out "
                    f"(first {ft.EVAL_BLOCKS} blocks) {monitor:.4f}, "
                    f"{time.perf_counter() - t0:.0f} s; {memory()}"
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
                "blocks": seen["blocks"],
                "series": seen["series"],
                "level_mix": seen.get("mix", {}),
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
            settings["held_out_loss_start"] = start_loss
            settings["held_out_loss"] = held_out_loss(
                model, val, args.batch_size
            )
            if args.real_validation:
                with args.real_validation.open("rb") as f:
                    real = pickle.load(f)
                settings["real_validation_sha256"] = ft._sha256(
                    args.real_validation
                )
                settings["real_validation_loss"] = held_out_loss(
                    model, real, args.batch_size
                )
            settings["held_out_loss_by_level"] = held_out_by_level(
                model, val, LEVEL_BLOCKS
            )
            settings["seconds"] = time.perf_counter() - t0
            (where / "settings.json").write_text(json.dumps(settings, indent=1))
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
                f"{settings['held_out_loss']:.4f} (start {start_loss:.4f}); "
                f"{memory()}"
            )
            report(last_checkpoint=f"n{done} {key}")
            # A checkpoint whose hash does not recompute from its commit
            # is kept but not published.
            mismatch = finetune_verify.problems(where)
            if mismatch:
                log(f"n{done} not published: {'; '.join(mismatch)}")
            elif args.publish:
                publish(where, key)
                report(last_release=f"chronos-ft-{key}")
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
    report(stage="finished", step=step, outbreaks=done)
    log("finished")


if __name__ == "__main__":
    main()
