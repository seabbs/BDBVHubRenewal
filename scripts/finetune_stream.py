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
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from bdbvhubrenewal.forecasters.foundation import (
    CHRONOS_MODEL,
    CHRONOS_REVISION,
)

sys.path.insert(0, str(Path(__file__).parent))
import finetune_chronos as ft

CHUNK = 500
"""Outbreaks per chunk."""

LEVEL_BLOCKS = 300
"""Held-out blocks scored by level at each checkpoint."""

SAVE_EVERY = 50
"""Steps between saves of the trainer's state, which a restarted trainer
resumes from."""

POLL = 60
"""Seconds between checks for a chunk's simulations."""

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


def ready_outbreaks(sim_dir: Path, seeds: range) -> list[str]:
    """Return the outbreaks of the finished batches, in order.

    The batches are read in seed order up to the first not yet finished.
    A finished batch is read once and remembered.
    """
    out: list[str] = []
    for seed in seeds:
        if (sim_dir, seed) not in _READY:
            if not seed_done(sim_dir, seed):
                break
            ids = read_ids(sim_dir / f"settings-{seed}.csv")
            _READY[sim_dir, seed] = sorted(ids, key=ft._order)
        out.extend(_READY[sim_dir, seed])
    return out


def cut(
    sim_dir: Path, ids: list[str], seed: int, per_simulation: int
) -> list[tuple[str, dict]]:
    """Return a sample of blocks of each simulation, with its draw.

    A simulation's candidate blocks are each of its observation draws
    (`<id>-<draw>`) cut in each of `CUTS` slots of each source; a sample of
    `per_simulation` of them, seeded by the simulation, is cut.
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
            for source in ("weekly", "daily")
            for j in range(ft.CUTS)
        ]
        rng = np.random.default_rng([seed, zlib.crc32(simulation.encode())])
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
                draws[outbreak], ft.strata(row), draw_rng, weeks, picks=picks
            )
            out.extend((outbreak, block) for block in blocks)
    return out


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
                inputs=[block],
                context_length=ft.CONTEXT,
                prediction_length=ft.HORIZON,
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
    from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode
    from chronos.chronos2.model import Chronos2Model
    from peft import LoraConfig, get_peft_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sim_dir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--train-seeds", type=int, nargs=2, required=True)
    parser.add_argument("--held-out-seeds", type=int, nargs=2, required=True)
    parser.add_argument("--outbreaks", type=int, default=ft.PREFIXES[-1])
    parser.add_argument("--holdout", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--note", default="")
    parser.add_argument("--grid", required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--blocks-per-simulation", type=int, default=12)
    parser.add_argument(
        "--prefixes", type=int, nargs="+", default=list(ft.PREFIXES)
    )
    parser.add_argument("--min-gain", type=float, default=0.01)
    parser.add_argument("--stop-from", type=int, default=8000)
    parser.add_argument("--memory-gb", type=float, default=ft.MEMORY_GB)
    args = parser.parse_args()
    out, sim_dir = args.out, args.sim_dir
    out.mkdir(parents=True, exist_ok=True)
    pause = out.parent / "PAUSE"
    status_file = out / "stream-status.json"
    status = {"stage": "starting", "step": 0, "outbreaks": 0}

    def report(**kwargs) -> None:
        status.update(kwargs, time=time.strftime("%Y-%m-%d %H:%M:%S"))
        status_file.write_text(json.dumps(status))

    extended = out / "train-last.txt"

    def training_seeds() -> range:
        # The driver adds seeds by writing the last one to `extended`.
        last = args.train_seeds[1]
        if extended.exists():
            last = max(last, int(extended.read_text()))
        return range(args.train_seeds[0], last + 1)

    held_seeds = range(args.held_out_seeds[0], args.held_out_seeds[1] + 1)
    prefixes = [p for p in args.prefixes if p < args.outbreaks]
    prefixes.append(args.outbreaks)
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
    ft.checkpoint_large_batches(ft.checkpoint_above(args.memory_gb))
    ft.MAX_SERIES = min(ft.MAX_SERIES, ft.max_series(args.memory_gb))
    val = [b for _, b in cut(sim_dir, held, args.seed, k)]
    log(f"{len(held)} held-out outbreaks, {len(val)} blocks")
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
    model = get_peft_model(pipeline.model, lora)
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
            f"{k} of each simulation's draws x {ft.CUTS} cut-off slots x "
            "2 sources, seeded by the simulation"
        ),
        "lora": {"r": 8, "alpha": 16},
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "context": ft.CONTEXT,
        "horizon": ft.HORIZON,
        "seed": args.seed,
        "note": args.note,
    }
    consumed = out / "consumed.txt"
    state_dir = out / "state"
    state = load_state(state_dir, model, optimiser)
    step, done, position = 0, 0, 0
    seen = {"blocks": 0, "series": 0}
    losses: list[float] = []
    skipped: list[dict] = []
    if state is None:
        consumed.write_text("")
        start_loss = ft.held_out_loss(model, val, args.batch_size)
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

    t0 = time.perf_counter()
    while done < args.outbreaks:
        hi = min(done + CHUNK, args.outbreaks)
        report(stage=f"waiting for outbreaks {done + 1}-{hi}")
        free_memory()
        while len(ready := ready_outbreaks(sim_dir, training_seeds())) < hi:
            if all(seed_done(sim_dir, s) for s in training_seeds()):
                raise RuntimeError(
                    f"{len(ready)} training outbreaks, fewer than {hi}"
                )
            time.sleep(POLL)
        ids = ready[done:hi]
        pairs = cut(sim_dir, ids, args.seed, k)
        blocks = [b for _, b in pairs]
        sizes = np.array([int(b["context"].shape[0]) for b in blocks])
        rng = np.random.default_rng([args.seed, done])
        plan = ft.plan_batches(sizes, args.batch_size, rng, ft.MAX_SERIES)
        data = Chronos2Dataset(
            inputs=blocks,
            context_length=ft.CONTEXT,
            prediction_length=ft.HORIZON,
            batch_size=args.batch_size,
            output_patch_size=ft.OUTPUT_PATCH,
            min_past=ft.CONTEXT,
            mode=DatasetMode.TRAIN,
        )
        log(
            f"outbreaks {done + 1}-{hi}: {len(blocks)} blocks, "
            f"{sizes.sum()} series, {len(plan)} steps"
        )
        while position < len(plan):
            indices = plan[position]
            while pause.exists():
                report(stage="paused")
                time.sleep(30)
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
                history.append({"step": step, "loss": value})
                report(
                    stage="training", step=step, outbreaks=done, gpu=memory()
                )
            if step % ft.EVAL_EVERY == 0:
                monitor = ft.held_out_loss(
                    model, val[: ft.EVAL_BLOCKS], args.batch_size
                )
                history.append({"step": step, "eval_loss": monitor})
                log(
                    f"step {step}: loss {value:.4f}, held-out "
                    f"(first {ft.EVAL_BLOCKS} blocks) {monitor:.4f}, "
                    f"{time.perf_counter() - t0:.0f} s; {memory()}"
                )
                (out / "history.json").write_text(json.dumps(history))
            if step % SAVE_EVERY == 0:
                snapshot()
        done, position = hi, 0
        seen["blocks"] += len(pairs)
        seen["series"] += int(sizes.sum())
        for block in blocks:
            mix = seen.setdefault("mix", {})
            mix[block["mix"]] = mix.get(block["mix"], 0) + 1
        if done in prefixes:
            where = out / f"n{done}"
            where.mkdir(parents=True, exist_ok=True)
            order = ready[:done]
            # The release lists its outbreaks and any block skipped.
            lines = [*order, *skipped_lines(skipped)]
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
                "tokens": seen["series"]
                * (ft.CONTEXT + ft.HORIZON)
                // ft.OUTPUT_PATCH,
                "skipped_blocks": skipped,
                "memory_gb": args.memory_gb,
            }
            key = ft._settings_hash(settings)
            settings["hash"] = key
            settings["device"] = device
            settings["held_out_loss_start"] = start_loss
            settings["held_out_loss"] = ft.held_out_loss(
                model, val, args.batch_size
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
            if args.publish:
                publish(where, key)
                report(last_release=f"chronos-ft-{key}")
            # From `stop_from` on, go on only while the last doubling cut
            # held-out loss by more than `min_gain`.
            losses.append(settings["held_out_loss"])
            if should_stop(losses, done, args.stop_from, args.min_gain):
                log(f"held-out loss gained under {args.min_gain:.0%}: stop")
                break
        snapshot()
        report(stage="training", step=step, outbreaks=done)
    report(stage="finished", step=step, outbreaks=done)
    log("finished")


if __name__ == "__main__":
    main()
