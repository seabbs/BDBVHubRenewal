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
        --train-seeds 5001 5360 --held-out-seeds 5901 5907 --publish

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

sys.path.insert(0, str(Path(__file__).parent))
import finetune_chronos as ft

CHUNK = 500
"""Outbreaks per chunk."""

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


def seed_done(sim_dir: Path, seed: int) -> bool:
    """Return whether a simulation batch has finished, checking it.

    Raises
    ------
    RuntimeError
        If the batch failed, or its counts and settings disagree.
    """
    counts = sim_dir / f"counts-{seed}.csv.gz"
    settings = sim_dir / f"settings-{seed}.csv"
    if not (counts.exists() and settings.exists()):
        log_file = sim_dir / "logs" / f"{seed}.log"
        if log_file.exists() and "Execution halted" in log_file.read_text():
            raise RuntimeError(f"simulation batch {seed} failed: {log_file}")
        return False
    a = set(pd.read_csv(settings, usecols=["outbreak"])["outbreak"])
    b = set(pd.read_csv(counts, usecols=["outbreak"])["outbreak"])
    if a != b:
        raise RuntimeError(f"batch {seed}: counts and settings disagree")
    return True


def ready_outbreaks(sim_dir: Path, seeds: range) -> list[str]:
    """Return the outbreaks of the finished batches, in order.

    The batches are read in seed order up to the first not yet finished.
    """
    out: list[str] = []
    for seed in seeds:
        if not seed_done(sim_dir, seed):
            break
        ids = pd.read_csv(sim_dir / f"settings-{seed}.csv")["outbreak"]
        out.extend(sorted(ids, key=ft._order))
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


def publish(where: Path, key: str) -> None:
    """Publish a checkpoint as the release `chronos-ft-<key>`."""
    files = [
        where / "finetuned-ckpt" / "adapter_config.json",
        where / "finetuned-ckpt" / "adapter_model.safetensors",
        where / "settings.json",
        where / "outbreaks.txt",
    ]
    sums = where / "sha256sums.txt"
    sums.write_text("".join(f"{ft._sha256(f)}  {f.name}\n" for f in files))
    n = json.loads((where / "settings.json").read_text())["n_outbreaks"]
    command = [
        "gh",
        "release",
        "create",
        f"chronos-ft-{key}",
        "-R",
        REPO,
        "--latest=false",
        "--title",
        f"Chronos-2 fine-tuned checkpoint {key} ({n} outbreaks)",
        "--notes",
        f"LoRA adapter of Chronos-2 for the _ft models, fine-tuned in one "
        f"pass on {n} simulated outbreaks (scripts/finetune_stream.py). "
        f"Check the files with sha256sums.txt.\n\nThis was opened by a "
        f"bot. Please ping @seabbs for any questions.",
        *map(str, files),
        str(sums),
    ]
    for attempt in range(1, 4):
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            log(f"published chronos-ft-{key}: {result.stdout.strip()}")
            return
        log(f"publish attempt {attempt} failed: {result.stderr.strip()}")
        time.sleep(60)
    raise RuntimeError(f"could not publish chronos-ft-{key}")


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
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--blocks-per-simulation", type=int, default=12)
    parser.add_argument(
        "--prefixes", type=int, nargs="+", default=list(ft.PREFIXES)
    )
    parser.add_argument("--min-gain", type=float, default=0.01)
    parser.add_argument("--stop-from", type=int, default=8000)
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
    val = [b for _, b in cut(sim_dir, held, args.seed, k)]
    log(f"{len(held)} held-out outbreaks, {len(val)} blocks")

    ft.checkpoint_large_batches()
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
        ft.CHRONOS_MODEL, revision=ft.CHRONOS_REVISION, device_map=device
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
        "chronos_revision": ft.CHRONOS_REVISION,
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
    if state is None:
        consumed.write_text("")
        start_loss = ft.held_out_loss(model, val, args.batch_size)
        history = [{"step": 0, "held_out_loss": start_loss}]
        log(f"held-out loss at the start {start_loss:.4f}")
    else:
        step, done, position = state["step"], state["done"], state["position"]
        start_loss, history = state["start_loss"], state["history"]
        seen, losses = state["seen"], state["losses"]
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
            },
        )

    def train_step(batch_blocks: list[int], share: float) -> float:
        """Accumulate gradients of a batch; split it if memory runs out."""
        try:
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in data._build_batch(batch_blocks).items()
                if k in accepted
            }
            loss = model(**batch).loss
            (loss * share).backward()
            value = float(loss.detach())
            del batch, loss
            return value
        except RuntimeError as error:
            if "out of memory" not in str(error):
                raise
            free_memory()
            if len(batch_blocks) == 1:
                log(f"skipped a block too large for memory: {memory()}")
                return float("nan")
            half = len(batch_blocks) // 2
            log(f"split a batch of {len(batch_blocks)} blocks for memory")
            parts = (batch_blocks[:half], batch_blocks[half:])
            values = [train_step(part, share / 2) for part in parts]
            return float(np.nanmean(values))

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
        plan = ft.plan_batches(sizes, args.batch_size, rng)
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
            value = train_step(indices, 1.0)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimiser.step()
            optimiser.zero_grad()
            # Batches differ in shape, so cached buffers pile up unless they
            # are freed every step.
            free_memory()
            step += 1
            position += 1
            with consumed.open("a") as f:
                f.write(f"{step}\t{','.join(pairs[i][0] for i in indices)}\n")
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
        if done in prefixes:
            where = out / f"n{done}"
            where.mkdir(parents=True, exist_ok=True)
            order = ready[:done]
            (where / "outbreaks.txt").write_text("\n".join(order) + "\n")
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
                "tokens": seen["series"]
                * (ft.CONTEXT + ft.HORIZON)
                // ft.OUTPUT_PATCH,
            }
            key = ft._settings_hash(settings)
            settings["hash"] = key
            settings["device"] = device
            settings["held_out_loss_start"] = start_loss
            settings["held_out_loss"] = ft.held_out_loss(
                model, val, args.batch_size
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
            gain = losses[-2] - losses[-1] if len(losses) > 1 else np.inf
            if done >= args.stop_from and gain < args.min_gain * losses[-2]:
                log(f"held-out loss gained under {args.min_gain:.0%}: stop")
                break
        snapshot()
        report(stage="training", step=step, outbreaks=done)
    report(stage="finished", step=step, outbreaks=done)
    log("finished")


if __name__ == "__main__":
    main()
