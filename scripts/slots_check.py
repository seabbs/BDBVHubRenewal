# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Check that a per-series slots checkpoint ties lags to their place.

The covariate ablation (`scripts/covariate_ablation.py`) found that
Chronos-2 cannot tell which place a lag series belongs to: moving whole
lag series between places gave bit-identical output. Per-series slots
(`bdbvhubrenewal.forecasters.slots`) add one learned vector to a place's
count series and all its lag series. This script reruns the place swap
with slots, on the same blocks, for zero-shot Chronos-2, the no-slots
fine-tuned adapter and a slots adapter.

A swap tests the slots only if each row's slot stays with its position,
the place of the target it sits beside, while the lag content moves to
another place. Slots are therefore set from the rows' positions after the
condition is applied (`model_ids`), exactly as the trainer's held-out
losses (`finetune_stream.block_slots`) and `chronos_sampler` set them:
one distinct slot per place drawn with `slots.EVAL_SEED`. If a lag series
took its slot along, the swap would only reorder rows, which Chronos-2
cannot see.

Blocks:
  sim      held-out simulations cut as the trainer cuts them, sampled as
           the covariate ablation sampled them (`sim_sample`).
  germany  German blocks of the national total and the 16 states, from
           `finetune_realdata.py blocks-germany` (`germany_blocks`).

Conditions (`covariate_ablation.condition`): given, swap_places (every
place takes another's whole lag series), removed (no lag rows) and
shuffle_places (at each step the places' lags go to random places).

For each block, condition and model it records the loss (Chronos-2's
scaled quantile loss, as in the ablation) on the nowcast and forecast
steps, and the change in the scaled predicted quantiles from `given`: its
max and mean absolute value over target rows, quantiles and steps. A
change of exactly zero means the output is bit-identical.

    python scripts/slots_check.py score OUT.csv --slots ADAPTER \
        --label LABEL [--noslots ADAPTER] [--shard I K] [--limit N]
    python scripts/slots_check.py synthetic OUT_DIR [--seed S]
    python scripts/slots_check.py summarise SHARD_DIR --label LABEL \
        --commit SHA --csv OUT.csv --blocks OUT.csv.gz
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import covariate_ablation as ca
import finetune_chronos as ft
import finetune_stream as fs

from bdbvhubrenewal.forecasters import slots
from bdbvhubrenewal.forecasters.finetune import AHEAD_WEEKS

CONDITIONS = ("given", "swap_places", "removed", "shuffle_places")
"""Conditions scored; the change in output is taken from `given`."""

AHEAD_STEPS = {"weekly": AHEAD_WEEKS, "daily": 7 * AHEAD_WEEKS}
"""Forecast steps after the nowcast steps of a German block."""

NOSLOTS = Path("~/bdbv-rounds/checkpoints/n1000/finetuned-ckpt")
"""The no-slots adapter the ablation used (release chronos-ft-de129103cacc)."""

SIM_DIR = Path("~/bdbv-rounds/rounds/r1/sims")
HELD = (30140, 30143)
GERMANY = Path("~/germany-ft/realdata/germany.pkl")

WINDOWS = ("nowcast", "forecast")


def model_ids(model, block: dict) -> np.ndarray | None:
    """Return each row's slot as inference assigns it, None without slots.

    The slots depend only on the block's shape, the targets then each
    place's lags in turn, so they stay with the rows' positions whatever
    content a condition moved into them.
    """
    if not slots.has_slots(model):
        return None
    return fs.block_slots([block], slots.n_slots(model))


def _batch(block: dict) -> dict:
    """Return Chronos-2's validation batch of a block, as the trainer does."""
    from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode

    data = Chronos2Dataset(
        inputs=[fs.tensors(block)],
        context_length=ft.CONTEXT,
        prediction_length=block["prediction_length"],
        batch_size=10**6,
        output_patch_size=ft.OUTPUT_PATCH,
        mode=DatasetMode.VALIDATION,
    )
    return {**next(iter(data)), "n_targets": block["n_targets"]}


def _forward(model, batch: dict):
    """Run the model on a batch, with whatever slots are assigned."""
    import torch

    with torch.no_grad():
        return model(
            context=batch["context"],
            group_ids=batch["group_ids"],
            future_covariates=batch["future_covariates"],
            num_output_patches=batch["num_output_patches"],
        )


def _scales(model, batch: dict):
    n = batch["n_targets"]
    _, loc_scale = model.instance_norm(batch["context"][:n].float())
    return loc_scale


def _scaled(model, batch: dict, out):
    """Return the targets' quantiles `[n, Q, P]`, scaled by their context."""
    n = batch["n_targets"]
    loc, scale = _scales(model, batch)
    preds = out.quantile_preds[:n].float()
    p, _ = model.instance_norm(preds, (loc[:, :, None], scale[:, :, None]))
    return p


def predict(model, block: dict):
    """Return the loss `[n, P]` and the scaled quantiles `[n, Q, P]`.

    The loss is Chronos-2's: twice the pinball loss summed over its
    quantile levels, on the target scaled by its own context.
    """
    import torch

    batch = _batch(block)
    with slots.assigned(model, model_ids(model, block)):
        out = _forward(model, batch)
    p = _scaled(model, batch, out)
    n = batch["n_targets"]
    target = batch["future_target"][:n].float()
    t, _ = model.instance_norm(target, _scales(model, batch))
    q = model.quantiles.float()[None, :, None]
    t = t[:, None, :]
    loss = 2 * torch.abs((t - p) * ((t <= p).float() - q))
    return loss.sum(dim=1), p


def output_change(given, other, d: int, h: int) -> dict[str, float]:
    """Return the max and mean absolute change of the scaled quantiles.

    Over the target rows, quantile levels and the nowcast steps (`:d`) or
    forecast steps (`d:h`). Zero means the output is bit-identical.
    """
    diff = (other - given).abs()
    out = {}
    for window, part in (
        ("nowcast", diff[:, :, :d]),
        ("forecast", diff[:, :, d:h]),
    ):
        out[f"{window}_max"] = float(part.max()) if part.numel() else np.nan
        out[f"{window}_mean"] = float(part.mean()) if part.numel() else np.nan
    return out


def synthetic_adapter(
    model, out: Path, n_slots: int, seed: int, scale: float
) -> Path:
    """Save an adapter whose LoRA is a no-op and whose slots are random.

    The adapter is made as the trainer makes one (`finetune_stream`'s
    `lora_config`), so it loads through `slots.load_adapter` as a real
    slots checkpoint does. LoRA's B starts at zero, so merging changes no
    weight; only the slots, drawn N(0, scale^2), differ from base.
    """
    import torch
    from peft import get_peft_model

    slots.attach(model, n_slots)
    weight = getattr(model, slots.SLOT_MODULE).embedding.weight
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        weight.copy_(scale * torch.randn(weight.shape, generator=gen))
    peft_model = get_peft_model(model, fs.lora_config(n_slots))
    out = Path(out)
    peft_model.save_pretrained(out)
    (out / "settings.json").write_text(
        json.dumps(
            {"hash": "synthetic", "seed": seed, "scale": scale, "n": n_slots}
        )
    )
    return out


def _horizon(block: dict) -> tuple[int, int]:
    """Return a padded block's start (steps of context) and horizon."""
    c, n = ft.CONTEXT, block["n_targets"]
    seen = ~np.isnan(block["context"][:n])
    first = int(seen[:, :c].any(axis=0).argmax())
    horizon = block["lag"] + AHEAD_STEPS[block["source"]]
    last = int(np.flatnonzero(seen[:, c:].any(axis=0)).max()) + 1
    if last != horizon:
        raise ValueError(
            f"block cut {block.get('cut')} lag {block['lag']}: truth ends "
            f"{last} steps on, not at the horizon {horizon}"
        )
    return c - first, horizon


def germany_blocks(
    path: Path, per_source: int, max_rows: int, seed: int
) -> list[dict]:
    """Return a sample of the German states blocks with start and horizon.

    Blocks of the national total and the 16 states with at most
    `max_rows` series, then up to `per_source` of each source drawn
    without replacement (`seed`), kept in their order.
    """
    blocks = pickle.loads(Path(path).expanduser().read_bytes())
    rng = np.random.default_rng(seed)
    out = []
    for source in ("weekly", "daily"):
        mine = [
            b
            for b in blocks
            if b["places"] == "states"
            and b["source"] == source
            and b["context"].shape[0] <= max_rows
        ]
        take = rng.choice(len(mine), min(per_source, len(mine)), replace=False)
        for i in sorted(take):
            b = mine[i]
            start, horizon = _horizon(b)
            out.append(
                {
                    **b,
                    "start": start,
                    "horizon": horizon,
                    "set": "germany",
                    "id": f"de-{source}-{b['cut']}-D{b['lag']}",
                }
            )
    return out


def sim_sample(
    sim_dir: Path,
    held: tuple[int, int],
    n_sims: int,
    per_source: int,
    max_rows: int,
) -> list[dict]:
    """Return the held-out sim blocks the covariate ablation sampled."""
    sims = ca.sim_blocks(Path(sim_dir).expanduser(), list(held), n_sims)
    rng = np.random.default_rng(1)
    out = []
    for source in ("weekly", "daily", "3day"):
        mine = [
            b
            for b in sims
            if b["source"] == source and b["context"].shape[0] <= max_rows
        ]
        take = rng.choice(len(mine), min(per_source, len(mine)), replace=False)
        out.extend({**mine[i], "set": "sim"} for i in sorted(take))
    return out


def _slot_weight_check(model, adapter: Path) -> None:
    """Raise unless the model's slots are the adapter file's."""
    from safetensors import safe_open

    weight = getattr(model, slots.SLOT_MODULE).embedding.weight.detach()
    with safe_open(Path(adapter) / "adapter_model.safetensors", "pt") as f:
        key = next(k for k in f.keys() if slots.SLOT_MODULE in k)  # noqa: SIM118
        saved = f.get_tensor(key)
    if not np.array_equal(weight.numpy(), saved.numpy()):
        raise ValueError(f"{adapter}: loaded slots differ from the file's")


def load_models(noslots: Path, slotted: Path) -> dict:
    """Return zero-shot Chronos-2 and both adapters merged into it."""
    from chronos import Chronos2Pipeline

    from bdbvhubrenewal.forecasters.foundation import (
        CHRONOS_MODEL,
        CHRONOS_REVISION,
    )

    def base():
        return Chronos2Pipeline.from_pretrained(
            CHRONOS_MODEL, revision=CHRONOS_REVISION, device_map="cpu"
        ).model

    models = {
        "zero-shot": base().eval(),
        "n1000": slots.load_adapter(base(), noslots).eval(),
        "slots": slots.load_adapter(base(), slotted).eval(),
    }
    if slots.has_slots(models["n1000"]):
        raise ValueError(f"{noslots} holds slots")
    if not slots.has_slots(models["slots"]):
        raise ValueError(f"{slotted} holds no slots")
    _slot_weight_check(models["slots"], slotted)
    return models


def all_blocks(args) -> list[dict]:
    """Return every block of the run, each with its `key`, cached.

    Cutting the simulations takes minutes, so the blocks are kept in
    `args.cache` with the arguments that chose them, and cut again only
    if those differ.
    """
    chosen = {
        k: str(getattr(args, k))
        for k in (
            "sim_dir",
            "held",
            "sims",
            "per_source",
            "max_rows",
            "germany",
            "germany_per_source",
            "germany_max_rows",
            "seed",
        )
    }
    cache = Path(args.cache).expanduser() if args.cache else None
    if cache is not None and cache.exists():
        saved = pickle.loads(cache.read_bytes())
        if saved["chosen"] == chosen:
            return saved["blocks"]
    blocks = []
    if args.sims:
        blocks += sim_sample(
            args.sim_dir, args.held, args.sims, args.per_source, args.max_rows
        )
    if args.germany_per_source:
        blocks += germany_blocks(
            args.germany,
            args.germany_per_source,
            args.germany_max_rows,
            args.seed,
        )
    for k, b in enumerate(blocks):
        b["key"] = k
    if cache is not None:
        tmp = cache.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps({"chosen": chosen, "blocks": blocks}))
        tmp.replace(cache)
    return blocks


def score(args) -> None:
    """Score the blocks of one shard under every condition and model."""
    import torch

    torch.set_num_threads(1)
    blocks = all_blocks(args)
    i, k = args.shard
    blocks = blocks[i::k]
    if args.limit:
        # A dry run: the first blocks of each set, multi-place ones first,
        # since a swap needs two places.
        by_set: dict[str, int] = {}
        keep = []
        for b in sorted(blocks, key=lambda b: b["n_targets"] == 1):
            if by_set.get(b["set"], 0) < args.limit:
                keep.append(b)
                by_set[b["set"]] = by_set.get(b["set"], 0) + 1
        blocks = keep
    print(
        f"{args.label} shard {i}/{k}: {len(blocks)} blocks, rows "
        f"{[b['context'].shape[0] for b in blocks]}",
        flush=True,
    )
    models = load_models(
        Path(args.noslots).expanduser(), Path(args.slots).expanduser()
    )
    rows = []
    t0 = time.time()
    for j, block in enumerate(blocks):
        d, h = ca.lag_of(block), block["horizon"]
        given = {}
        for cond in CONDITIONS:
            rng = np.random.default_rng(
                [block["key"], ca.CONDITIONS.index(cond)]
            )
            x = ca.condition(block, cond, rng)
            if x is None:
                continue
            for name, model in models.items():
                t1 = time.time()
                loss, p = predict(model, x)
                loss = loss.numpy()
                if cond == "given":
                    given[name] = p
                change = output_change(given[name], p, d, h)
                rows.append(
                    {
                        "key": block["key"],
                        "id": block["id"],
                        "set": block["set"],
                        "source": block.get("source", "weekly"),
                        "n_targets": block["n_targets"],
                        "rows": block["context"].shape[0],
                        "lag": d,
                        "horizon": h,
                        "model": name,
                        "condition": cond,
                        "nowcast": float(loss[:, :d].mean()),
                        "forecast": float(loss[:, d:h].mean()),
                        **{f"change_{k}": v for k, v in change.items()},
                        "seconds": time.time() - t1,
                    }
                )
        print(
            f"{args.label} block {j + 1}/{len(blocks)} "
            f"rows {block['context'].shape[0]} {time.time() - t0:.0f}s",
            flush=True,
        )
        pd.DataFrame(rows).to_csv(args.out, index=False)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"{args.label} shard {i} done {time.time() - t0:.0f}s", flush=True)


def groups(d: pd.DataFrame) -> pd.DataFrame:
    """Return the rows under each summary set (a row may sit in several)."""
    multi = np.where(d["n_targets"] > 1, "multi-place", "univariate")
    parts = [
        d[d["set"] == "sim"].assign(group="sims all"),
        d[d["set"] == "sim"].assign(group="sims " + multi[d["set"] == "sim"]),
        d[d["set"] == "germany"].assign(group="germany states"),
    ]
    de = d[d["set"] == "germany"]
    parts.append(de.assign(group="germany states " + de["source"]))
    return pd.concat(parts, ignore_index=True)


def summarise_rows(d: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    """Return the paired summary: each condition against `given`.

    Ratio = mean loss under the condition / mean under `given`, over the
    blocks scored under both; CI = 90% bootstrap over blocks; share_worse
    = share of blocks whose loss rose; output_change_max and _mean = the
    largest and the mean absolute change in the scaled quantiles.
    """
    rng = np.random.default_rng(1)
    d = groups(d)
    key = ["group", "model", "key"]
    rows = []
    for window in WINDOWS:
        given = d[d["condition"] == "given"].set_index(key)[window]
        for (group, model, cond), part in d[d["condition"] != "given"].groupby(
            ["group", "model", "condition"], sort=False
        ):
            part = part.set_index(key)
            g = given.loc[part.index].to_numpy()
            c = part[window].to_numpy()
            ok = np.isfinite(g) & np.isfinite(c)
            g, c = g[ok], c[ok]
            if not len(g):
                continue
            idx = rng.integers(0, len(g), (n_boot, len(g)))
            boot = c[idx].mean(1) / g[idx].mean(1)
            change_max = part[f"change_{window}_max"].to_numpy()[ok]
            rows.append(
                {
                    "set": group,
                    "model": model,
                    "condition": cond,
                    "window": window,
                    "given_loss": g.mean(),
                    "condition_loss": c.mean(),
                    "ratio": c.mean() / g.mean(),
                    "ci_low": np.quantile(boot, 0.05),
                    "ci_high": np.quantile(boot, 0.95),
                    "share_worse": float((c > g + 1e-9).mean()),
                    "n_blocks": len(g),
                    "n_identical": int((change_max == 0).sum()),
                    "output_change_max": float(change_max.max()),
                    "output_change_mean": float(
                        part[f"change_{window}_mean"].to_numpy()[ok].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarise(args) -> None:
    """Write the summary and per-block CSVs of a run's shards."""
    shards = sorted(Path(args.shard_dir).glob("shard-*.csv"))
    d = pd.concat((pd.read_csv(f) for f in shards), ignore_index=True)
    d = d.assign(checkpoint=args.label, commit=args.commit)
    s = summarise_rows(d).assign(checkpoint=args.label, commit=args.commit)
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(args.csv, index=False, float_format="%.6g")
    d.to_csv(args.blocks, index=False, float_format="%.6g")
    print(headline(s))


def headline(s: pd.DataFrame) -> str:
    """Return a one-line-per-set reading of the swap for the mailbox."""
    out = []
    swap = s[s["condition"] == "swap_places"]
    for (group, window), part in swap.groupby(["set", "window"], sort=False):
        if group not in ("sims multi-place", "germany states"):
            continue
        cells = [
            f"{r.model} x{r.ratio:.2f} ({r.ci_low:.2f}-{r.ci_high:.2f}) "
            f"identical {r.n_identical}/{r.n_blocks} "
            f"change max {r.output_change_max:.3g}"
            for r in part.itertuples()
        ]
        out.append(f"{group} {window} swap: " + "; ".join(cells))
    return "\n".join(out)


def synthetic(args) -> None:
    """Save a synthetic slots adapter on base Chronos-2."""
    from chronos import Chronos2Pipeline

    from bdbvhubrenewal.forecasters.foundation import (
        CHRONOS_MODEL,
        CHRONOS_REVISION,
    )

    model = Chronos2Pipeline.from_pretrained(
        CHRONOS_MODEL, revision=CHRONOS_REVISION, device_map="cpu"
    ).model
    synthetic_adapter(model, args.out, slots.N_SLOTS, args.seed, args.scale)
    print(f"synthetic slots adapter in {args.out}")


def main() -> None:
    """Run a subcommand."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("score")
    p.add_argument("out", type=Path)
    p.add_argument("--slots", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--noslots", default=str(NOSLOTS))
    p.add_argument("--sim-dir", default=str(SIM_DIR))
    p.add_argument("--held", type=int, nargs=2, default=HELD)
    p.add_argument("--sims", type=int, default=30)
    p.add_argument("--per-source", type=int, default=16)
    p.add_argument("--max-rows", type=int, default=150)
    p.add_argument("--germany", default=str(GERMANY))
    p.add_argument("--germany-per-source", type=int, default=16)
    p.add_argument("--germany-max-rows", type=int, default=300)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--shard", type=int, nargs=2, default=(0, 1))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--cache", default="")
    p.set_defaults(run=score)
    p = sub.add_parser("synthetic")
    p.add_argument("out", type=Path)
    p.add_argument("--seed", type=int, default=1)
    # Chronos-2's patch tokens have an sd of about 0.03; slots of a third
    # of that leave the loss near base's, larger ones swamp the tokens.
    p.add_argument("--scale", type=float, default=0.01)
    p.set_defaults(run=synthetic)
    p = sub.add_parser("summarise")
    p.add_argument("shard_dir", type=Path)
    p.add_argument("--label", required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--blocks", type=Path, required=True)
    p.set_defaults(run=summarise)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
