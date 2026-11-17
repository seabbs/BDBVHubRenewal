# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""What fine-tuned Chronos-2 learnt about reporting delays, for the SI.

Fine-tuned against zero-shot Chronos-2, on development data only.

`synthetic` uses held-out simulated blocks with daily steps, whose final
counts are known, so a block's true completeness at each day before its
cut-off is its final count over the count reported by the cut-off:

- completeness: by days before the cut-off $d$, the nowcast median over
  the count reported (the implied $1 / F(d)$) against the true final over
  reported;
- delay response: the lag rows shifted to imply a slower delay (the count
  reported at lag $d$ becomes the one reported at lag $d - k$), so the
  final counts are unchanged and the right nowcast is too: the ratio of
  the shifted nowcast to the given one, ideally 1, beside that of the
  reported counts, which a model blind to the delay would follow;
- lags: the nowcast loss with the lag rows against without them;
- scenarios: the nowcast loss by the delay-length and delay-change
  factors of the simulation;
- parts: the nowcast loss of the fine-tuned model with its biases zeroed
  (LoRA kept) and with its LoRA removed (biases kept).

`attention` reads the group attention: the learned biases by layer and
head, and on held-out simulated blocks and the German development data's
input blocks, the share of each count row's attention going to itself,
its place's other rows (its lags, its totals row), its parent, its
children and other places, by layer, and to its own lag $d$, by $d$.

The nowcast loss is twice the pinball loss averaged over Chronos-2's
quantile levels and the nowcast days, over the series' mean final count
plus one.

    python scripts/delay_learning.py synthetic OUT --adapter DIR \
        --blocks PKL --settings 'ROUNDS/r*/sims/settings-*.csv'
    python scripts/delay_learning.py attention OUT --adapter DIR \
        --blocks PKL --germany CSV --dates D1 D2 ...
"""

from __future__ import annotations

import argparse
import csv
import inspect
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

DAILY = ("daily", "daily+weekly", "daily+3day")
"""Block sources on daily steps, whose lag rows are daily."""

SHIFTS = (2, 4)
"""Days by which the lag rows are shifted to imply a slower delay."""

LIMIT_ROWS = 600
"""Largest block whose attention is read (the weights are rows^2)."""


# --- models -----------------------------------------------------------------


def _base():
    from chronos import Chronos2Pipeline

    from bdbvhubrenewal.forecasters.foundation import (
        CHRONOS_MODEL,
        CHRONOS_REVISION,
    )

    return Chronos2Pipeline.from_pretrained(
        CHRONOS_MODEL, revision=CHRONOS_REVISION, device_map="cpu"
    ).model


def models(adapter: Path) -> dict:
    """Return zero-shot, fine-tuned and the fine-tuned model's two parts.

    `biases zeroed` is the fine-tuned model with its biases set to zero
    (its LoRA kept); `LoRA removed` is base Chronos-2 with the fine-tuned
    biases. Zero-shot carries zero biases, so it is base Chronos-2.
    """
    import torch

    from bdbvhubrenewal.forecasters import series_bias as sb

    zero = _base()
    sb.attach(zero)
    tuned = sb.load_adapter(_base(), adapter)
    learnt = {
        n: p.detach().clone()
        for n, p in getattr(tuned, sb.BIAS_MODULE).named_parameters()
    }
    no_bias = sb.load_adapter(_base(), adapter)
    with torch.no_grad():
        for p in getattr(no_bias, sb.BIAS_MODULE).parameters():
            p.zero_()
    no_lora = _base()
    sb.attach(no_lora)
    with torch.no_grad():
        for n, p in getattr(no_lora, sb.BIAS_MODULE).named_parameters():
            p.copy_(learnt[n])
    out = {
        "zero-shot": zero,
        "fine-tuned": tuned,
        "biases zeroed": no_bias,
        "LoRA removed": no_lora,
    }
    return {k: m.eval() for k, m in out.items()}


# --- blocks -----------------------------------------------------------------


def rows_of(block: dict) -> np.ndarray:
    """Return a block's rows' places and parents (as the trainer)."""
    from bdbvhubrenewal.forecasters import series_bias as sb

    places = list(block["places"])
    return sb.block_rows(
        block.get("target_places", range(block["n_targets"])),
        block["n_covariates"],
        sb.parents_of(places),
    )


def predict(model, block: dict, attentions: bool = False):
    """Return the targets' quantiles `[n, Q, H]` (and the attention)."""
    import torch
    from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode

    from bdbvhubrenewal.forecasters import series_bias as sb

    context = torch.from_numpy(np.asarray(block["context"], dtype=np.float32))
    width = block["prediction_length"]
    data = Chronos2Dataset(
        inputs=[
            {
                **block,
                "context": context,
                "future_covariates": context[:, -width:],
            }
        ],
        context_length=context.shape[1] - width,
        prediction_length=width,
        batch_size=10**6,
        output_patch_size=16,
        mode=DatasetMode.VALIDATION,
    )
    batch = next(iter(data))
    keys = inspect.signature(type(model).forward).parameters
    inputs = {
        k: v for k, v in batch.items() if k in keys and k != "future_target"
    }
    if attentions:
        model.config._attn_implementation = "eager"
    with sb.assigned(model, rows_of(block)), torch.no_grad():
        out = model(**inputs, output_attentions=attentions)
    q = out.quantile_preds[: block["n_targets"]].float().numpy()
    return (q, out.enc_group_self_attn_weights) if attentions else q


def window(block: dict) -> tuple[int, int]:
    """Return the block's lag window $D$ and the first future column."""
    lag = int(block["meta"]["options"]["max_lag"])
    return lag, np.asarray(block["context"]).shape[1] - block[
        "prediction_length"
    ]


def reported(block: dict) -> np.ndarray:
    """Return each count row's count reported by the cut-off, `[L, D]`.

    Column $j$ is the $j$-th nowcast day; its count at the cut-off is the
    place's lag row $d = D - 1 - j$ there.
    """
    lag, at = window(block)
    x = np.asarray(block["context"])
    n_places = len(block["places"])
    out = np.full((n_places, lag), np.nan)
    for p in range(n_places):
        for j in range(lag):
            out[p, j] = x[block["n_targets"] + p * lag + (lag - 1 - j), at + j]
    return out


def truth(block: dict) -> np.ndarray:
    """Return each count row's final counts on the nowcast days, `[L, D]`."""
    lag, at = window(block)
    n_places = len(block["places"])
    return np.asarray(block["context"])[:n_places, at : at + lag]


def without_lags(block: dict) -> dict:
    """Return the block with no lag rows."""
    n = block["n_targets"]
    return {
        **block,
        "context": np.asarray(block["context"])[:n],
        "n_covariates": 0,
        "n_future_covariates": 0,
    }


def slower(block: dict, k: int) -> dict:
    """Return the block with its lag rows implying a delay $k$ days slower.

    The count reported at lag $d$ becomes the one reported at lag $d - k$
    (none for $d < k$), each still known only where lag $d$ was.
    """
    lag, _ = window(block)
    x = np.array(block["context"], dtype=float)
    n, n_places = block["n_targets"], len(block["places"])
    for p in range(n_places):
        rows = x[n + p * lag : n + (p + 1) * lag].copy()
        known = ~np.isnan(rows)
        moved = np.zeros_like(rows)
        moved[k:] = rows[: lag - k]
        moved[~known] = np.nan
        moved[:k][known[:k]] = 0.0
        x[n + p * lag : n + (p + 1) * lag] = moved
    return {**block, "context": x}


def loss(q: np.ndarray, y: np.ndarray, levels: np.ndarray) -> float:
    """Return the relative nowcast loss of one series (module docstring)."""
    ok = ~np.isnan(y)
    if not ok.any():
        return np.nan
    pin = [
        2 * np.abs((y[ok] - q[k, ok]) * ((y[ok] <= q[k, ok]) - lv))
        for k, lv in enumerate(levels)
    ]
    return float(np.mean(pin) / (np.abs(y[ok]).mean() + 1))


# --- synthetic --------------------------------------------------------------


def scenario_of(settings_glob: str) -> dict[str, dict[str, str]]:
    """Return each simulation's delay-length and delay-change factors."""
    out = {}
    for path in sorted(Path("/").glob(settings_glob.lstrip("/"))):
        with path.open() as f:
            records = list(csv.DictReader(f))
        for r in records:
            median = float(r["delay_median"] or "nan")
            out[r["outbreak"]] = {
                "delay_length": (
                    "short (<7 d)"
                    if median < 7
                    else ("medium (7-14 d)" if median <= 14 else "long (>14 d)")
                ),
                "delay_change": r["delay_change"],
            }
    return out


def synthetic(args) -> None:
    """Delay checks on held-out simulated blocks (module docstring)."""
    import torch

    torch.set_num_threads(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    blocks = [
        b
        for b in pickle.loads(Path(args.blocks).read_bytes())
        if b["source"] in DAILY
    ][: args.limit or None]
    scenarios = scenario_of(args.settings)
    built = models(Path(args.adapter))
    complete = defaultdict(lambda: np.zeros(3))
    rows = []
    for i, block in enumerate(blocks):
        y, r = truth(block), reported(block)
        n_places = len(block["places"])
        sim = block["outbreak"].rsplit("-", 1)[0]
        factors = scenarios.get(sim, {"delay_length": "?", "delay_change": "?"})
        for name, model in built.items():
            levels = model.quantiles.float().numpy()
            md = int(np.argmin(abs(levels - 0.5)))
            lag, _ = window(block)
            given = predict(model, block)[:n_places, :, :lag]
            entry = {"block": i, "model": name, **factors}
            entry["loss"] = np.nanmean(
                [loss(given[p], y[p], levels) for p in range(n_places)]
            )
            if name in ("zero-shot", "fine-tuned"):
                for p in range(n_places):
                    for j in range(lag):
                        if r[p, j] > 0 and not np.isnan(y[p, j]):
                            complete[(name, lag - 1 - j)] += (
                                given[p, md, j],
                                r[p, j],
                                y[p, j],
                            )
                bare = predict(model, without_lags(block))[:n_places, :, :lag]
                entry["loss_no_lags"] = np.nanmean(
                    [loss(bare[p], y[p], levels) for p in range(n_places)]
                )
                for k in SHIFTS:
                    shifted = slower(block, k)
                    moved = predict(model, shifted)[:n_places, md, :lag]
                    ok = given[:, md, :] > 0
                    entry[f"shift{k}_ratio"] = float(
                        np.median(moved[ok] / given[:, md, :][ok])
                    )
                    # How far the reported counts fell: the nowcast should
                    # not follow them, its ideal ratio is 1.
                    has = r > 0
                    entry[f"shift{k}_reported_ratio"] = float(
                        np.median(
                            np.nan_to_num(reported(shifted)[has]) / r[has]
                        )
                    )
            rows.append(entry)
        print(f"synthetic block {i + 1}/{len(blocks)}", flush=True)
    per_block = pd.DataFrame(rows)
    per_block.to_csv(out / "synthetic-blocks.csv.gz", index=False)
    comp = pd.DataFrame(
        [
            {
                "model": m,
                "days_before_cutoff": d,
                "implied": v[0] / v[1],
                "true": v[2] / v[1],
            }
            for (m, d), v in sorted(complete.items())
        ]
    )
    comp.to_csv(out / "completeness.csv", index=False)
    summarise(per_block, comp, out)


def summarise(per_block: pd.DataFrame, comp: pd.DataFrame, out: Path) -> None:
    """Write the small tables and the completeness figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    two = per_block[per_block["model"].isin(["zero-shot", "fine-tuned"])]
    lags = two.groupby("model")[["loss", "loss_no_lags"]].mean()
    lags["no_lags_over_lags"] = lags["loss_no_lags"] / lags["loss"]
    shifts = two.groupby("model")[
        [f"shift{k}_{w}" for k in SHIFTS for w in ("ratio", "reported_ratio")]
    ].median()
    pd.concat([lags, shifts], axis=1).round(3).to_csv(out / "delay-checks.csv")
    ref = per_block[per_block["model"] == "zero-shot"].set_index("block")[
        "loss"
    ]
    rel = per_block.assign(rel=per_block["loss"] / per_block["block"].map(ref))
    rel.groupby(["delay_length", "delay_change", "model"])["rel"].agg(
        ["median", "count"]
    ).round(3).to_csv(out / "scenarios.csv")
    per_block.groupby("model")["loss"].mean().round(4).to_csv(out / "parts.csv")
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    shown = comp[comp["days_before_cutoff"] < 21]
    truth_line = shown[shown["model"] == "fine-tuned"]
    ax.plot(
        truth_line["days_before_cutoff"],
        truth_line["true"],
        "k-",
        label="true 1 / F(d)",
    )
    for name, style in (("zero-shot", "C0--"), ("fine-tuned", "C1-")):
        part = shown[shown["model"] == name]
        ax.plot(part["days_before_cutoff"], part["implied"], style, label=name)
    ax.set_xlabel("days before the cut-off, d")
    ax.set_ylabel("nowcast / reported")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "completeness.png", dpi=200)


# --- attention --------------------------------------------------------------


KINDS = ("self", "own lags", "own totals", "parent", "children", "other places")
"""Where a count row's attention goes."""


def shares(block: dict, weights) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return each layer's attention shares and the lag use by $d$."""
    rows = rows_of(block)
    place, parent = rows
    n_rows = place.shape[0]
    n_places = len(block["places"])
    n = block["n_targets"]
    lag = block["n_covariates"] // n_places
    kind = np.empty((n_places, n_rows), dtype=object)
    for q in range(n_places):
        for k in range(n_rows):
            if k == q:
                kind[q, k] = "self"
            elif place[k] == q and k >= n:
                kind[q, k] = "own lags"
            elif place[k] == q:
                kind[q, k] = "own totals"
            elif place[k] == parent[q]:
                kind[q, k] = "parent"
            elif parent[k] == q:
                kind[q, k] = "children"
            else:
                kind[q, k] = "other places"
    by_layer, by_lag = [], []
    for layer, w in enumerate(weights):
        a = w.float().mean(dim=(0, 1)).numpy()[:n_places, :n_rows]
        for name in KINDS:
            by_layer.append(
                {
                    "layer": layer,
                    "kind": name,
                    "share": float(a[kind == name].sum() / n_places),
                }
            )
        for d in range(lag):
            cols = [n + q * lag + d for q in range(n_places)]
            by_lag.append(
                {
                    "layer": layer,
                    "d": d,
                    "share": float(np.mean(a[np.arange(n_places), cols])),
                }
            )
    return pd.DataFrame(by_layer), pd.DataFrame(by_lag)


def germany_blocks(path: str, dates: list[str]) -> list[dict]:
    """Return the German development data's daily input blocks, as forecast."""
    from bdbvhubrenewal.data.germany import hub_triangle, nowcast_tasks
    from bdbvhubrenewal.forecasters import foundation

    tri_df = hub_triangle(path)
    out = []
    for date in dates:
        for task in nowcast_tasks(tri_df, date, 1, 0):
            places = (foundation.NATIONAL, *task.locations)
            tri = foundation.hub_triangle(
                task.daily_history, "daily", places, task.target_end_dates[-1]
            )
            start = tri.n_context
            group = foundation.layout_block(
                tri, "ragged", places, start, foundation.week_rows(tri)
            )
            lags = tri.partial.shape[2]
            cells = tri.partial.transpose(0, 2, 1).reshape(
                len(places) * lags, -1
            )
            target = np.concatenate(
                [
                    group["target"],
                    np.full(
                        (group["target"].shape[0], len(tri.times) - start),
                        np.nan,
                    ),
                ],
                axis=1,
            )
            out.append(
                {
                    "context": np.concatenate([target, cells]),
                    "n_targets": target.shape[0],
                    "n_covariates": cells.shape[0],
                    "n_future_covariates": cells.shape[0],
                    "prediction_length": len(tri.times) - start,
                    "places": tuple(places),
                    "target_places": group["target_places"],
                    "source": "germany",
                }
            )
    return out


def attention(args) -> None:
    """Write the learned biases and the attention used (module docstring)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from bdbvhubrenewal.forecasters import series_bias as sb

    torch.set_num_threads(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    built = models(Path(args.adapter))
    bias = getattr(built["fine-tuned"], sb.BIAS_MODULE)
    names = [n for n, _ in bias.named_parameters()]
    fig, axes = plt.subplots(1, len(names), figsize=(3 * len(names), 3))
    top = max(float(p.abs().max()) for p in bias.parameters())
    for ax, (name, p) in zip(axes, bias.named_parameters(), strict=True):
        im = ax.imshow(p.detach().numpy(), cmap="RdBu_r", vmin=-top, vmax=top)
        ax.set_title(name.rstrip("_"))
        ax.set_xlabel("head")
    axes[0].set_ylabel("layer")
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(out / "biases.png", dpi=200, bbox_inches="tight")
    synthetic_blocks = [
        b
        for b in pickle.loads(Path(args.blocks).read_bytes())
        if b["source"] in DAILY and len(b["places"]) > 1
    ]
    sets = {
        "synthetic": synthetic_blocks[: args.limit or None],
        "germany": germany_blocks(args.germany, args.dates)
        if args.germany
        else [],
    }
    layers, lags = [], []
    for set_name, blocks in sets.items():
        for block in blocks:
            if np.asarray(block["context"]).shape[0] > LIMIT_ROWS:
                continue
            for name in ("zero-shot", "fine-tuned"):
                _, weights = predict(built[name], block, attentions=True)
                a, b = shares(block, weights)
                layers.append(a.assign(set=set_name, model=name))
                lags.append(b.assign(set=set_name, model=name))
            print(f"attention {set_name} block done", flush=True)
    layer_table = (
        pd.concat(layers)
        .groupby(["set", "model", "layer", "kind"])["share"]
        .mean()
    )
    layer_table.round(4).to_csv(out / "attention-shares.csv")
    lag_table = pd.concat(lags).groupby(["set", "model", "d"])["share"].mean()
    lag_table.round(4).to_csv(out / "lag-use.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    for ax, set_name in zip(axes, sets, strict=True):
        part = (
            lag_table.xs(set_name, level="set")
            if set_name in lag_table.index.get_level_values(0)
            else None
        )
        if part is None:
            continue
        for name, style in (("zero-shot", "C0--"), ("fine-tuned", "C1-")):
            s = part.xs(name, level="model")
            ax.plot(s.index, s.values, style, label=name)
        ax.set_title(set_name)
        ax.set_xlabel("own lag d")
    axes[0].set_ylabel("attention share (mean over layers)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "lag-use.png", dpi=200)


def main() -> None:
    """Run a subcommand."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    for name, run in (("synthetic", synthetic), ("attention", attention)):
        p = sub.add_parser(name)
        p.add_argument("out")
        p.add_argument("--adapter", required=True)
        p.add_argument("--blocks", required=True)
        p.add_argument("--limit", type=int, default=0)
        if name == "synthetic":
            p.add_argument("--settings", required=True)
        else:
            p.add_argument("--germany", default="")
            p.add_argument("--dates", nargs="*", default=[])
        p.set_defaults(run=run)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
