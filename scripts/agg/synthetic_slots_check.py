"""Fit one BDBV arm with a synthetic slots adapter, end to end.

Builds an adapter as a slots run saves one: Chronos-2 with slots attached
(`slots.attach`), random nonzero slot vectors and a LoRA adapter of
random weights, saved by PEFT with the slots as `modules_to_save`. The
`_ft` model then loads it through `slots.load_adapter` and fits the arm
(`groupings.release_grouping_forecasts`) at one origin, so a block over
the slots goes through `cased_group_slots` and Chronos-2's row order.

Checks, for every group fitted: the ids are those `cased_group_slots`
gives; a block over the slots gives slots to its places with cases only
and -1 to the others; each lag row has its place's slot in Chronos-2's
sorted row order; and the draws differ from zero-shot's, so the slots
took effect.

    python scripts/agg/synthetic_slots_check.py DATA OUT [--model M]
        [--grouping G] [--origin O]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bdbvhubrenewal.data import load_daily, load_delays, load_target
from bdbvhubrenewal.data.zones import load_zone_data
from bdbvhubrenewal.evaluation import backtest as bt
from bdbvhubrenewal.evaluation import groupings as gr
from bdbvhubrenewal.evaluation import store
from bdbvhubrenewal.forecasters import foundation, slots

LABEL = "slots-000000000000"


def build_adapter(out: Path) -> None:
    """Save a slots adapter with random nonzero slots and LoRA weights."""
    import torch
    from peft import LoraConfig, get_peft_model

    base = foundation._chronos.__wrapped__(None).model
    slots.attach(base, slots.N_SLOTS)
    config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["self_attention.q", "self_attention.v"],
        modules_to_save=[slots.SLOT_MODULE],
    )
    model = get_peft_model(base, config)
    torch.manual_seed(1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(0, 1e-3)
            if slots.SLOT_MODULE in name and "embedding" in name:
                p.normal_(0, 0.5)
    model.save_pretrained(out)
    assert slots.adapter_slots(out) == slots.N_SLOTS


def main() -> None:
    """Build the adapter, fit the arm and check its slots."""
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("out")
    parser.add_argument("--model", default="fm_chronos_weekly_ragged_ft")
    parser.add_argument("--grouping", default="joint")
    parser.add_argument("--origin", default="2026-10-17")
    args = parser.parse_args()
    out = Path(args.out)
    adapter = out / "adapter"
    if not (adapter / "adapter_config.json").exists():
        build_adapter(adapter)
    checkpoint = foundation.Checkpoint(LABEL, ())
    foundation.resolve_checkpoint = lambda _spec: checkpoint  # type: ignore[assignment]
    gr.resolve_checkpoint = foundation.resolve_checkpoint  # type: ignore[assignment]
    foundation.finetuned_adapter = lambda _c: adapter  # type: ignore[assignment]

    seen: list[tuple[dict, np.ndarray]] = []
    real = gr.cased_group_slots

    def spy(group: dict, n: int) -> np.ndarray:
        ids = real(group, n)
        seen.append((group, ids))
        return ids

    gr.cased_group_slots = spy  # type: ignore[assignment]

    target = load_target(args.data)
    config = bt.load_config("config/backtest.toml")
    tasks = store.origin_tasks(
        target,
        load_delays(args.data),
        config,
        load_daily(args.data),
        load_zone_data(args.data),
    )
    task = tasks[args.origin]
    run = gr.release_grouping_forecasts(
        [task], target, args.model, args.grouping, LABEL
    )
    assert slots.has_slots(foundation._chronos(checkpoint).model)
    zero = gr.release_grouping_forecasts(
        [task], target, args.model.removesuffix("_ft"), args.grouping
    )
    report = []
    for group, ids in seen:
        places = list(group["places"])
        n = len(places)
        context = np.nansum(np.atleast_2d(group["target"]), axis=1)
        past = list(group.get("past_covariates", {}))
        known = set(group.get("future_covariates", {}))
        rows = sorted(c for c in past if c not in known)
        rows += sorted(c for c in past if c in known)
        own = ids[:n]
        lagged = {c.rsplit("_lag_", 1)[0] for c in past}
        cased = (context > 0) | np.isin(places, list(lagged))
        if n > slots.N_SLOTS:
            assert ((own >= 0) == cased).all()
        else:
            assert (own >= 0).all()
        given = own[own >= 0]
        assert len(set(given)) == len(given)
        at = {p: i for i, p in enumerate(places)}
        want = [own[at[c.rsplit("_lag_", 1)[0]]] for c in rows]
        assert ids[n:].tolist() == want
        report.append(
            {
                "places": n,
                "with_slots": int((own >= 0).sum()),
                "lag_rows": len(rows),
                "first_rows": rows[:3],
            }
        )
    a = run.samples.sort_values(["location", "target_end_date", "sample"])
    b = zero.samples.sort_values(["location", "target_end_date", "sample"])
    differ = float(np.mean(a["value"].to_numpy() != b["value"].to_numpy()))
    assert differ > 0
    result = {"groups": report, "share_of_draws_differing": round(differ, 3)}
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print("ok")


if __name__ == "__main__":
    main()
