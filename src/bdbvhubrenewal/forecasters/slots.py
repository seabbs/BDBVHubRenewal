# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Per-series slots: tie each place's lag series to its count series.

Chronos-2's group attention has no series identity, so in the ragged
layout's joint block it cannot tell which place a lag series belongs to.
A slot is a learned vector, one of `N_SLOTS`, added to every token of a
series (its context patches, its future patches and its [REG] token). A
place's count series and all its lag series share one slot. Slots are
drawn as a fresh random permutation per block in training, so the model
learns which series pair rather than fixed identities; at inference and
in held-out losses they are drawn the same way with the fixed seed
`EVAL_SEED`.

The slots are a module of the model, `series_slots`, initialised at zero,
so a model with slots starts exactly as base Chronos-2. It trains beside
the LoRA adapter as one of PEFT's `modules_to_save`, so it is saved in,
and loaded from, the adapter's files (`load_adapter`).

`attach` adds the slots to a Chronos-2 model and hooks its input patch
embedding and [REG] embedding. `assigned` sets the slot of each row of
the next batch; with no slots set, or none attached, the model is
unchanged. Rows are the ragged layout's order: the targets, then each
place's lags in turn (`row_places`). Chronos-2's long-horizon unrolling
repeats each row, and the slots repeat with it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

import numpy as np

__all__ = [
    "EVAL_SEED",
    "N_SLOTS",
    "SLOT_MODULE",
    "adapter_slots",
    "assigned",
    "attach",
    "has_slots",
    "load_adapter",
    "n_slots",
    "row_places",
    "slot_ids",
]

N_SLOTS: Final = 128
"""Slots. A training block can hold about 95 targets (the country, place
and age totals and up to 60 leaves), so 64 is too few."""

SLOT_MODULE: Final = "series_slots"
"""Name of the slots module, PEFT's `modules_to_save` entry."""

EVAL_SEED: Final = 0
"""Seed of the slots at inference and in held-out losses."""

_STATE: Final = "_series_slot_state"
"""Attribute of a model with slots holding the next batch's slots."""


def row_places(n_targets: int, n_covariates: int) -> np.ndarray:
    """Return the place of each row of a ragged block.

    Rows are the `n_targets` targets, then each place's lag series in
    turn, `n_covariates // n_targets` per place.

    Raises
    ------
    ValueError
        If the covariates do not split evenly over the places.
    """
    if n_covariates % n_targets:
        raise ValueError(
            f"{n_covariates} covariates do not split over {n_targets} places"
        )
    lags = n_covariates // n_targets
    return np.concatenate(
        [np.arange(n_targets), np.repeat(np.arange(n_targets), lags)]
    )


def slot_ids(
    n_targets: int,
    n_covariates: int,
    rng: np.random.Generator,
    n_slots: int = N_SLOTS,
) -> np.ndarray:
    """Return each row's slot: a random distinct slot per place.

    Raises
    ------
    ValueError
        If the block has more places than slots.
    """
    if n_targets > n_slots:
        raise ValueError(f"{n_targets} places but only {n_slots} slots")
    chosen = np.asarray(rng.permutation(n_slots))[:n_targets]
    out: np.ndarray = chosen[row_places(n_targets, n_covariates)]
    return out


def _slots_module(n_slots: int, d_model: int) -> Any:
    import torch
    from torch import nn

    class SeriesSlots(nn.Module):
        """One learned vector per slot, zero at the start."""

        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(n_slots, d_model)
            nn.init.zeros_(self.embedding.weight)

        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            out: torch.Tensor = self.embedding(ids)
            return out

    return SeriesSlots()


def _base(model: Any) -> Any | None:
    """Return the Chronos-2 model holding slots inside `model`, if any."""
    for module in model.modules():
        if hasattr(module, _STATE):
            return module
    return None


def n_slots(model: Any) -> int:
    """Return the slots of a model with slots attached."""
    base = _base(model)
    if base is None:
        raise ValueError("the model has no slots")
    module = getattr(base, SLOT_MODULE)
    # PEFT wraps the module while it trains; its copy has the same size.
    return int(
        next(p for n, p in module.named_parameters() if "embedding" in n).shape[
            0
        ]
    )


def has_slots(model: Any) -> bool:
    """Whether a model (or a PEFT model around it) has slots attached."""
    return _base(model) is not None


def attach(model: Any, n_slots: int = N_SLOTS) -> None:
    """Add zero slots to a Chronos-2 model and hook its embeddings."""
    if has_slots(model):
        raise ValueError("the model already has slots")
    d_model = model.config.d_model
    weight = next(model.input_patch_embedding.parameters())
    module = _slots_module(n_slots, d_model).to(weight.device, weight.dtype)
    setattr(model, SLOT_MODULE, module)
    setattr(model, _STATE, {"ids": None})

    def add(_module: Any, _args: Any, output: Any) -> Any:
        ids = getattr(model, _STATE)["ids"]
        if ids is None:
            return output
        rows = output.shape[0]
        if rows % len(ids):
            raise ValueError(f"{rows} rows but slots for {len(ids)}")
        ids = ids.to(output.device).repeat_interleave(rows // len(ids))
        vectors = getattr(model, SLOT_MODULE)(ids)
        return output + vectors[:, None, :].to(output.dtype)

    model.input_patch_embedding.register_forward_hook(add)
    model.shared.register_forward_hook(add)


@contextmanager
def assigned(model: Any, ids: Any) -> Iterator[None]:
    """Give the rows of the next batches these slots; no-op without slots."""
    import torch

    base = _base(model)
    if base is None or ids is None:
        yield
        return
    state = getattr(base, _STATE)
    state["ids"] = torch.as_tensor(np.asarray(ids), dtype=torch.long)
    try:
        yield
    finally:
        state["ids"] = None


def adapter_slots(adapter: Path | str) -> int | None:
    """Return the slots an adapter directory holds, or None."""
    adapter = Path(adapter)
    config = json.loads((adapter / "adapter_config.json").read_text())
    if SLOT_MODULE not in (config.get("modules_to_save") or []):
        return None
    from safetensors import safe_open

    with safe_open(adapter / "adapter_model.safetensors", "pt") as f:
        for key in f.keys():  # noqa: SIM118
            if SLOT_MODULE in key and key.endswith("weight"):
                return int(f.get_slice(key).get_shape()[0])
    raise ValueError(f"{adapter} lists {SLOT_MODULE} but holds no weights")


def load_adapter(model: Any, adapter: Path | str) -> Any:
    """Return Chronos-2 `model` with an adapter merged, slots included.

    Slots are attached first when the adapter holds them, so PEFT loads
    their trained values; an adapter without slots loads as before.
    """
    from peft import PeftModel

    n = adapter_slots(adapter)
    if n is not None:
        attach(model, n)
    return PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
