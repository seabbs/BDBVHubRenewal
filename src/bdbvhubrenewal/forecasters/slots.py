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
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

import numpy as np

__all__ = [
    "EVAL_SEED",
    "N_SLOTS",
    "SLOT_MODULE",
    "adapter_parent_link",
    "adapter_slots",
    "assigned",
    "attach",
    "has_parent_link",
    "has_slots",
    "load_adapter",
    "n_slots",
    "parents_of",
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

COUNTRIES: Final = ("CD", "DE")
"""Codes of the whole country (BDBV, Germany): the parent of a place with
no nearer parent in its block."""


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


def parents_of(places: Sequence[str]) -> list[int]:
    """Return the index of each place's parent in `places`, -1 for none.

    A place's parent is its nearest prefix in the block, cut at a "-"
    (`R1-P03` in `R1`, `CD-NK-03` in `CD-NK`), else the country
    (`COUNTRIES`) when the block holds it, as for provinces, states, age
    groups and age totals. The country has none. Parent means belongs to:
    a parent need not be the sum of its children.
    """
    at = {p: i for i, p in enumerate(places)}
    root = next((at[c] for c in COUNTRIES if c in at), -1)
    out = []
    for p in places:
        parts = p.split("-")
        near = next(
            (
                at["-".join(parts[:k])]
                for k in range(len(parts) - 1, 0, -1)
                if "-".join(parts[:k]) in at
            ),
            None,
        )
        out.append(near if near is not None else -1 if at[p] == root else root)
    return out


def slot_ids(
    n_targets: int,
    n_covariates: int,
    rng: np.random.Generator,
    n_slots: int = N_SLOTS,
    parents: Sequence[int] | None = None,
) -> np.ndarray:
    """Return each row's slot: a random distinct slot per place.

    With `parents` (`parents_of`), each row also carries its place's
    parent's own slot, -1 for none: `[2, rows]`, own then parent.

    Raises
    ------
    ValueError
        If the block has more places than slots.
    """
    if n_targets > n_slots:
        raise ValueError(f"{n_targets} places but only {n_slots} slots")
    chosen = np.asarray(rng.permutation(n_slots))[:n_targets]
    rows = (
        row_places(n_targets, n_covariates)
        if n_covariates
        else np.arange(n_targets)
    )
    own: np.ndarray = chosen[rows]
    if parents is None:
        return own
    parent = np.array([chosen[q] if q >= 0 else -1 for q in parents], dtype=int)
    return np.stack([own, parent[rows]])


def _slots_module(n_slots: int, d_model: int, parent: bool = True) -> Any:
    import torch
    from torch import nn

    class SeriesSlots(nn.Module):
        """One learned vector per slot, zero at the start."""

        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(n_slots, d_model)
            nn.init.zeros_(self.embedding.weight)
            if parent:
                # Marks slots that add each place's parent's; not trained.
                self.register_buffer("parent_link", torch.ones(()))

        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            """Own slots `[rows]`, or own and parents' `[2, rows]`."""
            if ids.ndim == 2 and not hasattr(self, "parent_link"):
                ids = ids[0]
            if ids.ndim == 1:
                out: torch.Tensor = self.embedding(ids)
                return out
            own, parent = ids
            linked = (parent >= 0).unsqueeze(-1)
            vectors = self.embedding(parent.clamp(min=0)) * linked
            return self.embedding(own) + vectors

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


def attach(
    model: Any,
    n_slots: int = N_SLOTS,
    init_std: float = 0.0,
    seed: int = 0,
    parent: bool = True,
) -> None:
    """Add slots to a Chronos-2 model and hook its embeddings.

    The slots start at zero, so the model is unchanged, or with
    `init_std` as distinct codes drawn from a normal with that sd, seeded
    by `seed`. Loading an adapter that holds slots replaces them. With
    `parent`, each place also adds its parent's slot.
    """
    import torch

    if has_slots(model):
        raise ValueError("the model already has slots")
    d_model = model.config.d_model
    weight = next(model.input_patch_embedding.parameters())
    module = _slots_module(n_slots, d_model, parent)
    if init_std:
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            module.embedding.weight.normal_(0.0, init_std, generator=generator)
    module = module.to(weight.device, weight.dtype)
    setattr(model, SLOT_MODULE, module)
    setattr(model, _STATE, {"ids": None})

    def add(_module: Any, _args: Any, output: Any) -> Any:
        ids = getattr(model, _STATE)["ids"]
        if ids is None:
            return output
        rows, n = output.shape[0], ids.shape[-1]
        if rows % n:
            raise ValueError(f"{rows} rows but slots for {n}")
        ids = ids.to(output.device).repeat_interleave(rows // n, dim=-1)
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


def has_parent_link(model: Any) -> bool:
    """Whether a model's slots add each place's parent's."""
    base = _base(model)
    if base is None:
        return False
    return hasattr(getattr(base, SLOT_MODULE), "parent_link")


def adapter_parent_link(adapter: Path | str) -> bool:
    """Whether an adapter's slots hold the parent link (`parent_link`)."""
    from safetensors import safe_open

    with safe_open(Path(adapter) / "adapter_model.safetensors", "pt") as f:
        return any("parent_link" in key for key in f.keys())  # noqa: SIM118


def load_adapter(model: Any, adapter: Path | str) -> Any:
    """Return Chronos-2 `model` with an adapter merged, slots included.

    Slots are attached first when the adapter holds them, so PEFT loads
    their trained values; an adapter without slots loads as before.
    """
    from peft import PeftModel

    n = adapter_slots(adapter)
    if n is not None:
        attach(model, n, parent=adapter_parent_link(adapter))
    return PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
