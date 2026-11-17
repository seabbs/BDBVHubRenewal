# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Attention bias that ties a place's series together and to its parent.

Chronos-2's group attention has no series identity, so in the ragged
layout's joint block it cannot tell which place a lag series belongs to.
This adds learned biases to the group attention scores of every encoder
layer, per head, one for each kind of row pair (`pair_masks`): a row and
itself, $b_\text{self}$; a row and another of its place (its count series
and all its lag series), $b_\text{same}$; a child's row and its parent's
(`parents_of`), $b_\text{up}$; and a parent's row and its child's,
$b_\text{down}$. The biases start at zero, so a model with them starts
exactly as base Chronos-2, are kept in fp32, and need no codes, no table
and no scale, for any number of places.

Chronos-2's encoder builds one additive group mask, `(time, 1, q, k)`
over the batch axis (`Chronos2Encoder._construct_and_invert_group_time_mask`),
which each layer's `GroupSelfAttention` adds to its scores. `attach`
hooks each `GroupSelfAttention` to add its layer's bias to that mask,
from the rows' places and parents set by `assigned`. Rows are the ragged
layout's order: the targets, then each place's lags in turn
(`row_places`); rows that Chronos-2 repeats for long horizons
repeat their place. The biases train beside the LoRA adapter as one of
PEFT's `modules_to_save`, so they are saved in, and loaded from, the
adapter's files.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

import numpy as np

__all__ = [
    "BIAS_MODULE",
    "COUNTRIES",
    "PAIRS",
    "adapter_has_bias",
    "assigned",
    "attach",
    "batch_rows",
    "block_rows",
    "has_bias",
    "load_adapter",
    "pair_masks",
    "parents_of",
    "row_places",
]

BIAS_MODULE: Final = "series_bias"
"""Name of the bias module, PEFT's `modules_to_save` entry."""

_STATE: Final = "_series_bias_state"
"""Attribute of a model with the bias holding the next batch's rows."""


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


def block_rows(
    target_places: Sequence[int], n_covariates: int, parents: Sequence[int]
) -> np.ndarray:
    """Return each row's place and its place's parent, `[2, rows]`.

    The targets are at `target_places` (indices into `parents`; a place
    may have more than one target row, as its daily counts and its period
    totals), then each place's lags in turn, `n_covariates` in all. A
    parent is -1 for none.

    Raises
    ------
    ValueError
        If the covariates do not split evenly over the places.
    """
    n_places = len(parents)
    if n_covariates % n_places:
        raise ValueError(
            f"{n_covariates} covariates do not split over {n_places} places"
        )
    lags = np.repeat(np.arange(n_places), n_covariates // n_places)
    rows = np.concatenate([np.asarray(target_places, dtype=int), lags])
    parent = np.asarray(parents, dtype=int)
    return np.stack([rows, parent[rows]])


def batch_rows(blocks: Sequence[np.ndarray]) -> np.ndarray:
    """Return the rows of blocks batched together, places kept apart."""
    out, offset = [], 0
    for place, parent in blocks:
        out.append(
            np.stack(
                [place + offset, np.where(parent >= 0, parent + offset, -1)]
            )
        )
        offset += int(place.max()) + 1 if len(place) else 0
    return np.concatenate(out, axis=1)


def pair_masks(place: Any, parent: Any) -> tuple[Any, Any, Any, Any]:
    """Return the four kinds of row pair, each `[rows, rows]`, query first.

    A row and itself; a row and another row of its place (its count and
    lag series); a child's row and its parent's (up); and a parent's row
    and its child's (down). A parent of -1 links nothing.
    """
    import torch

    me = torch.eye(len(place), dtype=torch.bool, device=place.device)
    same = (place[:, None] == place[None, :]) & ~me
    up = (parent[:, None] == place[None, :]) & (parent[:, None] >= 0)
    return me, same, up, up.T


PAIRS: Final = ("self_", "same", "up", "down")
"""The biases, one per kind of pair (`pair_masks`), in its order."""


def _bias_module(n_layers: int, n_heads: int) -> Any:
    import torch
    from torch import nn

    class SeriesBias(nn.Module):  # type: ignore[misc]
        """Per layer and head, the bias of each kind of row pair, fp32."""

        def __init__(self) -> None:
            super().__init__()
            for name in PAIRS:
                zero = torch.zeros(n_layers, n_heads, dtype=torch.float32)
                setattr(self, name, nn.Parameter(zero))

        def forward(self, layer: int, masks: Sequence[Any]) -> Any:
            """Return layer `layer`'s bias, `[heads, rows, rows]`, fp32."""
            out = 0
            for name, mask in zip(PAIRS, masks, strict=True):
                out = out + getattr(self, name)[layer][:, None, None] * mask
            return out

    return SeriesBias()


def _base(model: Any) -> Any | None:
    for module in model.modules():
        if hasattr(module, _STATE):
            return module
    return None


def has_bias(model: Any) -> bool:
    """Whether a model (or a PEFT model around it) has the bias attached."""
    return _base(model) is not None


def attach(model: Any) -> None:
    """Add a zero bias to a Chronos-2 model and hook its group attention.

    Each `GroupSelfAttention`'s forward pre-hook adds its layer's bias to
    the group mask, from the rows set by `assigned`. A block recomputed in
    backward (gradient checkpointing) recomputes the bias from the same
    rows, so they must stay assigned until backward is done; the bias is
    then never held per layer between forward and backward.
    """
    from chronos.chronos2.layers import GroupSelfAttention

    if has_bias(model):
        raise ValueError("the model already has the series bias")
    config = model.config
    weight = next(model.input_patch_embedding.parameters())
    module = _bias_module(config.num_layers, config.num_heads)
    # Kept in fp32 whatever the model's dtype; cast where it is added.
    setattr(model, BIAS_MODULE, module.to(weight.device))
    setattr(model, _STATE, {"rows": None})
    groups = [m for m in model.modules() if isinstance(m, GroupSelfAttention)]
    if len(groups) != config.num_layers:
        raise ValueError(
            f"{len(groups)} group attention layers, not {config.num_layers}"
        )

    def hook(layer: int) -> Callable[..., Any]:
        def add(
            _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
        ) -> Any:
            rows = getattr(model, _STATE)["rows"]
            if rows is None:
                return None
            hidden = args[0]
            mask = kwargs["attention_mask"]
            n_rows, n = hidden.shape[0], rows.shape[-1]
            if n_rows % n:
                raise ValueError(f"{n_rows} rows but places for {n}")
            rows = rows.to(hidden.device).repeat_interleave(n_rows // n, dim=-1)
            masks = pair_masks(rows[0], rows[1])
            bias = getattr(model, BIAS_MODULE)(layer, masks)
            return args, {
                **kwargs,
                "attention_mask": mask + bias[None].to(mask.dtype),
            }

        return add

    for layer, group in enumerate(groups):
        group.register_forward_pre_hook(hook(layer), with_kwargs=True)


@contextmanager
def assigned(model: Any, rows: Any) -> Iterator[None]:
    """Give the next batches these rows' places; no-op without the bias.

    The rows must stay assigned through backward, which recomputes a
    checkpointed block's bias from them (`attach`).
    """
    import torch

    base = _base(model)
    if base is None or rows is None:
        yield
        return
    state = getattr(base, _STATE)
    before = state["rows"]
    state["rows"] = torch.as_tensor(np.asarray(rows), dtype=torch.long)
    try:
        yield
    finally:
        # Nested uses (a step around its forward) get back the outer rows.
        state["rows"] = before


def adapter_has_bias(adapter: Path | str) -> bool:
    """Whether an adapter directory holds the series bias."""
    config = json.loads((Path(adapter) / "adapter_config.json").read_text())
    return BIAS_MODULE in (config.get("modules_to_save") or [])


def load_adapter(model: Any, adapter: Path | str) -> Any:
    """Return Chronos-2 `model` with an adapter merged, its bias included.

    The bias is attached first when the adapter holds it, so PEFT loads
    its trained values; an adapter without it loads as before. Every
    weight in the adapter's file must reach the model (`_check_loaded`).
    """
    import warnings

    from peft import PeftModel

    if adapter_has_bias(adapter):
        attach(model)
    with warnings.catch_warnings():
        # PEFT only warns when an adapter weight has no file entry.
        warnings.filterwarnings("error", message="Found missing adapter keys")
        peft_model = PeftModel.from_pretrained(model, str(adapter))
    _check_loaded(peft_model, adapter)
    return peft_model.merge_and_unload()


def _check_loaded(peft_model: Any, adapter: Path | str) -> None:
    """Raise unless every weight in an adapter's file reached the model.

    PEFT drops a file entry the model has no parameter for without a
    word, as it would the bias for a model built without it.

    Raises
    ------
    ValueError
        If an adapter weight is not in the model or differs from the file.
    """
    import torch
    from safetensors.torch import load_file  # type: ignore[import-not-found]

    def plain(name: str) -> str:
        name = name.replace(".modules_to_save.default", "")
        return name.replace(".default", "")

    loaded = {
        plain(n): p
        for n, p in [
            *peft_model.named_parameters(),
            *peft_model.named_buffers(),
        ]
    }
    for key, value in load_file(
        str(Path(adapter) / "adapter_model.safetensors")
    ).items():
        have = loaded.get(key)
        if have is None:
            raise ValueError(f"{adapter}: adapter weight {key} has no place")
        if not torch.equal(have.detach().cpu().to(value.dtype), value):
            raise ValueError(f"{adapter}: adapter weight {key} did not load")
