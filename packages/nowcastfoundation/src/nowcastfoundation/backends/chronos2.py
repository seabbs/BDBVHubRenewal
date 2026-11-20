# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""The Chronos-2 backend: device, precision and batched quantiles.

`_predict` gives groups of series (`nowcastfoundation.backends.Series`)
to a Chronos-2 pipeline in forward passes of up to `chronos_batch()`
rows, with the series bias's places and parents where the model has it.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Sequence
from typing import Any, Final

import numpy as np

from nowcastfoundation import series_bias
from nowcastfoundation.backends import Series

__all__ = [
    "CHRONOS_KEYS",
    "CHRONOS_MODEL",
    "CHRONOS_REVISION",
    "PRECISIONS",
    "apply_threads",
    "chronos_batch",
    "inference_precision",
    "torch_device",
]

CHRONOS_MODEL: Final = "amazon/chronos-2"
"""The Hugging Face checkpoint."""

CHRONOS_REVISION: Final = "29ec3766d36d6f73f0696f85560a422f50e8498c"
"""The checkpoint's commit, pinned so runs are reproducible."""


def torch_device(device: str | None = None) -> str:
    """Return the torch device Chronos-2 runs on, to train or predict.

    `device`, by default `DEVICE`: the workflow resolves the machine's
    device (workflow/flowlib/machine.py) and passes it on. Unset or
    `auto`, run outside the workflow, the first of CUDA, MPS and the CPU
    torch has.
    """
    device = device or os.environ.get("DEVICE") or "auto"
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


PRECISIONS: Final = ("fp32", "fp16", "bf16", "auto")
"""Inference precisions (`PRECISION`). Training always runs in fp32."""


def inference_precision(device: str, precision: str | None = None) -> str:
    """Return the precision Chronos-2 forecasts in on `device`.

    `precision`, by default `PRECISION`, fp32 if unset: stored fits and
    submissions set none, so they stay fp32; the workflow gives scoring
    units `auto` (`flowlib.machine.torch_env`). `auto` takes each device's
    half precision: fp16 on CUDA GPUs below compute capability 8 (fp16
    tensor cores, no bf16), bf16 on compute capability 8 and above and on
    Apple GPUs, fp32 on the CPU (bf16 ran about 5 times slower on a CPU
    without native bf16). fp16 moved relative WIS by under 0.06% on a GPU
    without bf16.

    Raises
    ------
    ValueError
        If the precision is not one of `PRECISIONS`.
    """
    precision = precision or os.environ.get("PRECISION") or "fp32"
    if precision not in PRECISIONS:
        raise ValueError(f"precision {precision!r} is not one of {PRECISIONS}")
    if precision != "auto":
        return precision
    kind = device.split(":")[0]
    if kind == "cuda":
        import torch

        major, _ = torch.cuda.get_device_capability(device)
        return "bf16" if major >= 8 else "fp16"
    return "bf16" if kind == "mps" else "fp32"


def _autocast(device: str) -> Any:
    """Return the context Chronos-2 forecasts in.

    Autocast at half precision (`inference_precision`), or none at fp32.
    """
    precision = inference_precision(device)
    if precision == "fp32":
        return contextlib.nullcontext()
    import torch

    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.split(":")[0], dtype=dtype)


def apply_threads() -> None:
    """Set torch's CPU threads to `TORCH_THREADS`, if set (the rule's)."""
    n = os.environ.get("TORCH_THREADS", "").strip()
    if n:
        import torch

        torch.set_num_threads(int(n))


def chronos_batch() -> int:
    """Return the series in one Chronos-2 forward (`CHRONOS_BATCH`).

    256 by default, Chronos-2's own, which fits an 8 GB GPU.
    """
    return int(os.environ.get("CHRONOS_BATCH") or 256)


CHRONOS_KEYS: Final = ("target", "past_covariates", "future_covariates")
"""The parts of a group Chronos-2 reads; the rest name its rows' places."""


def _chronos_input(group: Series) -> dict[str, Any]:
    """Return the parts of a group Chronos-2 reads (`CHRONOS_KEYS`)."""
    return {k: group[k] for k in CHRONOS_KEYS if k in group}


def _group_rows(group: Series) -> np.ndarray:
    """Return each of a group's rows' place and parent, `[2, rows]`.

    In Chronos-2's row order: the targets, then the covariates by sorted
    name, past-only ones before known-future ones. Places are numbered in
    the group's `places` order; a target row's place is its
    `target_places` entry, one row per place by default.

    Raises
    ------
    ValueError
        If a covariate names no place, or the target rows do not match
        their places.
    """
    past = list(group.get("past_covariates", {}))
    known = set(group.get("future_covariates", {}))
    n_targets = np.atleast_2d(group["target"]).shape[0]
    named = dict.fromkeys(n.rsplit("_lag_", 1)[0] for n in past)
    places = list(group.get("places") or named)
    target_places = list(group.get("target_places", range(len(places))))
    if len(target_places) != n_targets:
        raise ValueError(
            f"{len(target_places)} target places for {n_targets} targets"
        )
    rows = sorted(n for n in past if n not in known)
    rows += sorted(n for n in past if n in known)
    at = {p: i for i, p in enumerate(places)}
    try:
        owner = [at[n.rsplit("_lag_", 1)[0]] for n in rows]
    except KeyError as err:
        raise ValueError(f"covariate of no target place: {err}") from err
    place = np.concatenate([target_places, owner]).astype(int)
    parent = np.asarray(series_bias.parents_of(places), dtype=int)
    return np.stack([place, parent[place]])


def _schema(group: Series) -> tuple[Any, ...]:
    """Return what groups predicted in one pass must share.

    Chronos-2 takes one schema a call: the same number of targets and the
    same covariate names.
    """
    return (
        np.atleast_2d(group["target"]).shape[0],
        tuple(sorted(group.get("past_covariates", {}))),
        tuple(sorted(group.get("future_covariates", {}))),
    )


def _n_rows(group: Series) -> int:
    """Return a group's rows in Chronos-2's batch: targets and covariates."""
    n = np.atleast_2d(group["target"]).shape[0]
    return int(n + len(group.get("past_covariates", {})))


def _passes(groups: Sequence[Series], budget: int) -> list[list[int]]:
    """Return the groups of each forward pass, by index.

    Groups of one schema share passes of at most `budget` rows, in order;
    a group larger than `budget` is a pass of its own.
    """
    by: dict[tuple[Any, ...], list[int]] = {}
    for i, g in enumerate(groups):
        by.setdefault(_schema(g), []).append(i)
    out: list[list[int]] = []
    for idx in by.values():
        chunk: list[int] = []
        rows = 0
        for i in idx:
            n = _n_rows(groups[i])
            if chunk and rows + n > budget:
                out.append(chunk)
                chunk, rows = [], 0
            chunk.append(i)
            rows += n
        out.append(chunk)
    return out


def _predict(
    pipeline: Any, groups: Sequence[Series], horizon: int, levels: np.ndarray
) -> list[Any]:
    """Return Chronos-2's quantiles at `levels` for each group.

    Groups of one schema share forward passes of up to `chronos_batch()`
    rows, each pass exactly one of Chronos-2's batches. With the series
    bias each pass carries its groups' rows' places and parents, kept
    apart (`series_bias.batch_rows`), as in held-out losses.
    """
    model = getattr(pipeline, "model", None)
    bias = model is not None and series_bias.has_bias(model)
    out: list[Any] = [None] * len(groups)
    for chunk in _passes(groups, chronos_batch()):
        part = [groups[i] for i in chunk]
        rows = (
            series_bias.batch_rows([_group_rows(g) for g in part])
            if bias
            else None
        )
        with (
            (
                series_bias.assigned(model, rows)
                if bias
                else contextlib.nullcontext()
            ),
            _autocast(torch_device()),
        ):
            quantiles, _ = pipeline.predict_quantiles(
                [_chronos_input(g) for g in part],
                batch_size=sum(_n_rows(g) for g in part),
                prediction_length=horizon,
                quantile_levels=levels.tolist(),
            )
        for i, q in zip(chunk, quantiles, strict=True):
            out[i] = q
    return out
