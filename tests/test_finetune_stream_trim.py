"""Leading context patches with no observation are cut from a batch."""

import inspect
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("chronos")

sys.path.insert(0, str(Path(__file__).parent / "forecasters"))
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import finetune_stream as fs  # noqa: E402
from chronos_toy import batch_of, tiny_model, toy_block  # noqa: E402
from test_series_bias import toy_rows  # noqa: E402

from nowcastfoundation import series_bias  # noqa: E402

PAD = 75


def padded(batch: dict) -> dict:
    """The batch with `PAD` all-missing context steps on the left."""
    ctx = batch["context"]
    nan = torch.full((ctx.shape[0], PAD), float("nan"), dtype=ctx.dtype)
    return {**batch, "context": torch.cat([nan, ctx], dim=1)}


def test_only_whole_leading_all_missing_patches_are_cut():
    batch = {**batch_of(toy_block()), "series_rows": toy_rows()}
    ctx = batch["context"]
    ctx[:, :20] = float("nan")
    ctx[1, 18] = 1.0
    trimmed = fs.trim_context(padded(batch))
    kept = trimmed["context"]
    # Whole patches, from the end, back to the first observed step.
    assert kept.shape[1] % fs.INPUT_PATCH == 0
    assert kept.shape[1] >= ctx.shape[1] - 18
    assert kept.shape[1] < ctx.shape[1] - 18 + fs.INPUT_PATCH
    assert torch.equal(
        kept.nan_to_num(-1),
        padded(batch)["context"][:, -kept.shape[1] :].nan_to_num(-1),
    )
    # Nothing else in the batch changes.
    assert trimmed["series_rows"] is batch["series_rows"]
    assert trimmed["future_covariates"] is batch["future_covariates"]


def _loss_and_grads(batch: dict):
    from chronos.chronos2.model import Chronos2Model

    model = tiny_model()
    series_bias.attach(model)
    model.train()
    module = getattr(model, series_bias.BIAS_MODULE)
    with torch.no_grad():
        for i, p in enumerate(module.parameters()):
            p.fill_(0.1 * (i + 1))
    accepted = inspect.signature(Chronos2Model.forward).parameters
    loss = fs.train_step(model, batch, accepted, "cpu", 1.0)
    grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    return loss, grads


def test_a_trimmed_batch_trains_as_the_padded_one():
    batch = padded({**batch_of(toy_block()), "series_rows": toy_rows()})
    loss, grads = _loss_and_grads(batch)
    cut_loss, cut_grads = _loss_and_grads(fs.trim_context(batch))
    assert (
        fs.trim_context(batch)["context"].shape[1] < batch["context"].shape[1]
    )
    assert cut_loss == pytest.approx(loss, rel=1e-6)
    assert len(grads) == len(cut_grads)
    for a, b in zip(grads, cut_grads, strict=True):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-7)


def test_the_trainer_s_batches_are_trimmed():
    import finetune_chronos as ft
    import numpy as np
    from chronos.chronos2.dataset import DatasetMode

    block = toy_block()
    ctx = block["context"]
    pad = np.full((ctx.shape[0], ft.CONTEXT - (ctx.shape[1] - 16)), np.nan)
    block = {
        **block,
        "context": np.concatenate([pad, ctx], 1).astype(np.float32),
    }
    data = fs.ByLength([block], 256, DatasetMode.VALIDATION)
    batch = data._build_batch([0])
    assert batch["context"].shape[1] == 48
