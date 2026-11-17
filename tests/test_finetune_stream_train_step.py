"""A training step keeps the bias's rows until backward is done."""

import inspect
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("chronos")
pytest.importorskip("peft")

sys.path.insert(0, str(Path(__file__).parent / "forecasters"))
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import finetune_stream as fs  # noqa: E402
from chronos_toy import batch_of, tiny_model, toy_block  # noqa: E402
from test_series_bias import toy_rows  # noqa: E402

from bdbvhubrenewal.forecasters import series_bias  # noqa: E402


def step_grads(monkeypatch, checkpointed: bool):
    from chronos.chronos2.model import Chronos2EncoderBlock, Chronos2Model

    model = tiny_model()
    series_bias.attach(model)
    model.train()
    module = getattr(model, series_bias.BIAS_MODULE)
    with torch.no_grad():
        for i, p in enumerate(module.parameters()):
            p.fill_(0.1 * (i + 1))
    if checkpointed:
        monkeypatch.setattr(
            Chronos2EncoderBlock, "forward", Chronos2EncoderBlock.forward
        )
        fs.checkpoint_large_batches(0)
    batch = {**batch_of(toy_block()), "series_rows": toy_rows()}
    accepted = inspect.signature(Chronos2Model.forward).parameters
    fs.train_step(model, batch, accepted, "cpu", 1.0)
    return [p.grad.clone() for p in module.parameters()]


def test_a_checkpointed_step_trains_the_bias_as_a_plain_one(monkeypatch):
    plain = step_grads(monkeypatch, False)
    checked = step_grads(monkeypatch, True)
    for a, b in zip(plain, checked, strict=True):
        assert torch.allclose(a, b, atol=1e-6)


def test_assigned_restores_the_rows_it_replaced():
    model = tiny_model()
    series_bias.attach(model)
    state = getattr(model, series_bias._STATE)
    with series_bias.assigned(model, toy_rows()):
        outer = state["rows"]
        with series_bias.assigned(model, toy_rows()):
            pass
        assert state["rows"] is outer
    assert state["rows"] is None


def test_a_bf16_step_runs_on_cpu_with_a_finite_fp32_loss_and_fp32_grads():
    from chronos.chronos2.model import Chronos2Model

    model = tiny_model()
    series_bias.attach(model)
    model.train()
    batch = {**batch_of(toy_block()), "series_rows": toy_rows()}
    accepted = inspect.signature(Chronos2Model.forward).parameters
    seen = {}
    original = Chronos2Model._compute_loss

    def spy(self, **kwargs):
        loss = original(self, **kwargs)
        seen["dtype"] = loss.dtype
        seen["autocast"] = torch.is_autocast_enabled("cpu")
        return loss

    Chronos2Model._compute_loss = spy
    try:
        value = fs.train_step(model, batch, accepted, "cpu", 1.0, bf16=True)
    finally:
        Chronos2Model._compute_loss = original
    assert torch.isfinite(torch.tensor(value))
    # The loss is reduced in fp32, outside autocast.
    assert seen == {"dtype": torch.float32, "autocast": False}
    params = [p for p in model.parameters() if p.grad is not None]
    assert params
    assert all(p.dtype == torch.float32 for p in params)
    assert all(p.grad.dtype == torch.float32 for p in params)


def test_bf16_is_on_by_default_for_the_restart():
    args = fs.build_parser().parse_args(
        ["sims", "out", "--held-out-seeds", "1", "2", "--grid", "r5"]
    )
    assert args.bf16 is True
    off = fs.build_parser().parse_args(
        ["s", "o", "--held-out-seeds", "1", "2", "--grid", "r5", "--no-bf16"]
    )
    assert off.bf16 is False
