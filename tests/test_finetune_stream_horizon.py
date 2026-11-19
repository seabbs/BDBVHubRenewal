"""The per-horizon loss: nowcast-only, nowcast plus a week, distillation."""

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("chronos")
pytest.importorskip("peft")

sys.path.insert(0, str(Path(__file__).parent / "forecasters"))
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import finetune_stream as fs  # noqa: E402
from chronos_toy import batch_of, tiny_model, toy_block  # noqa: E402

NOWCAST = 2
"""The toy block's nowcast steps: six lag series over three places."""


def loss_of(model, batch, mode: str) -> float:
    """Return the training loss of `batch` under horizon loss `mode`."""
    from chronos.chronos2.model import Chronos2Model

    accepted = inspect.signature(Chronos2Model.forward).parameters
    rows = len(batch["context"])
    batch = {**batch, "horizon_rows": np.tile([NOWCAST, 7], (rows, 1))}
    with (
        fs.horizon_loss(mode),
        fs._HorizonLoss(batch["horizon_rows"]) as loss,
    ):
        if mode == "distil":
            loss.capture_base(model, batch, accepted)
        return float(fs.forward(model, batch, accepted).loss)


def changed_after(step: int) -> dict:
    """The toy batch with every target from `step` on scaled up."""
    batch = batch_of(toy_block())
    target = batch["future_target"].clone()
    target[:, step:] *= 3.0
    return {**batch, "future_target": target}


def test_all_is_chronos_own_loss():
    from chronos.chronos2.model import Chronos2Model

    model = tiny_model()
    batch = batch_of(toy_block())
    accepted = inspect.signature(Chronos2Model.forward).parameters
    plain = float(fs.forward(model, batch, accepted).loss)
    assert loss_of(model, batch, "all") == pytest.approx(plain, rel=1e-6)


def test_nowcast_ignores_the_targets_after_the_cutoff():
    model = tiny_model()
    base = loss_of(model, batch_of(toy_block()), "nowcast")
    assert loss_of(model, changed_after(NOWCAST), "nowcast") == pytest.approx(
        base, rel=1e-6
    )
    assert loss_of(model, changed_after(NOWCAST - 1), "nowcast") != (
        pytest.approx(base, rel=1e-6)
    )


def test_nowcast_week_ignores_targets_after_a_week():
    model = tiny_model()
    base = loss_of(model, batch_of(toy_block()), "nowcast+1w")
    after = changed_after(NOWCAST + 7)
    assert loss_of(model, after, "nowcast+1w") == pytest.approx(base, rel=1e-6)
    inside = changed_after(NOWCAST + 6)
    assert loss_of(model, inside, "nowcast+1w") != pytest.approx(base, rel=1e-6)


def test_nowcast_loss_is_rescaled_to_the_steps_it_keeps():
    model = tiny_model()
    batch = batch_of(toy_block())
    every = loss_of(model, batch, "all")
    kept = loss_of(model, batch, "nowcast")
    # Kept steps are 2 of 16; without rescaling the loss would be about
    # an eighth of the full one.
    assert kept > every / 4


def peft_model():
    from peft import get_peft_model

    from bdbvhubrenewal.forecasters import series_bias

    model = tiny_model()
    series_bias.attach(model)
    return get_peft_model(model, fs.lora_config(True))


def test_distil_adds_nothing_while_the_adapter_is_at_its_start():
    model = peft_model()
    batch = batch_of(toy_block())
    assert loss_of(model, batch, "distil") == pytest.approx(
        loss_of(model, batch, "all"), rel=1e-6
    )


def test_distil_penalises_forecasts_that_move_from_base():
    model = peft_model()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(0, 0.5)
    batch = batch_of(toy_block())
    assert loss_of(model, batch, "distil") > loss_of(model, batch, "all")


def test_by_length_gives_rows_their_nowcast_and_week_steps():
    from chronos.chronos2.dataset import DatasetMode

    block = {**toy_block(), "source": "3day", "places": ("A", "B", "C")}
    with fs.horizon_loss("nowcast"):
        data = fs.ByLength([block], 256, DatasetMode.VALIDATION)
        batch = data._build_batch([0])
    rows = batch["horizon_rows"]
    assert rows.shape == (len(block["context"]), 2)
    assert (rows[:, 0] == NOWCAST).all()
    assert (rows[:, 1] == 2).all()


def test_all_leaves_batches_and_settings_as_before():
    from chronos.chronos2.dataset import DatasetMode

    data = fs.ByLength([toy_block()], 256, DatasetMode.VALIDATION)
    assert "horizon_rows" not in data._build_batch([0])
    assert fs.horizon_settings("all", 1.0) == {}
    assert fs.horizon_settings("nowcast", 1.0) == {
        "horizon_loss": {"mode": "nowcast"}
    }
    assert fs.horizon_settings("distil", 0.5) == {
        "horizon_loss": {"mode": "distil", "distil_weight": 0.5}
    }
