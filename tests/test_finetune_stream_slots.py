"""The trainer's per-series slots (`finetune_stream --series-slots`)."""

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
from test_series_slots import (  # noqa: E402
    LAG,
    N_PLACES,
    swapped,
    tiny_model,
    toy_block,
)

from bdbvhubrenewal.forecasters import slots  # noqa: E402


def padded(block: dict) -> dict:
    """A toy block as the trainer pads it: context then horizon."""
    return {**block, "levels": ["national"] * N_PLACES}


@pytest.fixture
def short_context(monkeypatch):
    # The toy blocks have 48 steps of context, not the trainer's 288.
    monkeypatch.setattr(fs.ft, "CONTEXT", 48)


@pytest.mark.usefixtures("short_context")
def test_batches_carry_slots_paired_by_place():
    from chronos.chronos2.dataset import DatasetMode

    blocks = [padded(toy_block(0)), padded(toy_block(1))]
    data = fs.ByLength(
        blocks,
        100,
        DatasetMode.TRAIN,
        n_slots=8,
        slot_rng=np.random.default_rng(0),
    )
    batch = data._build_batch([0, 1])
    ids = np.asarray(batch["slot_ids"])
    per = N_PLACES * (1 + LAG)
    assert len(ids) == len(batch["context"]) == 2 * per
    places = slots.row_places(N_PLACES, N_PLACES * LAG)
    for k in range(2):
        mine = ids[k * per : (k + 1) * per]
        for p in range(N_PLACES):
            assert set(mine[places == p]) == {mine[p]}
    # Training draws a fresh permutation each time a block is batched.
    draws = {
        tuple(np.asarray(data._build_batch([0])["slot_ids"])) for _ in range(10)
    }
    assert len(draws) > 1


@pytest.mark.usefixtures("short_context")
def test_validation_slots_are_fixed():
    from chronos.chronos2.dataset import DatasetMode

    blocks = [padded(toy_block(0))]
    a = fs.ByLength(blocks, 100, DatasetMode.VALIDATION, n_slots=8)
    b = fs.ByLength(blocks, 100, DatasetMode.VALIDATION, n_slots=8)
    assert np.array_equal(
        a._build_batch([0])["slot_ids"], b._build_batch([0])["slot_ids"]
    )


@pytest.mark.usefixtures("short_context")
def test_batches_without_slots_are_unchanged():
    from chronos.chronos2.dataset import DatasetMode

    data = fs.ByLength([padded(toy_block(0))], 100, DatasetMode.VALIDATION)
    assert "slot_ids" not in data._build_batch([0])


@pytest.mark.usefixtures("short_context")
def test_held_out_loss_uses_the_slots():
    from test_series_slots import trained

    block, swap = padded(toy_block()), padded(swapped(toy_block()))
    plain = tiny_model()
    # Zero slots leave the loss exactly as it was.
    zero = tiny_model()
    slots.attach(zero, n_slots=8)
    assert fs.held_out_loss(zero, [block], 100) == fs.held_out_loss(
        plain, [block], 100
    )
    # A model with slots scores each block with its fixed slots.
    model, seen = trained(), []
    real = slots.assigned

    def spy(m, ids):
        seen.append(None if ids is None else np.asarray(ids))
        return real(m, ids)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(fs.slots, "assigned", spy)
        fs.held_out_loss(model, [block, swap], 100)
    want = fs.block_slots([block, swap], 8)
    assert len(seen) == 1
    assert np.array_equal(seen[0], want)


def test_lora_config_saves_the_slots_with_the_adapter():
    assert fs.lora_config(0).modules_to_save is None
    assert list(fs.lora_config(128).modules_to_save) == [slots.SLOT_MODULE]


def test_slot_settings_enter_the_hash_only_when_on():
    assert fs.slot_settings(0) == {}
    on = fs.slot_settings(128, init_std=0.02, init_seed=1, lr=1e-4)
    assert on["series_slots"]["n"] == 128
    assert on["series_slots"]["init"] == "normal"
    assert on["series_slots"]["init_std"] == 0.02
    assert on["series_slots"]["init_seed"] == 1
    assert on["series_slots"]["learning_rate"] == 1e-4


def test_the_slots_train_in_their_own_group_at_their_own_rate():
    from peft import get_peft_model
    from test_series_slots import lora

    model = tiny_model()
    slots.attach(model, 8, init_std=0.02, seed=1)
    model = get_peft_model(model, lora())
    opt = torch.optim.AdamW(fs.param_groups(model, 1e-5, 1e-4))
    assert [g["lr"] for g in opt.param_groups] == [1e-5, 1e-4]
    slot_params = {
        id(p)
        for n, p in model.named_parameters()
        if slots.SLOT_MODULE in n and p.requires_grad
    }
    assert {id(p) for p in opt.param_groups[1]["params"]} == slot_params
    assert not slot_params & {id(p) for p in opt.param_groups[0]["params"]}
    # Both decay linearly to zero over the same steps.
    fs.set_learning_rates(opt, 50, 100)
    assert [g["lr"] for g in opt.param_groups] == [
        fs.learning_rate(50, 1e-5, 100),
        fs.learning_rate(50, 1e-4, 100),
    ]


def test_random_slots_round_trip_through_the_trainer_state(tmp_path):
    from peft import get_peft_model
    from test_series_slots import lora

    model = tiny_model()
    slots.attach(model, 8, init_std=0.02, seed=1)
    model = get_peft_model(model, lora())
    opt = torch.optim.AdamW(fs.param_groups(model, 1e-5, 1e-4))
    fs.save_state(tmp_path / "state", model, opt, {"step": 0})
    fresh = tiny_model()
    slots.attach(fresh, 8)
    fresh = get_peft_model(fresh, lora())
    opt2 = torch.optim.AdamW(fs.param_groups(fresh, 1e-5, 1e-4))
    fs.load_state(tmp_path / "state", fresh, opt2)

    def slot_weight(m):
        return next(
            p
            for n, p in m.named_parameters()
            if slots.SLOT_MODULE in n and p.requires_grad
        )

    assert float(slot_weight(model).abs().sum()) > 0
    assert torch.equal(slot_weight(model), slot_weight(fresh))
    assert [g["lr"] for g in opt2.param_groups] == [1e-5, 1e-4]


def test_trainer_state_round_trips_the_slots(tmp_path):
    from test_series_slots import trained

    model = trained()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])
    fs.save_state(tmp_path / "state", model, opt, {"step": 3})
    fresh = tiny_model()
    slots.attach(fresh, n_slots=8)
    from peft import get_peft_model
    from test_series_slots import lora

    fresh = get_peft_model(fresh, lora())
    opt2 = torch.optim.Adam([p for p in fresh.parameters() if p.requires_grad])
    assert fs.load_state(tmp_path / "state", fresh, opt2) == {"step": 3}

    def weights(m):
        return {
            n: p
            for n, p in m.named_parameters()
            if slots.SLOT_MODULE in n and p.requires_grad
        }

    want, got = weights(model), weights(fresh)
    assert want
    assert want.keys() == got.keys()
    assert all(torch.equal(want[n], got[n]) for n in want)


def test_continuing_without_slots_in_the_adapter_is_refused(tmp_path):
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(r=2, lora_alpha=4, target_modules=["self_attention.q"])
    get_peft_model(tiny_model(), config).save_pretrained(tmp_path / "round0")
    model = tiny_model()
    slots.attach(model, n_slots=8)
    with pytest.raises(ValueError, match="slots"):
        fs.lora_model(model, fs.lora_config(8), tmp_path / "round0")
