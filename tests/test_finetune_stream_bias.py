"""The trainer's series bias (`finetune_stream --series-bias`)."""

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
from chronos_toy import HIER, LAG, N_PLACES, tiny_model, toy_block  # noqa: E402

from bdbvhubrenewal.forecasters import series_bias  # noqa: E402


def padded(block: dict) -> dict:
    """A toy block as the trainer pads it: context then horizon."""
    return {**block, "levels": ["national", "leaf", "leaf"]}


@pytest.fixture
def short_context(monkeypatch):
    # The toy blocks have 48 steps of context, not the trainer's 288.
    monkeypatch.setattr(fs.ft, "CONTEXT", 48)


@pytest.mark.usefixtures("short_context")
def test_batches_carry_each_rows_place_and_parent():
    from chronos.chronos2.dataset import DatasetMode

    blocks = [padded(toy_block(0)), padded(toy_block(1))]
    data = fs.ByLength(blocks, 100, DatasetMode.TRAIN, bias=True)
    batch = data._build_batch([0, 1])
    place, parent = np.asarray(batch["series_rows"])
    per = N_PLACES * (1 + LAG)
    assert len(place) == len(batch["context"]) == 2 * per
    rows = series_bias.row_places(N_PLACES, N_PLACES * LAG)
    assert list(place[:per]) == list(rows)
    assert list(place[per:]) == list(rows + N_PLACES)
    # Levels give the country as every other place's parent.
    assert list(parent[:N_PLACES]) == [-1, 0, 0]
    assert list(parent[per : per + N_PLACES]) == [-1, 3, 3]
    plain = fs.ByLength(blocks, 100, DatasetMode.TRAIN)._build_batch([0, 1])
    assert "series_rows" not in plain


def test_block_parents_come_from_place_names():
    block = {"n_targets": len(HIER), "n_covariates": 0, "places": HIER}
    _, parent = fs.block_rows([block])
    assert list(parent) == series_bias.parents_of(HIER)


@pytest.mark.usefixtures("short_context")
def test_held_out_loss_with_a_zero_bias_is_base_chronos():
    block = padded(toy_block())
    model = tiny_model()
    series_bias.attach(model)
    assert fs.held_out_loss(model, [block], 100) == fs.held_out_loss(
        tiny_model(), [block], 100
    )


def test_lora_config_saves_the_bias_with_the_adapter():
    assert fs.lora_config(False).modules_to_save is None
    assert list(fs.lora_config(True).modules_to_save) == [
        series_bias.BIAS_MODULE
    ]


def test_bias_settings_enter_the_hash_only_when_on():
    assert fs.bias_settings(False) == {}
    on = fs.bias_settings(True, 1e-3)["series_bias"]
    assert on["init"] == "zero"
    assert on["learning_rate"] == 1e-3
    assert "nearest" in on["parent_rule"]


def test_the_bias_trains_in_its_own_group_at_its_own_rate():
    model = tiny_model()
    series_bias.attach(model)
    model = fs.lora_model(model, fs.lora_config(True), None)
    opt = torch.optim.AdamW(fs.param_groups(model, 1e-5, 1e-3))
    assert [g["lr"] for g in opt.param_groups] == [1e-5, 1e-3]
    bias = {
        id(p)
        for n, p in model.named_parameters()
        if series_bias.BIAS_MODULE in n and p.requires_grad
    }
    assert bias
    assert {id(p) for p in opt.param_groups[1]["params"]} == bias
    fs.set_learning_rates(opt, 50, 100)
    assert [g["lr"] for g in opt.param_groups] == [
        fs.learning_rate(50, 1e-5, 100),
        fs.learning_rate(50, 1e-3, 100),
    ]


def test_the_bias_round_trips_through_the_trainer_state(tmp_path):
    model = tiny_model()
    series_bias.attach(model)
    model = fs.lora_model(model, fs.lora_config(True), None)
    for n, p in model.named_parameters():
        if series_bias.BIAS_MODULE in n and p.requires_grad:
            p.data.fill_(0.25)
    opt = torch.optim.AdamW(fs.param_groups(model, 1e-5, 1e-3))
    fs.save_state(tmp_path / "state", model, opt, {"step": 3})
    fresh = tiny_model()
    series_bias.attach(fresh)
    fresh = fs.lora_model(fresh, fs.lora_config(True), None)
    opt2 = torch.optim.AdamW(fs.param_groups(fresh, 1e-5, 1e-3))
    assert fs.load_state(tmp_path / "state", fresh, opt2) == {"step": 3}
    got = [
        p
        for n, p in fresh.named_parameters()
        if series_bias.BIAS_MODULE in n and p.requires_grad
    ]
    assert got
    assert all(bool((p == 0.25).all()) for p in got)


def test_continuing_without_the_bias_in_the_adapter_is_refused(tmp_path):
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(r=2, lora_alpha=4, target_modules=["self_attention.q"])
    get_peft_model(tiny_model(), config).save_pretrained(tmp_path / "round0")
    model = tiny_model()
    series_bias.attach(model)
    with pytest.raises(ValueError, match="bias"):
        fs.lora_model(model, fs.lora_config(True), tmp_path / "round0")


def test_bias_summary_gives_each_pairs_values_by_layer_and_head():
    model = tiny_model()
    assert fs.bias_summary(model) == {}
    series_bias.attach(model)
    model = fs.lora_model(model, fs.lora_config(True), None)
    for n, p in model.named_parameters():
        if n.endswith(".same") and p.requires_grad:
            with torch.no_grad():
                p.copy_(torch.arange(p.numel(), dtype=p.dtype).view_as(p))
    summary = fs.bias_summary(model)
    assert set(summary) == set(series_bias.PAIRS)
    same = summary["same"]
    n = len(same["values"]) * len(same["values"][0])
    assert same["min"] == 0.0
    assert same["max"] == float(n - 1)
    assert same["mean"] == pytest.approx((n - 1) / 2)
    assert summary["up"]["max"] == 0.0


def test_each_group_reports_its_current_rate_by_name():
    model = tiny_model()
    series_bias.attach(model)
    model = fs.lora_model(model, fs.lora_config(True), None)
    opt = torch.optim.AdamW(fs.param_groups(model, 1e-5, 1e-3))
    fs.set_learning_rates(opt, 25, 100)
    assert fs.group_rates(opt) == {
        "lora": fs.learning_rate(25, 1e-5, 100),
        "bias": fs.learning_rate(25, 1e-3, 100),
    }


def test_a_places_period_totals_row_is_in_its_place():
    block = {
        "n_targets": 6,
        "n_covariates": 6,
        "places": ("CD", "CD-NK", "CD-IT"),
        "target_places": [0, 1, 2, 0, 1, 2],
    }
    place, parent = fs.block_rows([block])
    assert list(place) == [0, 1, 2, 0, 1, 2, 0, 0, 1, 1, 2, 2]
    assert list(parent) == [-1, 0, 0, -1, 0, 0, -1, -1, 0, 0, 0, 0]
