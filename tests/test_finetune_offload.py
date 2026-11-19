"""Evaluation offloaded from the trainer (`finetune_stream --eval offload`)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("chronos")
pytest.importorskip("peft")

sys.path.insert(0, str(Path(__file__).parent / "forecasters"))
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import finetune_evaluate as fe  # noqa: E402
import finetune_stream as fs  # noqa: E402
from chronos_toy import tiny_model, toy_block  # noqa: E402

from bdbvhubrenewal.forecasters import series_bias  # noqa: E402


@pytest.fixture
def short_context(monkeypatch):
    # The toy blocks have 48 steps of context, not the trainer's 288.
    monkeypatch.setattr(fs.ft, "CONTEXT", 48)


def blocks(n: int = 4) -> list[dict]:
    """Toy held-out blocks with levels and a source each."""
    return [
        {
            **toy_block(i),
            "levels": ["national", "leaf", "leaf"],
            "source": "daily" if i % 2 else "weekly",
        }
        for i in range(n)
    ]


def trained(bias: bool = True):
    """A tiny Chronos-2 with a LoRA adapter (and bias) moved off zero."""
    model = tiny_model()
    if bias:
        series_bias.attach(model)
    model = fs.lora_model(model, fs.lora_config(bias), None)
    with torch.no_grad():
        for i, p in enumerate(p for p in model.parameters() if p.requires_grad):
            p.add_(0.01 * (i + 1))
    return model


def test_offload_is_refused_while_early_stopping_can_act():
    fs.check_offload("offload", stop_from=8401, outbreaks=8400)
    fs.check_offload("local", stop_from=100, outbreaks=8400)
    with pytest.raises(ValueError, match="stop"):
        fs.check_offload("offload", stop_from=8400, outbreaks=8400)


def test_evaluation_mode_defaults_to_local():
    args = fs.build_parser().parse_args(
        ["sims", "out", "--held-out-seeds", "1", "2", "--grid", "r5"]
    )
    assert args.eval == "local"


def test_a_snapshot_is_written_whole(tmp_path):
    model = trained()
    where = tmp_path / "evals" / "step-0000100"
    fs.save_snapshot(model, where, {"step": 100, "kinds": ["monitor"]})
    assert (where / "adapter" / "adapter_model.safetensors").exists()
    assert series_bias.adapter_has_bias(where / "adapter")
    assert json.loads((where / "request.json").read_text())["step"] == 100
    assert sorted(p.name for p in where.parent.iterdir()) == ["step-0000100"]
    # Saving again replaces it.
    fs.save_snapshot(model, where, {"step": 100, "kinds": ["by_source"]})
    assert json.loads((where / "request.json").read_text())["kinds"] == [
        "by_source"
    ]


@pytest.mark.usefixtures("short_context")
def test_the_evaluator_scores_a_snapshot_as_the_trainer_would(tmp_path):
    model = trained()
    val = blocks()
    where = tmp_path / "step-0000100"
    fs.save_snapshot(model, where, {"step": 100})
    base = tiny_model()
    series_bias.attach(base)
    got = fe.evaluate(
        where / "adapter",
        base,
        val,
        ["monitor", "by_source", "checkpoint"],
        batch_size=100,
        real=val[:2],
    )
    assert got["eval_loss"] == fs.held_out_loss(
        model, val[: fs.ft.EVAL_BLOCKS], 100
    )
    assert got["by_source"] == fs.held_out_by_source(
        model, val, 100, fs.ft.EVAL_BLOCKS
    )
    assert got["held_out_loss"] == fs.held_out_loss(model, val, 100)
    assert got["real_validation_loss"] == fs.held_out_loss(model, val[:2], 100)
    assert got["held_out_loss_by_level"] == fs.held_out_by_level(
        model, val, fs.LEVEL_BLOCKS
    )
    assert got["held_out_loss_by_source"] == fs.held_out_by_source(
        model, val, 100
    )
    assert np.isfinite(got["eval_loss"])
    # Each total comes with its nowcast and forecast parts.
    for name in ("eval_loss", "held_out_loss", "real_validation_loss"):
        assert got[f"{name}_nowcast"] + got[f"{name}_forecast"] == (
            pytest.approx(got[name], rel=1e-6)
        )


def test_held_out_blocks_round_trip_with_their_count(tmp_path):
    val = blocks(3)
    path = tmp_path / "held-out-blocks.pkl"
    fs.save_blocks(val, path)
    got = fe.load_blocks(path, n=3)
    assert len(got) == 3
    assert np.array_equal(got[1]["context"], val[1]["context"], equal_nan=True)
    with pytest.raises(ValueError, match="3"):
        fe.load_blocks(path, n=4)
