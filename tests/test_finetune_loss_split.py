"""Held-out loss split into nowcast and forecast steps."""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("chronos")

sys.path.insert(0, str(Path(__file__).parent / "forecasters"))
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import finetune_stream as fs  # noqa: E402
from chronos_toy import LAG, N_PLACES, T, tiny_model, toy_block  # noqa: E402


@pytest.fixture
def short_context(monkeypatch):
    # The toy blocks have 48 steps of context, not the trainer's 288.
    monkeypatch.setattr(fs.ft, "CONTEXT", T)


def blocks(n: int = 4) -> list[dict]:
    """Toy held-out blocks, each place with `LAG` lag series."""
    return [
        {**toy_block(i), "places": tuple(f"P{p}" for p in range(N_PLACES))}
        for i in range(n)
    ]


def only_nowcast(block: dict) -> dict:
    """The block with its targets after the cut-off missing."""
    x = block["context"].copy()
    x[: block["n_targets"], T + fs.nowcast_steps(block) :] = np.nan
    return {**block, "context": x}


def test_nowcast_steps_are_the_release_lags_of_each_place():
    assert fs.nowcast_steps(blocks(1)[0]) == LAG
    # A block of period totals has two target rows a place.
    block = {**blocks(1)[0], "n_targets": 2 * N_PLACES}
    assert fs.nowcast_steps(block) == LAG


@pytest.mark.usefixtures("short_context")
def test_the_parts_sum_to_the_held_out_loss_which_is_unchanged():
    model = tiny_model()
    val = blocks()
    parts = fs.held_out_parts(model, val, batch_size=7)
    assert parts["loss"] == fs.held_out_loss(model, val, batch_size=7)
    assert parts["nowcast"] + parts["forecast"] == pytest.approx(
        parts["loss"], rel=1e-6
    )
    assert parts["nowcast"] > 0
    assert parts["forecast"] > 0


@pytest.mark.usefixtures("short_context")
def test_the_nowcast_part_is_the_loss_on_the_steps_to_the_cut_off():
    model = tiny_model()
    val = blocks()
    parts = fs.held_out_parts(model, val, batch_size=100)
    masked = fs.held_out_loss(model, [only_nowcast(b) for b in val], 100)
    assert parts["nowcast"] == pytest.approx(masked, rel=1e-5)
