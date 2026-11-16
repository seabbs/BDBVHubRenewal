"""A replay of a run stops at a set step (`finetune_stream --stop-step`)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import finetune_stream as fs


def test_stop_only_at_the_set_step():
    assert not fs.stop_at(609, 610)
    assert fs.stop_at(610, 610)
    assert not fs.stop_at(611, 610)
    # Without a stop step the run goes on.
    assert not fs.stop_at(610, None)


def test_a_stop_checkpoint_is_labelled_as_a_replay():
    out = fs.replay_settings(
        stop_step=610, label=125, chunk_steps=1219, chunk_outbreaks=250
    )
    assert out["n_outbreaks"] == 125
    replay = out["replay"]
    assert replay["stop_step"] == 610
    assert replay["chunk_steps"] == 1219
    assert replay["chunk_outbreaks"] == 250
    assert "replay n125" in replay["note"]
    # The blocks come from all of the chunk's outbreaks, not 125 whole ones.
    assert "250" in replay["note"]


def test_the_stop_folder_is_not_a_prefix_folder(tmp_path):
    where = fs.replay_dir(tmp_path, 125)
    assert where == tmp_path / "replay-n125"
    assert not where.name.startswith("n")


def test_stop_step_is_an_argument():
    parser = fs.parser()
    args = parser.parse_args(
        ["sims", "out", "--grid", "r4", "--held-out-seeds", "1", "2"]
    )
    assert args.stop_step is None
    assert args.stop_label is None
    args = parser.parse_args(
        [
            "sims",
            "out",
            "--grid",
            "r4",
            "--held-out-seeds",
            "1",
            "2",
            "--stop-step",
            "610",
            "--stop-label",
            "125",
        ]
    )
    assert (args.stop_step, args.stop_label) == (610, 125)


def test_checkpoint_threshold_can_be_lowered_without_changing_blocks():
    base = ["sims", "out", "--grid", "r4", "--held-out-seeds", "1", "2"]
    assert fs.parser().parse_args(base).checkpoint_above is None
    args = fs.parser().parse_args([*base, "--checkpoint-above", "64"])
    assert args.checkpoint_above == 64
    # The memory budget, which sets the largest block, is unchanged.
    assert args.memory_gb == fs.ft.MEMORY_GB
