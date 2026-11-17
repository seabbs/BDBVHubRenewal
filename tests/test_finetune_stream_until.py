"""A segment of a continuous run stops at a prefix (`--until`)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import finetune_stream as fs


def test_until_is_a_flag():
    args = fs.build_parser().parse_args(
        [
            "sims",
            "out",
            "--grid",
            "r5",
            "--held-out-seeds",
            "1",
            "2",
            "--until",
            "500",
        ]
    )
    assert args.until == 500
    defaults = fs.build_parser().parse_args(
        ["sims", "out", "--grid", "r5", "--held-out-seeds", "1", "2"]
    )
    assert defaults.until is None


def test_until_must_be_a_prefix():
    fs.check_until(500, [250, 500, 1000])
    fs.check_until(None, [250, 500])
    with pytest.raises(ValueError, match="600"):
        fs.check_until(600, [250, 500, 1000])


def test_the_state_is_kept_with_the_checkpoint(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "state.json").write_text('{"step": 3}')
    (state / "adapter.pt").write_bytes(b"a")
    where = tmp_path / "n500"
    where.mkdir()
    fs.keep_state(state, where)
    assert (where / "state" / "state.json").read_text() == '{"step": 3}'
    assert (where / "state" / "adapter.pt").read_bytes() == b"a"
    # Kept again, it is replaced, not merged.
    (state / "adapter.pt").unlink()
    fs.keep_state(state, where)
    assert not (where / "state" / "adapter.pt").exists()


def test_the_random_state_round_trips(tmp_path):
    torch = pytest.importorskip("torch")
    torch.manual_seed(1)
    fs.save_rng(tmp_path)
    first = torch.rand(3)
    torch.manual_seed(99)
    assert fs.load_rng(tmp_path)
    assert torch.equal(torch.rand(3), first)


def test_no_saved_random_state_leaves_it_as_it_is(tmp_path):
    pytest.importorskip("torch")
    assert not fs.load_rng(tmp_path)


def test_a_segment_trains_while_the_state_is_short_of_it():
    assert fs.until_start(750, 1000, have_prefix=False) == "train"
    assert fs.until_start(0, 250, have_prefix=False) == "train"
    assert fs.until_start(500, None, have_prefix=False) == "train"


def test_a_segment_already_saved_at_its_prefix_only_keeps_its_state():
    assert fs.until_start(1000, 1000, have_prefix=True) == "kept"


def test_a_state_past_the_prefix_cannot_make_it():
    with pytest.raises(SystemExit, match="past n1000"):
        fs.until_start(1250, 1000, have_prefix=True)
    with pytest.raises(SystemExit, match="n1000 is missing"):
        fs.until_start(1000, 1000, have_prefix=False)
