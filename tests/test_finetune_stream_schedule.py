"""The learning rate falls linearly to zero over the run's simulations."""

import itertools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import finetune_stream as fs


def schedule(chunks: list[tuple[int, int]], outbreaks: int, base: float):
    """The rate of every step of chunks of (simulations, steps)."""
    rates, done = [], 0
    for size, steps in chunks:
        for position in range(steps):
            sims = fs.sims_done(done, size, position, steps)
            rates.append(fs.learning_rate(sims, base, outbreaks))
        done += size
    return rates, fs.learning_rate(done, base, outbreaks)


@pytest.mark.parametrize("per_sim", [4.5625, 6.03])
def test_the_rate_reaches_zero_at_the_last_simulation(per_sim):
    # 33 chunks of 250 and a last of 150 are 8,400 simulations, packed
    # into a varying number of steps a chunk.
    sizes = [250] * 33 + [150]
    chunks = [
        (s, round(s * per_sim) + (i % 3) - 1) for i, s in enumerate(sizes)
    ]
    rates, after = schedule(chunks, 8400, 1e-5)
    assert rates[0] == 1e-5
    assert all(a > b for a, b in itertools.pairwise(rates))
    assert rates[-1] > 0
    assert after == 0.0
    # Half way through the simulations, half the rate, however packed.
    done = 250 * 16 + 200
    sims = fs.sims_done(250 * 16, 250, 4 * chunks[16][1] // 5, chunks[16][1])
    assert sims == pytest.approx(done, abs=1)
    assert fs.learning_rate(4200, 1e-5, 8400) == pytest.approx(5e-6)


def test_sims_done_interpolates_within_a_chunk():
    assert fs.sims_done(500, 250, 0, 1000) == 500
    assert fs.sims_done(500, 250, 500, 1000) == 625
    assert fs.sims_done(500, 250, 1000, 1000) == 750


def test_the_trainer_has_no_step_schedule():
    args = fs.build_parser().parse_args(
        ["s", "o", "--grid", "r5", "--held-out-seeds", "1", "2"]
    )
    assert not hasattr(args, "lr_steps")
