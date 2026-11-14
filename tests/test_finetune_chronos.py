"""Places of the fine-tune's training blocks."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from finetune_chronos import choose_places, plan_batches, size_class

STRATA = [f"P{p}-A{a}" for p in range(1, 5) for a in range(1, 4)]


def test_places_by_strata_give_one_two_and_three_levels():
    rng = np.random.default_rng(0)
    levels = set()
    for _ in range(200):
        places = choose_places(STRATA, int(rng.integers(1, 20)), rng)
        totals = [p for p in places if "-" not in p]
        strata = [p for p in places if "-" in p]
        # A stratum's place total comes with it, or no totals at all.
        assert not totals or {s.split("-")[0] for s in strata} <= set(totals)
        levels.add((bool(totals), bool(strata)))
    assert levels == {(True, True), (True, False), (False, True)}


def test_places_stay_within_the_budget():
    rng = np.random.default_rng(1)
    for budget in range(0, 20):
        assert len(choose_places(STRATA, budget, rng)) <= max(budget, 0)


def test_batches_use_every_block_once_and_pack_by_size():
    rng = np.random.default_rng(2)
    sizes = np.exp(rng.uniform(0, np.log(2000), 500)).astype(int) + 1
    plan = plan_batches(sizes, 256, rng)
    used = sorted(i for batch in plan for i in batch)
    assert used == list(range(len(sizes)))
    for batch in plan:
        # One size class per batch, and only a block of 256 series or more
        # makes a batch past twice the batch size.
        assert len({size_class(int(sizes[i])) for i in batch}) == 1
        total = sizes[batch].sum()
        assert total < 2 * 256 or len(batch) == 1
