"""Places of the fine-tune's training blocks."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from finetune_chronos import choose_places

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
