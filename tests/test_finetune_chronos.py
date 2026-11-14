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


def test_a_mixed_batch_is_moved_aside_to_run_again(tmp_path):
    import pandas as pd
    from finetune_stream import seed_done

    pd.DataFrame({"outbreak": ["5001-0001", "5001-0002"]}).to_csv(
        tmp_path / "settings-5001.csv", index=False
    )
    pd.DataFrame({"outbreak": ["5001-0001-1", "5001-0002-3"]}).to_csv(
        tmp_path / "counts-5001.csv.gz", index=False
    )
    # Draws of the same simulations: whole.
    assert seed_done(tmp_path, 5001)
    pd.DataFrame({"outbreak": ["5001-0001-1", "5001-0009-1"]}).to_csv(
        tmp_path / "counts-5001.csv.gz", index=False
    )
    # Another run's counts: moved aside, not done.
    assert not seed_done(tmp_path, 5001)
    assert not (tmp_path / "settings-5001.csv").exists()
    assert list((tmp_path / "bad").iterdir())


def test_a_non_finite_loss_or_gradient_leaves_the_weights():
    import torch
    from finetune_stream import finite_step

    layer = torch.nn.Linear(2, 1)
    params = list(layer.parameters())
    optimiser = torch.optim.AdamW(params, lr=0.1)
    before = [p.detach().clone() for p in params]
    layer(torch.ones(1, 2)).sum().backward()
    for p in params:
        p.grad[...] = float("nan")
    assert not finite_step(params, optimiser, 1.0)
    # A skipped block leaves no gradients and a NaN loss.
    assert not finite_step(params, optimiser, float("nan"))
    assert all(torch.equal(a, b) for a, b in zip(before, params, strict=True))
    layer(torch.ones(1, 2)).sum().backward()
    assert finite_step(params, optimiser, 0.5)
    assert not torch.equal(before[0], params[0])


def test_memory_budget_sets_checkpointing_and_block_size():
    from finetune_chronos import checkpoint_above, max_series, memory_gb

    # Measured: 800 series without checkpointing about 43 GB; 1,000, 1,500
    # and 2,000 with it about 9.6, 17.3 and 25.9 GB.
    assert abs(memory_gb(800, checkpointed=False) - 43) < 2
    for series, gb in ((1000, 9.6), (1500, 17.3), (2000, 25.9)):
        assert abs(memory_gb(series, checkpointed=True) - gb) < 1
    budget = 36
    above = checkpoint_above(budget)
    assert memory_gb(above, False) <= budget < memory_gb(above + 1, False)
    most = max_series(budget)
    assert memory_gb(most, True) <= budget < memory_gb(most + 1, True)


def test_skipped_blocks_are_listed_for_the_release():
    from finetune_stream import skipped_lines

    skipped = [{"outbreak": "5043-0013", "step": 3700, "series": 897}]
    assert skipped_lines(skipped) == [
        "# skipped 5043-0013 at step 3700, 897 series"
    ]
