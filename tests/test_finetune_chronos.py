"""Places of the fine-tune's training blocks."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from finetune_chronos import (
    choose_places,
    plan_batches,
    size_class,
    target_levels,
)

STRATA = [f"P{p}-A{a}" for p in range(1, 5) for a in range(1, 4)]


def test_level_mix_is_about_a_quarter_each_for_nested_strata():
    rng = np.random.default_rng(0)
    mixes = []
    for _ in range(2000):
        places, mix = choose_places(STRATA, int(rng.integers(4, 20)), rng)
        levels = set(target_levels(places))
        expected = {
            "all_levels": {"national", "intermediate", "leaf"},
            "leaves_national": {"national", "leaf"},
            "leaves": {"leaf"},
            "national": {"national"},
        }[mix]
        assert levels == expected
        mixes.append(mix)
    shares = {m: mixes.count(m) / len(mixes) for m in set(mixes)}
    assert all(abs(share - 0.25) < 0.04 for share in shares.values())
    assert len(shares) == 4


def test_flat_strata_have_no_intermediate_level():
    rng = np.random.default_rng(1)
    flat = [f"P{i:02d}" for i in range(1, 9)]
    mixes = {choose_places(flat, 10, rng)[1] for _ in range(200)}
    assert mixes == {"leaves_national", "leaves", "national"}


def test_places_stay_within_the_budget():
    rng = np.random.default_rng(1)
    for budget in range(1, 20):
        places, _ = choose_places(STRATA, budget, rng)
        assert len(places) <= budget + 1


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


def test_no_batch_holds_more_series_than_the_budget_allows():
    rng = np.random.default_rng(3)
    sizes = rng.integers(100, 200, 50)
    for batch in plan_batches(sizes, 256, rng, most=300):
        assert sizes[batch].sum() <= 300 or len(batch) == 1


def test_flat_places_can_nest_in_regions_with_levels_recorded():
    import pandas as pd
    from finetune_chronos import regions, target_levels

    rng = np.random.default_rng(4)
    places = [f"P{i:02d}" for i in range(1, 9)]
    mapping = regions(places, rng)
    assert set(mapping) == set(places)
    parents = {m.split("-")[0] for m in mapping.values()}
    assert 2 <= len(parents) <= 6
    assert target_levels(("CD", "R1", "R1-P01", "A1")) == (
        "national",
        "intermediate",
        "leaf",
        "intermediate",
    )
    assert target_levels(("CD", "P01", "P02")) == ("national", "leaf", "leaf")
    del pd


def test_ages_are_not_grouped_into_regions():
    import pandas as pd
    from finetune_chronos import outbreak_blocks

    days = pd.date_range("2026-01-01", periods=140)
    rows = [
        (f"A{a}", str(d.date()), str((d + pd.Timedelta(days=2)).date()), 5)
        for a in range(1, 6)
        for d in days
    ]
    counts = pd.DataFrame(
        rows, columns=["location", "reference_date", "report_date", "count"]
    )
    blocks = outbreak_blocks(
        counts, [f"A{a}" for a in range(1, 6)], np.random.default_rng(0), 1, 4
    )
    assert blocks
    assert all("intermediate" not in b["levels"] for b in blocks)


def test_an_empty_batch_is_done_with_no_simulations(tmp_path):
    import pandas as pd
    from finetune_stream import ready_outbreaks

    (tmp_path / "settings-5001.csv").write_text("")
    (tmp_path / "counts-5001.csv.gz").write_text("")
    pd_rows = "outbreak\n5002-0001\n"
    (tmp_path / "settings-5002.csv").write_text(pd_rows)
    pd.DataFrame({"outbreak": ["5002-0001-1"]}).to_csv(
        tmp_path / "counts-5002.csv.gz", index=False
    )
    assert ready_outbreaks(tmp_path, range(5001, 5003)) == ["5002-0001"]


def test_stop_rule_needs_two_checkpoints():
    from finetune_stream import should_stop

    assert not should_stop([0.2], 8000, 8000, 0.01)
    assert should_stop([0.2, 0.199], 16000, 8000, 0.01)
    assert not should_stop([0.2, 0.15], 16000, 8000, 0.01)
    assert not should_stop([0.2, 0.199], 4000, 8000, 0.01)
