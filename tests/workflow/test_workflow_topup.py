"""Topping up the training rounds' missing grid cells (flowlib.topup)."""

import pandas as pd
import pytest
from flowlib import segments, topup


def _round(root, r, cells, excluded=(), block=None, seed=None):
    """A round's settings for `cells` (indices) and its QC exclusions."""
    d = root / f"r{r}"
    (d / "sims").mkdir(parents=True)
    seed = seed or 30000 + 100 * r
    rows = [
        {
            "outbreak": f"{seed}-{k + 1:04d}",
            "cell_block": r if block is None else block,
            "cell_index": c,
        }
        for k, c in enumerate(cells)
    ]
    pd.DataFrame(rows).to_csv(d / "sims" / f"settings-{seed}.csv", index=False)
    (d / "qc").mkdir()
    pd.DataFrame(
        {"simulation": list(excluded), "reason": "r", "evidence": "e"}
    ).to_csv(d / "qc" / "excluded.csv", index=False)
    return d


def test_missing_cells_are_excluded_and_failed_ones(tmp_path):
    # Round 1 holds 838 of its 840 cells (two failed all their tries), and
    # the QC excluded one of them.
    cells = [c for c in range(840) if c not in (5, 6)]
    _round(tmp_path, 1, cells, excluded=["30100-0001"])
    got = topup.missing_cells(tmp_path, [1], topup_round=9, block=840)
    assert got == [840 + 0, 840 + 5, 840 + 6]


def test_kept_top_ups_fill_their_cells(tmp_path):
    _round(tmp_path, 1, [c for c in range(840) if c not in (5, 6)])
    # The top-up round simulated cells 5 and 6 of block 1; the QC excluded
    # the second, so it is still missing.
    _round(tmp_path, 9, [5, 6], excluded=["30900-0002"], block=1, seed=30900)
    assert topup.missing_cells(tmp_path, [1], topup_round=9, block=840) == [846]


def test_a_try_s_batches_hold_at_most_a_batch_of_cells(tmp_path):
    seeds = segments.round_seeds(32, 30000, segments.ROUND_SEEDS)
    batches = topup.plan(list(range(45)), seeds, attempt=1, batch=21)
    assert [s for s, _ in batches] == [33220, 33221, 33222]
    assert [len(c) for _, c in batches] == [21, 21, 3]
    path = tmp_path / "plan-1.txt"
    topup.write_plan(batches, path)
    assert topup.read_plan(path) == batches
    assert topup.cells_of([path], 33222) == [42, 43, 44]
    assert topup.cells_of([path], 33299) == []
    assert topup.plan([], seeds, 0, 21) == []


def test_a_try_has_a_bounded_number_of_seeds():
    seeds = segments.round_seeds(32, 30000, segments.ROUND_SEEDS)
    with pytest.raises(ValueError, match="more than the 20"):
        topup.plan(list(range(21 * 21)), seeds, 0, 21)
    with pytest.raises(ValueError, match="past the top-up round"):
        topup.plan([1], seeds, 5, 21)


def test_exclusions_are_read_as_the_trainer_reads_them(tmp_path):
    # One simulation a line, as segments.excluded reads it too.
    d = _round(tmp_path, 1, [0, 1])
    (d / "qc" / "excluded.csv").write_text("30100-0002\n")
    assert topup.kept_cells(d, 840) == {840}
