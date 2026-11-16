"""Places of the fine-tune's training blocks."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import finetune_chronos as ft
from finetune_chronos import (
    choose_places,
    plan_batches,
    size_class,
    target_levels,
)

STRATA = [f"P{p}-A{a}" for p in range(1, 5) for a in range(1, 4)]


def test_univariate_blocks_are_about_their_share():
    import finetune_chronos

    rng = np.random.default_rng(0)
    mixes = []
    for _ in range(3000):
        places, mix = choose_places(STRATA, int(rng.integers(4, 20)), rng)
        levels = set(target_levels(places))
        expected = {
            "all_levels": {"national", "intermediate", "leaf"},
            "leaves_national": {"national", "leaf"},
            "leaves": {"leaf"},
            "national": {"national"},
            "single": {"leaf"},
        }[mix]
        assert levels == expected
        mixes.append(mix)
    univariate = sum(m in ("national", "single") for m in mixes) / len(mixes)
    assert finetune_chronos.UNIVARIATE_SHARE == 0.2
    assert abs(univariate - 0.2) < 0.03
    # The multivariate mixes share the rest about evenly.
    for m in ("all_levels", "leaves_national", "leaves"):
        assert abs(mixes.count(m) / len(mixes) - 0.8 / 3) < 0.04


def test_flat_strata_have_no_intermediate_level():
    rng = np.random.default_rng(1)
    flat = [f"P{i:02d}" for i in range(1, 9)]
    mixes = {choose_places(flat, 10, rng)[1] for _ in range(400)}
    assert mixes == {"leaves_national", "leaves", "national", "single"}


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
    import pytest

    torch = pytest.importorskip("torch")
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


def test_a_split_batch_starts_from_clear_gradients():
    from finetune_stream import accumulate

    grads: list[int] = []
    skipped: list[int] = []

    def backward(part, _share):
        if len(part) > 2:
            grads.append(-1)  # a partial backward before running out
            raise RuntimeError("MPS backend out of memory")
        if 9 in part:
            raise RuntimeError("MPS backend out of memory")
        grads.extend(part)
        return 0.5

    def zero():
        grads.clear()

    value = accumulate([1, 2, 3, 4, 9], backward, zero, skipped.append)
    # The partial backward is cleared; every block but the one too large
    # for memory is used once.
    assert sorted(grads) == [1, 2, 3, 4]
    assert skipped == [9]
    assert value == 0.5


def _checkpoint(tmp_path):
    ckpt = tmp_path / "finetuned-ckpt"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_text("w")
    (tmp_path / "settings.json").write_text('{"n_outbreaks": 1000}')
    (tmp_path / "outbreaks.txt").write_text("1\n")
    return tmp_path


class _Gh:
    """Fake gh: `assets` maps the release's asset names to their content."""

    def __init__(self, assets=None):
        self.assets = assets
        self.calls = []

    def __call__(self, command, **_):
        import subprocess

        self.calls.append(command[:3])
        action = command[2]
        if action == "view":
            if self.assets is None:
                return subprocess.CompletedProcess(command, 1, "", "not found")
            names = [{"name": n} for n in self.assets]
            out = __import__("json").dumps({"assets": names})
            return subprocess.CompletedProcess(command, 0, out, "")
        if action == "download":
            sums = self.assets.get("sha256sums.txt", "")
            return subprocess.CompletedProcess(command, 0, sums, "")
        if action in ("create", "upload"):
            self.assets = self.assets or {}
            for f in command:
                if Path(f).is_file():
                    self.assets[Path(f).name] = Path(f).read_text()
            return subprocess.CompletedProcess(command, 0, "ok", "")
        raise AssertionError(command)


def test_publish_is_done_when_the_release_matches(tmp_path):
    from finetune_stream import publish

    where = _checkpoint(tmp_path)
    gh = _Gh()
    publish(where, "abc", run=gh)
    assert [c[2] for c in gh.calls] == ["view", "create"]
    gh.calls.clear()
    publish(where, "abc", run=gh)
    assert "create" not in [c[2] for c in gh.calls]
    assert "upload" not in [c[2] for c in gh.calls]


def test_publish_uploads_only_missing_assets(tmp_path):
    from finetune_stream import publish

    where = _checkpoint(tmp_path)
    gh = _Gh()
    publish(where, "abc", run=gh)
    del gh.assets["outbreaks.txt"]
    gh.calls.clear()
    publish(where, "abc", run=gh)
    assert [c[2] for c in gh.calls if c[2] != "download"] == ["view", "upload"]
    assert "outbreaks.txt" in gh.assets


def test_publish_refuses_a_release_with_other_sums(tmp_path):
    import pytest
    from finetune_stream import publish

    where = _checkpoint(tmp_path)
    gh = _Gh({"sha256sums.txt": "0  adapter_model.safetensors\n"})
    with pytest.raises(RuntimeError, match="differ"):
        publish(where, "abc", run=gh)
    assert "upload" not in [c[2] for c in gh.calls]


def test_a_finished_batch_is_read_once(tmp_path, monkeypatch):
    import finetune_stream
    import pandas as pd

    (tmp_path / "settings-5003.csv").write_text("outbreak\n5003-0001\n")
    pd.DataFrame({"outbreak": ["5003-0001-1"]}).to_csv(
        tmp_path / "counts-5003.csv.gz", index=False
    )
    first = finetune_stream.ready_outbreaks(tmp_path, range(5003, 5004))
    reads = []
    monkeypatch.setattr(
        finetune_stream, "read_ids", lambda p: reads.append(p) or pd.Series()
    )
    again = finetune_stream.ready_outbreaks(tmp_path, range(5003, 5004))
    assert again == first == ["5003-0001"]
    assert reads == []


def test_strata_with_no_cases_give_no_empty_blocks():
    import pandas as pd
    from finetune_chronos import outbreak_blocks

    rows = [
        ("P01", onset, onset + pd.Timedelta(days=d), 5)
        for onset in pd.date_range("2026-01-03", periods=140)
        for d in (0, 2, 6)
    ]
    counts = pd.DataFrame(
        rows, columns=["location", "reference_date", "report_date", "count"]
    )
    counts["reference_date"] = counts["reference_date"].dt.date.astype(str)
    counts["report_date"] = counts["report_date"].dt.date.astype(str)
    # P02 and P03 had no cases, so their leaves are never released.
    for seed in range(20):
        rng = np.random.default_rng(seed)
        for block in outbreak_blocks(counts, ["P01", "P02", "P03"], rng, 3):
            assert block["context"].shape[0] > 0


def _sim_dir(tmp_path):
    import pandas as pd

    rows, settings = [], []
    for seed in (6001, 6002):
        for sim in (1, 2):
            simulation = f"{seed}-{sim:04d}"
            settings.append((simulation, 2, 1, 28))
            for draw in (1, 2):
                for place, scale in (("P01", 1), ("P02", draw + sim)):
                    for onset in pd.date_range("2026-01-03", periods=120):
                        for d in (0, 2, 6):
                            report = onset + pd.Timedelta(days=d)
                            rows.append(
                                (
                                    f"{simulation}-{draw}",
                                    place,
                                    onset.date().isoformat(),
                                    report.date().isoformat(),
                                    scale,
                                )
                            )
    counts = pd.DataFrame(
        rows,
        columns=[
            "outbreak",
            "location",
            "reference_date",
            "report_date",
            "count",
        ],
    )
    table = pd.DataFrame(
        settings, columns=["outbreak", "n_places", "n_ages", "max_delay"]
    )
    for seed in (6001, 6002):
        part = counts["outbreak"].str.startswith(str(seed))
        counts[part].to_csv(tmp_path / f"counts-{seed}.csv.gz", index=False)
        keep = table["outbreak"].str.startswith(str(seed))
        table[keep].to_csv(tmp_path / f"settings-{seed}.csv", index=False)
    return [o for o, *_ in settings]


def test_blocks_cut_in_workers_match_blocks_cut_here(tmp_path):
    from finetune_stream import cut, cut_in_workers

    ids = _sim_dir(tmp_path)
    here = cut(tmp_path, ids, 1, 6)
    pairs, empty = cut_in_workers(None, tmp_path, ids, 1, 6, parts=3)
    assert [o for o, _ in pairs] == [o for o, _ in here]
    for (_, a), (_, b) in zip(pairs, here, strict=True):
        assert np.array_equal(
            np.asarray(a["context"]), np.asarray(b["context"]), equal_nan=True
        )
    assert empty == []


def test_blocks_cut_by_a_process_pool_match(tmp_path):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    import finetune_chronos
    from finetune_stream import cut, cut_in_workers, init_worker

    ids = _sim_dir(tmp_path)
    here = cut(tmp_path, ids, 1, 6)
    with ProcessPoolExecutor(
        2,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=init_worker,
        initargs=(
            finetune_chronos.MAX_SERIES,
            finetune_chronos.MAX_LAGS,
            finetune_chronos.SOURCES,
        ),
    ) as pool:
        pairs, _ = cut_in_workers(pool, tmp_path, ids, 1, 6, parts=2)
    assert [o for o, _ in pairs] == [o for o, _ in here]


def test_a_straggler_is_folded_into_a_later_chunk(tmp_path):
    from finetune_stream import next_chunk

    ids = _sim_dir(tmp_path)  # seeds 6001 and 6002, two simulations each
    # Seed 6002 finished; 6001 has not.
    for name in ("counts-6001.csv.gz", "settings-6001.csv"):
        (tmp_path / name).rename(tmp_path / f"late-{name}")
    seeds = range(6001, 6003)
    assert next_chunk(tmp_path, seeds, [], 2, waited=60, patience=1200) is None
    # After the patience, later finished seeds train first.
    late = next_chunk(tmp_path, seeds, [], 2, waited=1300, patience=1200)
    assert late == ids[2:]
    # The straggler joins the next chunk once it finishes.
    for name in ("counts-6001.csv.gz", "settings-6001.csv"):
        (tmp_path / f"late-{name}").rename(tmp_path / name)
    assert (
        next_chunk(tmp_path, seeds, late, 2, waited=0, patience=1200)
        == (ids[:2])
    )


def test_blocks_are_cut_without_torch(tmp_path):
    import subprocess

    ids = _sim_dir(tmp_path)
    # CI's test environment has no torch: cutting blocks must not need it.
    code = (
        "import builtins, sys\n"
        "real = builtins.__import__\n"
        "def no_torch(name, *args, **kwargs):\n"
        "    if name.split('.')[0] == 'torch':\n"
        "        raise ModuleNotFoundError(name)\n"
        "    return real(name, *args, **kwargs)\n"
        "builtins.__import__ = no_torch\n"
        "sys.path.insert(0, 'scripts'); import finetune_stream as fs\n"
        f"pairs = fs.cut(__import__('pathlib').Path({str(tmp_path)!r}), "
        f"{ids!r}, 1, 6)\nprint(len(pairs))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    assert int(out.stdout.split()[-1]) > 0


def test_learning_rate_decays_linearly_to_zero_over_its_steps():
    from finetune_stream import learning_rate

    # Constant without a schedule length.
    assert learning_rate(5000, 1e-5, 0) == 1e-5
    # Linear decay to zero over the steps, as Chronos-2's own fit.
    assert learning_rate(0, 1e-5, 1000) == 1e-5
    assert abs(learning_rate(500, 1e-5, 1000) - 5e-6) < 1e-15
    assert learning_rate(1000, 1e-5, 1000) == 0.0
    assert learning_rate(1500, 1e-5, 1000) == 0.0


def _line_list():
    import pandas as pd

    rows = []
    for week in pd.date_range("2026-05-02", "2026-09-26", freq="7D"):
        for place, n in (("CD-A", 6), ("CD-B", 3)):
            for k in range(n):
                rows.append((place, week.date().isoformat(), 2 + 3 * k))
    return pd.DataFrame(rows, columns=["location", "onset_week", "delay"])


def test_real_validation_blocks_stop_before_the_scored_weeks():
    import pytest
    from finetune_chronos import real_blocks

    cuts = ["2026-07-25", "2026-08-01", "2026-08-08"]
    blocks = real_blocks(_line_list(), cuts, last_week="2026-09-05", lag=4)
    assert len(blocks) == 3
    assert all(b["context"].shape[0] > 0 for b in blocks)
    # A cut whose forecast weeks reach past the last week is refused.
    with pytest.raises(ValueError, match="past"):
        real_blocks(_line_list(), ["2026-08-15"], "2026-09-05", 4)


def test_real_validation_context_holds_only_reports_by_the_cut():
    from finetune_chronos import CONTEXT, real_blocks

    late = _line_list()
    late.loc[len(late)] = ("CD-A", "2026-07-25", 30)  # reported 08-24
    a = real_blocks(_line_list(), ["2026-07-25"], "2026-09-05", 4)[0]
    b = real_blocks(late, ["2026-07-25"], "2026-09-05", 4)[0]
    start = CONTEXT
    # The late report changes the final count ahead, not the context.
    assert np.array_equal(
        a["context"][:, :start], b["context"][:, :start], equal_nan=True
    )
    assert not np.array_equal(a["context"], b["context"], equal_nan=True)


def _d(block) -> int:
    return block["n_covariates"] // block["n_targets"]


def test_one_source_cuts_only_that_source(tmp_path, monkeypatch):
    import finetune_chronos
    from finetune_stream import cut

    ids = _sim_dir(tmp_path)
    monkeypatch.setattr(finetune_chronos, "SOURCES", ("weekly",))
    blocks = [b for _, b in cut(tmp_path, ids, 1, 32)]
    assert blocks
    assert all(_d(b) <= 7 for b in blocks)


def test_workers_take_the_trainer_block_settings():
    import finetune_chronos
    from finetune_stream import init_worker

    saved = (
        finetune_chronos.MAX_SERIES,
        finetune_chronos.MAX_LAGS,
        finetune_chronos.SOURCES,
    )
    try:
        lags = {"daily": (28,)}
        init_worker(100, lags, ("daily",))
        assert finetune_chronos.MAX_SERIES == 100
        assert lags == finetune_chronos.MAX_LAGS
        assert finetune_chronos.SOURCES == ("daily",)
    finally:
        (
            finetune_chronos.MAX_SERIES,
            finetune_chronos.MAX_LAGS,
            finetune_chronos.SOURCES,
        ) = saved


def test_blocks_reach_deployed_sizes_on_many_places():
    import pandas as pd
    from finetune_chronos import outbreak_blocks

    places = [f"P{i:02d}" for i in range(1, 27)]
    rows = [
        (p, onset, onset + pd.Timedelta(days=d), 3)
        for p in places
        for onset in pd.date_range("2026-01-03", periods=280)
        for d in (0, 3, 9)
    ]
    counts = pd.DataFrame(
        rows, columns=["location", "reference_date", "report_date", "count"]
    )
    for col in ("reference_date", "report_date"):
        counts[col] = counts[col].dt.date.astype(str)
    big = []
    for seed in range(30):
        for b in outbreak_blocks(
            counts, places, np.random.default_rng(seed), 3
        ):
            big.append((b["n_targets"], b["context"].shape[0]))
    # BDBV's daily block: 27 targets and 783 series.
    assert any(t >= 27 and s >= 700 for t, s in big)


def test_three_day_source_cuts_three_day_blocks(tmp_path, monkeypatch):
    import finetune_chronos
    from finetune_stream import cut

    ids = _sim_dir(tmp_path)
    monkeypatch.setattr(finetune_chronos, "SOURCES", ("3day",))
    blocks = [b for _, b in cut(tmp_path, ids, 1, 32)]
    assert blocks
    for b in blocks:
        assert b["source"] == "3day"
        # D runs 5-16 steps of 3 days.
        assert 5 <= _d(b) <= 16


def test_blocks_predict_their_own_horizon_in_whole_patches(tmp_path):
    from finetune_chronos import CONTEXT, OUTPUT_PATCH
    from finetune_stream import cut

    ids = _sim_dir(tmp_path)
    for _, b in cut(tmp_path, ids, 1, 32):
        truth = b["context"][0, CONTEXT:]
        horizon = int((~np.isnan(truth)).sum())
        length = b["prediction_length"]
        assert length % OUTPUT_PATCH == 0
        assert length - OUTPUT_PATCH < horizon <= length
        assert b["context"].shape[1] == CONTEXT + length


def test_batches_share_one_prediction_length():
    from finetune_chronos import plan_by_length

    rng = np.random.default_rng(1)
    lengths = np.array([16] * 30 + [64] * 30 + [80] * 10)
    sizes = rng.integers(5, 300, len(lengths))
    plan = plan_by_length(lengths, sizes, 256, rng, 2000)
    used = sorted(i for length, batch in plan for i in batch)
    assert used == list(range(len(lengths)))
    for length, batch in plan:
        assert set(lengths[batch]) == {length}
        assert sizes[batch].sum() <= 2000 or len(batch) == 1


def test_a_seed_list_gives_the_training_seeds_in_its_order(tmp_path):
    from finetune_stream import read_seed_list

    path = tmp_path / "seeds.txt"
    path.write_text("10005\n10001\n\n10003\n")
    assert read_seed_list(path) == [10005, 10001, 10003]


def test_a_round_continues_from_the_previous_adapter(tmp_path):
    import pytest

    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from finetune_stream import lora_model
    from peft import LoraConfig

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name in "qkvo":
                setattr(self, name, torch.nn.Linear(4, 4))

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attention = Attention()

    config = LoraConfig(r=2, lora_alpha=4, target_modules=["self_attention.q"])
    torch.manual_seed(0)
    first = lora_model(Net(), config, None)
    for p in first.parameters():
        if p.requires_grad:
            p.data.normal_()
    first.save_pretrained(tmp_path / "round0")
    torch.manual_seed(1)
    second = lora_model(Net(), config, tmp_path / "round0")
    trained = {n: p for n, p in first.named_parameters() if "lora" in n}
    loaded = {n: p for n, p in second.named_parameters() if "lora" in n}
    assert loaded.keys() == trained.keys()
    assert all(torch.equal(loaded[n], trained[n]) for n in trained)
    # The loaded adapter trains on.
    assert all(p.requires_grad for p in loaded.values())


def test_real_training_blocks_spread_evenly_over_the_chunks():
    from finetune_stream import real_share

    real = list(range(10))
    parts = [real_share(real, chunk, 4) for chunk in range(4)]
    assert sorted(x for p in parts for x in p) == real
    assert {len(p) for p in parts} <= {2, 3}


def test_each_draw_is_cut_at_every_snapshot_once():
    from finetune_chronos import CUTS, SOURCES
    from finetune_stream import snapshots

    draws = [f"9001-0001-{d}" for d in range(1, 11)]
    picks = snapshots(draws, np.random.default_rng(0))
    # 10 draws x 16 cut-offs: 160 blocks, one source each.
    assert CUTS == 16
    assert len(picks) == 160
    assert len({(o, j) for o, _, j in picks}) == 160
    assert {s for _, s, _ in picks} <= set(SOURCES)


def test_level_mix_includes_single_series():
    rng = np.random.default_rng(0)
    mixes = [choose_places(STRATA, 10, rng) for _ in range(400)]
    single = [places for places, mix in mixes if mix == "single"]
    assert single
    assert all(len(places) == 1 and places[0] in STRATA for places in single)


def test_excluded_simulations_are_not_read(tmp_path, monkeypatch):
    import finetune_stream

    sims = tmp_path / "sims"
    sims.mkdir()
    _sim_dir(sims)
    exclude = tmp_path / "excluded.txt"
    monkeypatch.setattr(finetune_stream, "_READY", {})
    monkeypatch.setattr(finetune_stream, "EXCLUDE_FILE", exclude)
    exclude.write_text("6001-0001\n")
    assert finetune_stream.batch_outbreaks(sims, 6001) == ["6001-0002"]
    # Exclusions that arrive later apply to batches read later.
    exclude.write_text("simulation,reason\n6001-0001,r\n6002-0002,r\n")
    assert finetune_stream.batch_outbreaks(sims, 6002) == ["6002-0001"]


def test_a_pass_samples_draws_across_every_snapshot():
    from finetune_stream import snapshots

    draws = [f"9001-0001-{d}" for d in range(1, 11)]
    a = snapshots(draws, np.random.default_rng(0), n_draws=2, subset_seed=7)
    b = snapshots(draws, np.random.default_rng(5), n_draws=2, subset_seed=7)
    c = snapshots(draws, np.random.default_rng(0), n_draws=2, subset_seed=8)
    # 16 snapshots, two draws each: 32 blocks, spread over the 10 draws.
    assert len(a) == 32
    assert len({(o, j) for o, _, j in a}) == 32
    assert all(sum(j == k for _, _, j in a) == 2 for k in range(16))
    assert len({o for o, _, _ in a}) > 2
    # The subset seed alone picks the draws, so a later pass can take others.
    assert {(o, j) for o, _, j in b} == {(o, j) for o, _, j in a}
    assert {(o, j) for o, _, j in c} != {(o, j) for o, _, j in a}


def test_stale_block_files_fail_when_loaded(tmp_path):
    import pickle

    import pytest
    from finetune_stream import load_blocks

    good = {
        "context": np.zeros((1, 4)),
        "n_targets": 1,
        "prediction_length": 16,
        "levels": ("national",),
    }
    stale = {k: v for k, v in good.items() if k != "prediction_length"}
    path = tmp_path / "blocks.pkl"
    path.write_bytes(pickle.dumps([good]))
    assert [b["prediction_length"] for b in load_blocks(path)] == [16]
    path.write_bytes(pickle.dumps([good, stale]))
    with pytest.raises(ValueError, match="prediction_length"):
        load_blocks(path)


def test_every_source_is_native_or_binned():
    ft.check_sources(("weekly", "daily", "3day"))
    # A source with no binning would be cut as daily: refuse it.
    with pytest.raises(ValueError, match="5day"):
        ft.check_sources(("weekly", "5day"))


def test_the_package_must_be_this_worktrees(tmp_path):
    ft.check_package(Path(__file__).parents[1])
    with pytest.raises(RuntimeError, match="not this worktree"):
        ft.check_package(tmp_path)


def _fake_torch(monkeypatch):
    import types

    torch = types.ModuleType("torch")
    torch.save = lambda obj, path: Path(path).write_text(json.dumps(obj))
    torch.load = lambda path, **_: json.loads(Path(path).read_text())
    peft = types.ModuleType("peft")
    peft.get_peft_model_state_dict = lambda model: model["adapter"]
    peft.set_peft_model_state_dict = lambda model, sd: model.update(adapter=sd)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "peft", peft)


class _Optimiser:
    def __init__(self, step):
        self.step = step

    def state_dict(self):
        return {"step": self.step}

    def load_state_dict(self, sd):
        self.step = sd["step"]


def test_a_crash_while_saving_state_keeps_the_last_state(tmp_path, monkeypatch):
    from finetune_stream import load_state, save_state

    _fake_torch(monkeypatch)
    where = tmp_path / "state"
    save_state(where, {"adapter": [1]}, _Optimiser(1), {"step": 50})
    rename = Path.rename

    def crash(self, target):
        if self.name == "state.tmp":
            raise OSError("crash")
        return rename(self, target)

    monkeypatch.setattr(Path, "rename", crash)
    with pytest.raises(OSError, match="crash"):
        save_state(where, {"adapter": [2]}, _Optimiser(2), {"step": 100})
    monkeypatch.setattr(Path, "rename", rename)
    # The restarted trainer resumes from the last state that was saved.
    model, optimiser = {"adapter": None}, _Optimiser(0)
    assert load_state(where, model, optimiser) == {"step": 50}
    assert model["adapter"] == [1]
    assert optimiser.step == 1
    # The next save replaces it and leaves no old copy.
    save_state(where, {"adapter": [3]}, _Optimiser(3), {"step": 150})
    assert load_state(where, model, optimiser) == {"step": 150}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["state"]


def test_the_code_commit_is_the_scripts_repository(tmp_path, monkeypatch):
    from finetune_stream import code_commit

    here = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    # Launched from another repository, it still names this one's commit.
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)
    assert code_commit() == here
