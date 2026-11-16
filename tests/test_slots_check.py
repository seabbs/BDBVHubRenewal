"""The slots place-swap check (`scripts/slots_check.py`).

A swap tests the slots only if each row's slot stays with its position
(the place of the target the row sits beside) while the lag content moves
to another place. These tests pin that down on a tiny Chronos-2.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("chronos")
pytest.importorskip("peft")

sys.path.insert(0, str(Path(__file__).parent / "forecasters"))
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import covariate_ablation as ca  # noqa: E402
import finetune_stream as fs  # noqa: E402
import slots_check as sc  # noqa: E402
from test_series_slots import (  # noqa: E402
    LAG,
    N_PLACES,
    H,
    T,
    tiny_model,
    toy_block,
)

from bdbvhubrenewal.forecasters import foundation, slots  # noqa: E402

N_SLOTS = 8


@pytest.fixture(autouse=True)
def short_context(monkeypatch):
    # The toy blocks have 48 steps of context, not the trainer's 288.
    monkeypatch.setattr(ca.ft, "CONTEXT", T)


def block() -> dict:
    """A ragged block whose places differ in shape, not only in noise.

    Chronos-2 scales each series by its own context, so places that
    differ only in level look alike to it. Here one place rises, one
    falls and one peaks, so whose lags sit where matters.
    """
    s = np.arange(T + H)
    target = np.stack(
        [10 + 2.0 * s, 200 - 2.5 * s, 50 + 40 * np.exp(-(((s - 40) / 8) ** 2))]
    ).astype(np.float32)
    lags = np.stack(
        [target[p] * share for p in range(N_PLACES) for share in (0.4, 0.8)]
    ).astype(np.float32)
    lags[:, T + 4 :] = np.nan
    return {
        **toy_block(),
        "context": np.concatenate([target, lags]),
        "start": T,
        "horizon": H,
    }


def with_slots(seed: int | None = 3, scale: float = 0.1):
    """A tiny model with slots: random nonzero ones, or zero if no seed."""
    model = tiny_model()
    slots.attach(model, n_slots=N_SLOTS)
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
        weight = getattr(model, slots.SLOT_MODULE).embedding.weight
        with torch.no_grad():
            weight.copy_(scale * torch.randn(weight.shape, generator=gen))
    return model


def swap(b: dict) -> dict:
    return ca.condition(b, "swap_places", np.random.default_rng(0))


def test_swap_moves_lag_content_but_not_slots():
    model = with_slots()
    given = ca.condition(block(), "given", np.random.default_rng(0))
    swapped = swap(block())
    lags = slice(N_PLACES, None)
    assert not np.array_equal(
        given["context"][lags], swapped["context"][lags], equal_nan=True
    )
    assert np.array_equal(
        given["context"][:N_PLACES], swapped["context"][:N_PLACES]
    )
    assert np.array_equal(
        sc.model_ids(model, given), sc.model_ids(model, swapped)
    )


def test_swap_changes_output_with_nonzero_slots():
    model = with_slots()
    _, given = sc.predict(model, ca.condition(block(), "given", None))
    _, swapped = sc.predict(model, swap(block()))
    change = sc.output_change(given, swapped, LAG, H)
    # An untrained model mixes the group nearly evenly, so the effect is
    # small; it is still three orders above the rounding the swap causes
    # without slots (below).
    assert change["nowcast_max"] > 1e-4
    assert change["forecast_max"] > 1e-4


def test_moving_slots_with_the_lags_leaves_output_unchanged():
    # The trap: if each lag series took its slot with it, the swap would
    # be a mere reordering of rows, which Chronos-2 cannot see.
    model = with_slots()
    b = ca.condition(block(), "given", None)
    ids = sc.model_ids(model, b)
    perm = np.array([2, 0, 1])
    x = b["context"].copy()
    rows = x[N_PLACES:].reshape(N_PLACES, LAG, -1)
    x[N_PLACES:] = rows[perm].reshape(N_PLACES * LAG, -1)
    moved = ids.copy()
    moved[N_PLACES:] = ids[N_PLACES:].reshape(N_PLACES, LAG)[perm].ravel()
    _, given = sc.predict(model, b)
    with slots.assigned(model, moved):
        # `predict` sets the positional slots itself; this checks the same
        # forward with the slots moved along with the rows.
        batch = sc._batch({**b, "context": x, "future_covariates": x[:, T:]})
        out = sc._forward(model, batch)
    p = sc._scaled(model, batch, out)
    assert torch.allclose(given, p, atol=1e-5)


def test_zero_slots_are_bit_identical_to_base_given_and_swapped():
    base = tiny_model()
    zero = with_slots(seed=None)
    for b in (ca.condition(block(), "given", None), swap(block())):
        lb, pb = sc.predict(base, b)
        lz, pz = sc.predict(zero, b)
        assert torch.equal(pb, pz)
        assert torch.equal(lb, lz)


def test_base_swap_changes_output_only_by_rounding():
    # The rows are summed in another order, so not always bit-identical.
    base = tiny_model()
    _, given = sc.predict(base, ca.condition(block(), "given", None))
    _, swapped = sc.predict(base, swap(block()))
    change = sc.output_change(given, swapped, LAG, H)
    assert change["nowcast_max"] < 1e-5
    assert change["forecast_max"] < 1e-5


@pytest.mark.parametrize("name", ["given", "removed", "swap_places"])
def test_slots_match_the_trainer_and_inference(monkeypatch, name):
    from chronos import Chronos2Pipeline

    model = with_slots()
    b = ca.condition(block(), name, np.random.default_rng(0))
    ids = sc.model_ids(model, b)
    assert np.array_equal(ids, fs.block_slots([b], N_SLOTS))
    # The ids chronos_sampler gives the same group.
    seen = []
    real = slots.assigned

    def record(m, i):
        seen.append(np.asarray(i))
        return real(m, i)

    monkeypatch.setattr(slots, "assigned", record)
    monkeypatch.setattr(
        foundation, "_chronos", lambda _c=None: Chronos2Pipeline(model=model)
    )
    x = b["context"]
    k = b["n_covariates"]
    names = [f"p{p}_lag_{d}" for p in range(N_PLACES) for d in range(LAG)]
    group = {"target": x[:N_PLACES, :T]}
    if k:
        group["past_covariates"] = dict(
            zip(names, x[N_PLACES:, :T], strict=True)
        )
        group["future_covariates"] = dict(
            zip(names, x[N_PLACES:, T:], strict=True)
        )
    foundation.chronos_sampler([group], H, 5, 1)
    assert len(seen) == 1
    assert np.array_equal(seen[0], ids)


def test_no_slots_gives_no_ids():
    assert sc.model_ids(tiny_model(), block()) is None


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "chronos 2.3.2 sorts dict covariate keys, so CD-* lag rows come "
        "before CD's while this branch's chronos_sampler gives slots in "
        "input order; fixed upstream by _group_slots (checkpoint-scoring "
        "1e124b859, slots-loader-roworder 9bca8068c)"
    ),
)
def test_inference_rows_keep_their_place_slot(monkeypatch):
    """Each prepared row gets its own place's slot at inference.

    Fails on this branch's `foundation.chronos_sampler`. The fix
    (`_group_slots`, which reads the group's "places") is on
    checkpoint-scoring 1e124b859 and slots-loader-roworder 9bca8068c, so
    this test is expected to XPASS, and so fail as strict, once that code
    is merged here; then drop the marker. The check itself scores tensor
    blocks in the trainer's order and is not affected.
    """
    from chronos import Chronos2Pipeline

    model = with_slots()
    places = ["CD", "CD-KN", "CD-EQ"]
    b = block()
    x = b["context"]
    names = [f"{p}_lag_{d}" for p in places for d in range(LAG)]
    group = {
        "target": x[:N_PLACES, :T],
        "past_covariates": dict(zip(names, x[N_PLACES:, :T], strict=True)),
        "future_covariates": dict(zip(names, x[N_PLACES:, T:], strict=True)),
    }
    if hasattr(foundation, "_group_slots"):
        # The fixed sampler reads the places layout_block now sets.
        group["places"] = places
    seen = {}
    real = slots.assigned

    def record(m, i):
        seen["ids"] = np.asarray(i)
        return real(m, i)

    def grab(_module, args, kwargs):
        seen["context"] = kwargs.get("context", args[0] if args else None)

    model.register_forward_pre_hook(grab, with_kwargs=True)
    monkeypatch.setattr(slots, "assigned", record)
    monkeypatch.setattr(
        foundation, "_chronos", lambda _c=None: Chronos2Pipeline(model=model)
    )
    foundation.chronos_sampler([group], H, 5, 1)
    ctx = seen["context"].numpy()
    ids = seen["ids"]
    lags = x[N_PLACES:, :T].reshape(N_PLACES, LAG, T)
    for row in range(N_PLACES, ctx.shape[0]):
        tail = ctx[row, -T:]
        owner = [
            p
            for p in range(N_PLACES)
            if any(
                np.allclose(tail, lags[p, d], equal_nan=True)
                for d in range(LAG)
            )
        ]
        assert owner, row
        assert ids[row] == ids[owner[0]], (row, owner)


def test_synthetic_adapter_loads_random_slots_and_a_no_op_lora(tmp_path):
    sc.synthetic_adapter(tiny_model(), tmp_path / "ad", N_SLOTS, 3, 1.0)
    assert slots.adapter_slots(tmp_path / "ad") == N_SLOTS
    loaded = slots.load_adapter(tiny_model(), tmp_path / "ad")
    weight = loaded.series_slots.embedding.weight
    assert float(weight.abs().min()) > 0
    base = tiny_model().state_dict()
    for name, value in loaded.state_dict().items():
        if name in base:
            assert torch.equal(value, base[name]), name


def fake_german(lag: int, source: str, places: str) -> dict:
    """A padded German block as `finetune_chronos.german_blocks` makes."""
    ahead = 28 if source == "daily" else 4
    h = lag + ahead
    n = 17 if places == "states" else 1
    length = -(-h // 16) * 16
    c = ca.ft.CONTEXT
    x = np.full((n + n * lag, c + length), np.nan, dtype=np.float32)
    start = 20
    x[:n, c - start : c + h] = 1.0
    x[n:, c - start : c] = 0.5
    return {
        "context": x,
        "future_covariates": x[:, c:],
        "n_targets": n,
        "n_covariates": n * lag,
        "n_future_covariates": n * lag,
        "prediction_length": length,
        "source": source,
        "lag": lag,
        "places": places,
        "cut": "2021-05-01",
    }


def test_german_blocks_are_states_with_start_and_horizon(tmp_path):
    blocks = [
        fake_german(2, "weekly", "states"),
        fake_german(3, "weekly", "states"),
        fake_german(2, "weekly", "national"),
        fake_german(14, "daily", "states"),
        fake_german(49, "daily", "states"),
    ]
    path = tmp_path / "germany.pkl"
    path.write_bytes(pickle.dumps(blocks))
    got = sc.germany_blocks(path, per_source=5, max_rows=300, seed=1)
    assert {b["n_targets"] for b in got} == {17}
    # 17 x (1 + 49) rows is over the cap.
    assert sorted((b["source"], b["lag"]) for b in got) == [
        ("daily", 14),
        ("weekly", 2),
        ("weekly", 3),
    ]
    for b in got:
        assert b["horizon"] == b["lag"] + (28 if b["source"] == "daily" else 4)
        assert b["start"] == 20
        assert b["set"] == "germany"


def checkpoint(root: Path, name: str, n: int, h: str, with_slots: bool):
    """A checkpoint folder as the trainer's driver copies it."""
    d = root / name
    (d / "finetuned-ckpt").mkdir(parents=True)
    settings = {"hash": h, "n_outbreaks": n}
    if with_slots:
        settings["series_slots"] = {"n": 128}
    (d / "settings.json").write_text(json.dumps(settings))
    return d / "finetuned-ckpt"


def test_comparator_is_the_no_slots_checkpoint_at_the_same_n(tmp_path):
    noslots = tmp_path / "checkpoints"
    want = checkpoint(noslots, "n1000", 1000, "de129103cacc", False)
    checkpoint(noslots, "n2000", 2000, "7962fd718a47", False)
    s1000 = checkpoint(tmp_path / "slots", "n1000", 1000, "aaa", True)
    s250 = checkpoint(tmp_path / "slots", "n250", 250, "bbb", True)
    assert sc.comparator(s1000, noslots) == ("de129103cacc", want, 1000)
    assert sc.comparator(s250, noslots) == (None, None, 250)


def test_comparator_reads_a_release_cache_holding_settings(tmp_path):
    noslots = tmp_path / "checkpoints"
    checkpoint(noslots, "n2000", 2000, "7962fd718a47", False)
    cache = tmp_path / "chronos-slots-ccc"
    cache.mkdir()
    (cache / "settings.json").write_text(
        json.dumps({"hash": "ccc", "n_outbreaks": 2000, "series_slots": {}})
    )
    got = sc.comparator(cache, noslots)
    assert got[0] == "7962fd718a47"
    assert got[2] == 2000


def per_block() -> pd.DataFrame:
    """Per-block rows of three models on given, two sources and 3-day."""
    rows = []
    for key, source in enumerate(["weekly", "weekly", "daily", "3day"]):
        for model, loss in (
            ("zero-shot", 4.0),
            ("no-slots", 2.0),
            ("slots", 1.0),
        ):
            for cond in ("given", "swap_places"):
                rows.append(
                    {
                        "key": key,
                        "set": "sim",
                        "source": source,
                        "n_targets": 3,
                        "model": model,
                        "condition": cond,
                        "nowcast": loss * (1.5 if cond != "given" else 1),
                        "forecast": loss,
                        "change_nowcast_max": 0.1 if model == "slots" else 0,
                        "change_nowcast_mean": 0.0,
                        "change_forecast_max": 0.0,
                        "change_forecast_mean": 0.0,
                        "slot_sd": 0.01 if model == "slots" else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def test_summary_gives_given_loss_by_source_slots_against_no_slots():
    s = sc.summarise_rows(per_block(), n_boot=50)
    vs = s[
        (s["model"] == "slots")
        & (s["against"] == "no-slots")
        & (s["condition"] == "given")
        & (s["window"] == "nowcast")
    ].set_index("set")
    # 3-day blocks are reported apart from daily ones.
    assert {"sims weekly", "sims daily", "sims 3day", "sims all"} <= set(
        vs.index
    )
    assert vs.loc["sims weekly", "n_blocks"] == 2
    assert vs.loc["sims 3day", "n_blocks"] == 1
    assert vs.loc["sims all", "ratio"] == pytest.approx(0.5)
    assert vs.loc["sims all", "loss"] == pytest.approx(1.0)
    assert vs.loc["sims all", "against_loss"] == pytest.approx(2.0)
    zs = s[
        (s["model"] == "slots")
        & (s["against"] == "zero-shot")
        & (s["set"] == "sims all")
        & (s["window"] == "nowcast")
    ]
    assert zs["ratio"].item() == pytest.approx(0.25)


def test_summary_compares_each_condition_with_given():
    s = sc.summarise_rows(per_block(), n_boot=50)
    swap = s[
        (s["condition"] == "swap_places")
        & (s["against"] == "given")
        & (s["set"] == "sims all")
        & (s["window"] == "nowcast")
    ].set_index("model")
    assert swap.loc["slots", "ratio"] == pytest.approx(1.5)
    assert swap.loc["slots", "n_above_floor"] == 4
    assert swap.loc["no-slots", "n_unchanged"] == 4


def test_summary_without_a_comparator_compares_with_zero_shot_only():
    d = per_block()
    d = d[d["model"] != "no-slots"]
    s = sc.summarise_rows(d, n_boot=50)
    given = s[(s["condition"] == "given") & (s["model"] == "slots")]
    assert set(given["against"]) == {"zero-shot"}


def test_headline_work_comes_first_with_a_comparator():
    first = sc.phase_work("headline", has_comparator=True)
    rest = sc.phase_work("rest", has_comparator=True)
    assert first == {
        ("slots", "given"),
        ("slots", "swap_places"),
        ("no-slots", "given"),
        ("no-slots", "swap_places"),
    }
    every = {
        (m, c)
        for m in ("zero-shot", "no-slots", "slots")
        for c in sc.CONDITIONS
    }
    assert first | rest == every
    assert not first & rest


def test_headline_work_without_a_comparator_uses_zero_shot():
    first = sc.phase_work("headline", has_comparator=False)
    rest = sc.phase_work("rest", has_comparator=False)
    assert ("zero-shot", "given") in first
    assert ("slots", "swap_places") in first
    assert all(m != "no-slots" for m, _ in first | rest)
    every = {(m, c) for m in ("zero-shot", "slots") for c in sc.CONDITIONS}
    assert first | rest == every


def test_interim_summary_has_slots_against_no_slots_only():
    d = per_block()
    d = d[(d["model"] != "zero-shot")]
    s = sc.summarise_rows(d, n_boot=50)
    given = s[(s["condition"] == "given") & (s["model"] == "slots")]
    assert set(given["against"]) == {"no-slots"}
    swap = s[
        (s["condition"] == "swap_places")
        & (s["model"] == "slots")
        & (s["window"] == "nowcast")
    ]
    assert (swap["n_above_floor"] > 0).all()
