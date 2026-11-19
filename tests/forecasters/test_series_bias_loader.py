"""The series bias at inference: masks by name, the loader and samplers."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("chronos")
pytest.importorskip("peft")

from chronos_toy import LAG, N_PLACES, H, T, tiny_model, toy_block  # noqa: E402
from test_series_bias import predict, toy_rows  # noqa: E402

from bdbvhubrenewal.forecasters import series_bias  # noqa: E402


def trained_bias(steps: int = 5, pairs=series_bias.PAIRS):
    """A tiny model whose bias (of `pairs`) and LoRA trained a few steps."""
    from chronos_toy import batch_of
    from peft import LoraConfig, get_peft_model

    model = tiny_model()
    series_bias.attach(model, pairs)
    config = LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=["self_attention.q"],
        modules_to_save=[series_bias.BIAS_MODULE],
    )
    peft_model = get_peft_model(model, config)
    params = [p for p in peft_model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=0.1)
    batch = batch_of(toy_block())
    for _ in range(steps):
        with series_bias.assigned(peft_model, toy_rows()):
            loss = peft_model(
                context=batch["context"],
                group_ids=batch["group_ids"],
                future_covariates=batch["future_covariates"],
                num_output_patches=batch["num_output_patches"],
                future_target=batch["future_target"],
            ).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    return peft_model.eval()


@pytest.mark.parametrize("attn", ["eager", "sdpa"])
def test_zero_bias_is_base_chronos_on_both_attention_paths(attn):
    base, model = tiny_model(), tiny_model()
    for m in (base, model):
        m.config._attn_implementation = attn
    series_bias.attach(model)
    block = toy_block()
    assert torch.equal(predict(model, block, toy_rows()), predict(base, block))


def test_germany_pairs_by_name():
    # Germany at inference: CD, CD-<state>, CD-A<i> (data/germany.py).
    places = ["CD", "CD-BY", "CD-BE", "CD-A0", "CD-A5"]
    parents = series_bias.parents_of(places)
    assert parents == [-1, 0, 0, 0, 0]
    rows = series_bias.block_rows(range(len(places)), 2 * len(places), parents)
    same, up, down = series_bias.pair_masks(
        torch.as_tensor(rows[0]), torch.as_tensor(rows[1])
    ).values()
    place = rows[0]
    for q in range(len(place)):
        for k in range(len(place)):
            assert bool(same[q, k]) == (place[q] == place[k] and q != k)
            assert bool(up[q, k]) == (place[q] != 0 and place[k] == 0)
            assert bool(down[q, k]) == (place[q] == 0 and place[k] != 0)
    # States and ages are siblings, not linked to each other.
    assert not up[1, 2]
    assert not down[1, 2]


def bdbv_places(n: int) -> list[str]:
    """CD, its provinces and their zones: `n` places in all."""
    places = ["CD"]
    p = 0
    while len(places) < n:
        province = f"CD-P{p:02d}"
        places.append(province)
        z = 0
        while len(places) < n and z < 30:
            places.append(f"{province}-{z:02d}")
            z += 1
        p += 1
    return places


@pytest.mark.parametrize("n", [600, 1000])
def test_bdbv_pairs_by_name_beyond_128_places(n):
    places = bdbv_places(n)
    parents = series_bias.parents_of(places)
    at = {p: i for i, p in enumerate(places)}
    for i, p in enumerate(places):
        if p == "CD":
            assert parents[i] == -1
        elif p.count("-") == 1:
            assert parents[i] == at["CD"]
        else:
            assert parents[i] == at[p.rsplit("-", 1)[0]]
    rows = series_bias.block_rows(range(n), 0, parents)
    assert rows[0].max() == n - 1
    same, up, down = series_bias.pair_masks(
        torch.as_tensor(rows[0]), torch.as_tensor(rows[1])
    ).values()
    assert not same.any()
    # Each place with a parent points up to it, and it down to the place.
    assert int(up.sum()) == int(down.sum()) == sum(q >= 0 for q in parents)
    assert torch.equal(up, down.T)


def test_a_1000_place_block_runs_with_the_bias():
    model = trained_bias()
    n = 1000
    rng = np.random.default_rng(0)
    target = rng.poisson(5.0, (n, T + H)).astype(np.float32)
    block = {
        "context": target,
        "n_targets": n,
        "n_covariates": 0,
        "n_future_covariates": 0,
        "prediction_length": H,
    }
    rows = series_bias.block_rows(
        range(n), 0, series_bias.parents_of(bdbv_places(n))
    )
    out = predict(model, block, rows)
    assert out.shape[0] == n
    assert torch.isfinite(out).all()


def test_a_block_with_no_parents_links_nothing():
    places = ["R1", "R2", "R3"]
    parents = series_bias.parents_of(places)
    assert parents == [-1, -1, -1]
    rows = series_bias.block_rows(range(3), 6, parents)
    _, up, down = series_bias.pair_masks(
        torch.as_tensor(rows[0]), torch.as_tensor(rows[1])
    ).values()
    assert not up.any()
    assert not down.any()


def test_a_bias_adapter_loads_through_the_strict_loader(tmp_path):
    model = trained_bias()
    model.save_pretrained(tmp_path / "bias")
    loaded = series_bias.load_adapter(tiny_model(), tmp_path / "bias")
    assert series_bias.has_bias(loaded)
    block = toy_block()
    want = predict(model, block, toy_rows())
    assert torch.allclose(predict(loaded, block, toy_rows()), want, atol=1e-5)


def test_a_dropped_bias_weight_is_an_error(tmp_path, monkeypatch):
    model = trained_bias()
    model.save_pretrained(tmp_path / "bias")
    # A loader that does not attach the bias would drop it silently.
    monkeypatch.setattr(series_bias, "adapter_has_bias", lambda _a: False)
    with pytest.raises(ValueError, match="has no place"):
        series_bias.load_adapter(tiny_model(), tmp_path / "bias")


def test_inference_gives_a_bias_model_its_rows_by_name(monkeypatch, tmp_path):
    from contextlib import contextmanager

    from chronos import Chronos2Pipeline

    from bdbvhubrenewal.forecasters import foundation

    model = trained_bias()
    model.save_pretrained(tmp_path / "bias")
    loaded = series_bias.load_adapter(tiny_model(), tmp_path / "bias")
    monkeypatch.setattr(
        foundation,
        "_chronos",
        lambda _checkpoint=None: Chronos2Pipeline(model=loaded),
    )
    seen = []
    real = series_bias.assigned

    @contextmanager
    def spy(m, rows):
        seen.append(np.asarray(rows))
        with real(m, rows):
            yield

    monkeypatch.setattr(series_bias, "assigned", spy)
    places = ["CD", "CD-KN", "CD-KN-01"]
    x = toy_block()["context"]
    names = [f"{p}_lag_{d}" for p in places for d in range(LAG)]
    lags = dict(zip(names, x[N_PLACES:], strict=True))
    group = {
        "target": x[:N_PLACES, :T],
        "past_covariates": {k: v[:T] for k, v in lags.items()},
        "future_covariates": {k: v[T:] for k, v in lags.items()},
        "places": places,
    }
    foundation.chronos_sampler([group], H, 5, 1)
    place, parent = seen[-1]
    assert list(place[:N_PLACES]) == [0, 1, 2]
    assert list(parent[:N_PLACES]) == [-1, 0, 1]
    # Lags in Chronos-2's sorted row order carry their own place.
    for k, name in enumerate(sorted(lags)):
        assert place[N_PLACES + k] == places.index(name.rsplit("_lag_", 1)[0])


def test_the_bias_survives_gradient_checkpointing(monkeypatch):
    # The trainer recomputes large batches' encoder blocks in backward
    # (finetune_stream.checkpoint_large_batches); with the rows still
    # assigned, the recompute adds the same bias.
    import sys
    from pathlib import Path

    from chronos.chronos2.model import Chronos2EncoderBlock
    from chronos_toy import batch_of

    sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
    import finetune_stream as fs

    def grads(checkpointed: bool):
        model = tiny_model()
        series_bias.attach(model)
        model.train()
        module = getattr(model, series_bias.BIAS_MODULE)
        with torch.no_grad():
            module.same.fill_(0.5)
            module.up.fill_(-0.3)
            module.down.fill_(0.4)
        if checkpointed:
            monkeypatch.setattr(
                Chronos2EncoderBlock, "forward", Chronos2EncoderBlock.forward
            )
            fs.checkpoint_large_batches(0)
        batch = batch_of(toy_block())
        with series_bias.assigned(model, toy_rows()):
            loss = model(
                context=batch["context"],
                group_ids=batch["group_ids"],
                future_covariates=batch["future_covariates"],
                num_output_patches=batch["num_output_patches"],
                future_target=batch["future_target"],
            ).loss
            loss.backward()
        return [p.grad for p in module.parameters()]

    plain, checked = grads(False), grads(True)
    for a, b in zip(plain, checked, strict=True):
        assert torch.allclose(a, b, atol=1e-6)


def test_an_adapter_holds_and_states_its_pairs(tmp_path):
    trained_bias(1).save_pretrained(tmp_path / "bias")
    assert series_bias.adapter_pairs(tmp_path / "bias") == series_bias.PAIRS


def test_an_adapter_loads_with_the_pairs_it_was_trained_with(tmp_path):
    # Checkpoints already published hold one more kind of pair; they load
    # as saved, so their scores do not change.
    pairs = (series_bias._SAVED_ONLY, *series_bias.PAIRS)
    model = trained_bias(pairs=pairs)
    model.save_pretrained(tmp_path / "bias")
    assert series_bias.adapter_pairs(tmp_path / "bias") == pairs
    loaded = series_bias.load_adapter(tiny_model(), tmp_path / "bias")
    assert series_bias.model_pairs(loaded) == pairs
    block = toy_block()
    want = predict(model, block, toy_rows())
    assert torch.allclose(predict(loaded, block, toy_rows()), want, atol=1e-5)


def test_an_unknown_kind_of_pair_is_refused():
    with pytest.raises(ValueError, match="unknown series bias pairs"):
        series_bias.pairs_in([f"m.{series_bias.BIAS_MODULE}.sideways"])
    with pytest.raises(ValueError, match="unknown series bias pairs"):
        series_bias.attach(tiny_model(), ("sideways",))


def test_attach_for_gives_a_model_its_adapter_s_pairs(tmp_path):
    trained_bias(1).save_pretrained(tmp_path / "bias")
    model = tiny_model()
    assert series_bias.attach_for(model, tmp_path / "bias")
    assert series_bias.model_pairs(model) == series_bias.PAIRS


def test_masks_are_made_only_for_the_pairs_asked():
    place = torch.as_tensor([0, 0, 1])
    parent = torch.as_tensor([-1, -1, 0])
    assert list(series_bias.pair_masks(place, parent)) == list(
        series_bias.PAIRS
    )
    pairs = (series_bias._SAVED_ONLY, *series_bias.PAIRS)
    got = series_bias.pair_masks(place, parent, pairs)
    assert torch.equal(
        got[series_bias._SAVED_ONLY], torch.eye(3, dtype=torch.bool)
    )
