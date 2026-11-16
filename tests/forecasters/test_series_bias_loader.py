"""The series bias at inference: masks by name, the loader and samplers."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("chronos")
pytest.importorskip("peft")

from test_series_bias import predict, toy_rows  # noqa: E402
from test_series_slots import LAG, N_PLACES, H, T, tiny_model, toy_block  # noqa: E402

from bdbvhubrenewal.forecasters import series_bias, slots  # noqa: E402


def trained_bias(steps: int = 5):
    """A tiny model whose bias and LoRA trained a few steps."""
    from peft import LoraConfig, get_peft_model
    from test_series_slots import batch_of

    model = tiny_model()
    series_bias.attach(model)
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
    parents = slots.parents_of(places)
    assert parents == [-1, 0, 0, 0, 0]
    rows = series_bias.block_rows(len(places), 2 * len(places), parents)
    same, linked = series_bias.pair_masks(
        torch.as_tensor(rows[0]), torch.as_tensor(rows[1])
    )
    place = rows[0]
    for q in range(len(place)):
        for k in range(len(place)):
            assert bool(same[q, k]) == (place[q] == place[k])
            want = (place[q] == 0) != (place[k] == 0)
            assert bool(linked[q, k]) == want
    # States and ages are siblings, not linked to each other.
    assert not linked[1, 2]


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
    parents = slots.parents_of(places)
    at = {p: i for i, p in enumerate(places)}
    for i, p in enumerate(places):
        if p == "CD":
            assert parents[i] == -1
        elif p.count("-") == 1:
            assert parents[i] == at["CD"]
        else:
            assert parents[i] == at[p.rsplit("-", 1)[0]]
    rows = series_bias.block_rows(n, 0, parents)
    assert rows[0].max() == n - 1
    same, linked = series_bias.pair_masks(
        torch.as_tensor(rows[0]), torch.as_tensor(rows[1])
    )
    assert int(same.sum()) == n
    # Each place with a parent is linked to it, both ways round.
    assert int(linked.sum()) == 2 * sum(q >= 0 for q in parents)


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
    rows = series_bias.block_rows(n, 0, slots.parents_of(bdbv_places(n)))
    out = predict(model, block, rows)
    assert out.shape[0] == n
    assert torch.isfinite(out).all()


def test_a_block_with_no_parents_links_nothing():
    places = ["R1", "R2", "R3"]
    parents = slots.parents_of(places)
    assert parents == [-1, -1, -1]
    rows = series_bias.block_rows(3, 6, parents)
    _, linked = series_bias.pair_masks(
        torch.as_tensor(rows[0]), torch.as_tensor(rows[1])
    )
    assert not linked.any()


def test_a_bias_adapter_loads_through_the_strict_loader(tmp_path):
    model = trained_bias()
    model.save_pretrained(tmp_path / "bias")
    loaded = slots.load_adapter(tiny_model(), tmp_path / "bias")
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
        slots.load_adapter(tiny_model(), tmp_path / "bias")


def test_inference_gives_a_bias_model_its_rows_by_name(monkeypatch, tmp_path):
    from contextlib import contextmanager

    from chronos import Chronos2Pipeline

    from bdbvhubrenewal.forecasters import foundation

    model = trained_bias()
    model.save_pretrained(tmp_path / "bias")
    loaded = slots.load_adapter(tiny_model(), tmp_path / "bias")
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
    # (finetune_stream.checkpoint_large_batches), after the rows are
    # unset; the bias must reach that recompute as an input.
    import sys
    from pathlib import Path

    from chronos.chronos2.model import Chronos2EncoderBlock
    from test_series_slots import batch_of

    sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
    import finetune_stream as fs

    def grads(checkpointed: bool):
        model = tiny_model()
        series_bias.attach(model)
        model.train()
        module = getattr(model, series_bias.BIAS_MODULE)
        with torch.no_grad():
            module.same.fill_(0.5)
            module.parent.fill_(-0.3)
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
        return module.same.grad, module.parent.grad

    plain, checked = grads(False), grads(True)
    assert torch.allclose(plain[0], checked[0], atol=1e-6)
    assert torch.allclose(plain[1], checked[1], atol=1e-6)
