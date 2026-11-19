"""Attention bias for same-place and parent-child series pairs."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("chronos")
pytest.importorskip("peft")

from chronos_toy import (  # noqa: E402
    HIER,
    LAG,
    N_PLACES,
    batch_of,
    tiny_model,
    toy_block,
)

from bdbvhubrenewal.forecasters import series_bias  # noqa: E402


def predict(model, block, rows=None):
    batch = batch_of(block)
    with series_bias.assigned(model, rows), torch.no_grad():
        out = model(
            context=batch["context"],
            group_ids=batch["group_ids"],
            future_covariates=batch["future_covariates"],
            num_output_patches=batch["num_output_patches"],
        )
    return out.quantile_preds


def toy_rows():
    return series_bias.block_rows(range(N_PLACES), N_PLACES * LAG, [-1, 0, 0])


def test_zero_bias_is_base_chronos_bit_for_bit():
    base = tiny_model()
    model = tiny_model()
    series_bias.attach(model)
    block = toy_block()
    assert torch.equal(predict(model, block, toy_rows()), predict(base, block))


def masks(place, parent):
    return series_bias.pair_masks(
        torch.as_tensor(place), torch.as_tensor(parent)
    )


def test_pairs_mark_same_place_child_to_parent_and_back():
    # Any number of places.
    n = 300
    got = masks(*series_bias.block_rows(range(n), 0, [-1] + [0] * (n - 1)))
    assert list(got) == list(series_bias.PAIRS) == ["same", "up", "down"]
    same, up, down = got.values()
    assert same.shape == up.shape == down.shape == (n, n)
    assert not same.any()  # one row per place: no other row of it
    assert bool(up[1, 0])  # a child's row attends to its parent's
    assert bool(down[0, 1])  # and the parent's to its child's
    assert not bool(up[0, 1])
    assert not bool(down[1, 0])
    assert not bool(up[1, 2])
    assert not bool(down[1, 2])
    # Each place's lags are its own: same with each other and its count.
    par = series_bias.parents_of(HIER)
    rows = series_bias.row_places(len(HIER), 2 * len(HIER))
    same, up, down = masks(
        *series_bias.block_rows(range(len(HIER)), 2 * len(HIER), par)
    ).values()
    for i, p in enumerate(rows):
        for j, q in enumerate(rows):
            assert bool(same[i, j]) == (p == q and i != j)
            assert bool(up[i, j]) == (par[p] == q)
            assert bool(down[i, j]) == (par[q] == p)


def test_the_three_biases_are_zero_fp32_scalars_per_layer_and_head():
    model = tiny_model().to(torch.bfloat16)
    series_bias.attach(model)
    module = getattr(model, series_bias.BIAS_MODULE)
    names = dict(module.named_parameters())
    assert set(names) == {"same", "up", "down"}
    for p in names.values():
        assert p.shape == (model.config.num_layers, model.config.num_heads)
        assert p.dtype == torch.float32
        assert not p.any()


def test_blocks_batched_together_keep_their_own_places():
    a = series_bias.block_rows(range(2), 2, [-1, 0])
    rows = series_bias.batch_rows([a, a])
    place, parent = rows
    assert list(place) == [0, 1, 0, 1, 2, 3, 2, 3]
    assert list(parent) == [-1, 0, -1, 0, -1, 2, -1, 2]


def test_the_bias_trains_moves_the_output_and_saves(tmp_path):
    from peft import LoraConfig, PeftModel, get_peft_model

    model = tiny_model()
    series_bias.attach(model)
    config = LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=["self_attention.q"],
        modules_to_save=[series_bias.BIAS_MODULE],
    )
    peft_model = get_peft_model(model, config)
    params = [
        p
        for n, p in peft_model.named_parameters()
        if series_bias.BIAS_MODULE in n and p.requires_grad
    ]
    assert params
    opt = torch.optim.Adam(params, lr=0.1)
    block = toy_block()
    batch = batch_of(block)
    for _ in range(5):
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
    assert any(float(p.abs().sum()) > 0 for p in params)
    peft_model.save_pretrained(tmp_path / "ad")
    assert series_bias.adapter_has_bias(tmp_path / "ad")
    fresh = tiny_model()
    series_bias.attach(fresh)
    loaded = PeftModel.from_pretrained(fresh, tmp_path / "ad")
    got = [
        p
        for n, p in loaded.named_parameters()
        if series_bias.BIAS_MODULE in n and "modules_to_save" in n
    ]
    assert all(torch.equal(a, b) for a, b in zip(params, got, strict=True))
    assert not torch.equal(
        predict(loaded, block, toy_rows()), predict(tiny_model(), block)
    )


def test_inference_rows_follow_chronos_order_by_name():
    from bdbvhubrenewal.forecasters.foundation import _group_rows

    lags = 11
    names = [f"{p}_lag_{d}" for p in HIER for d in range(lags)]
    group = {
        "target": np.zeros((len(HIER), 4)),
        "past_covariates": {n: np.zeros(4) for n in names},
        "future_covariates": {n: np.zeros(2) for n in names},
        "places": HIER,
    }
    place, parent = _group_rows(group)
    par = series_bias.parents_of(HIER)
    at = {p: i for i, p in enumerate(HIER)}
    assert list(place[: len(HIER)]) == list(range(len(HIER)))
    assert list(parent[: len(HIER)]) == par
    for k, name in enumerate(sorted(names)):
        i = at[name.rsplit("_lag_", 1)[0]]
        assert place[len(HIER) + k] == i
        assert parent[len(HIER) + k] == par[i]
