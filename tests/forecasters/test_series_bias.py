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
    return series_bias.block_rows(N_PLACES, N_PLACES * LAG, [-1, 0, 0])


def test_zero_bias_is_base_chronos_bit_for_bit():
    base = tiny_model()
    model = tiny_model()
    series_bias.attach(model)
    block = toy_block()
    assert torch.equal(predict(model, block, toy_rows()), predict(base, block))


def masks(rows):
    return series_bias.pair_masks(*(torch.as_tensor(r) for r in rows))


def test_five_relations_tie_a_places_rows_and_its_parents():
    par = series_bias.parents_of(HIER)
    n = len(HIER)
    rows = series_bias.block_rows(n, 2 * n, par)
    place, _, role = rows
    count_lag, lag_count, lag_lag, up, down = masks(rows)
    for i in range(len(place)):
        for j in range(len(place)):
            p, q = place[i], place[j]
            same = p == q and i != j
            assert bool(count_lag[i, j]) == (
                same and role[i] == 0 and role[j] == 1
            )
            assert bool(lag_count[i, j]) == (
                same and role[i] == 1 and role[j] == 0
            )
            assert bool(lag_lag[i, j]) == (
                same and role[i] == 1 and role[j] == 1
            )
            assert bool(up[i, j]) == (par[p] == q)
            assert bool(down[i, j]) == (par[q] == p)
    # The diagonal keeps base Chronos-2: no relation holds there.
    for m in (count_lag, lag_count, lag_lag, up, down):
        assert not m.diagonal().any()
    # Any number of places: one row each, so no same-place pairs.
    big = series_bias.block_rows(300, 0, [-1] + [0] * 299)
    count_lag, lag_count, lag_lag, up, down = masks(big)
    assert not (count_lag | lag_count | lag_lag).any()
    assert int(up.sum()) == int(down.sum()) == 299


def test_the_five_biases_are_zero_fp32_scalars_per_layer_and_head():
    model = tiny_model().to(torch.bfloat16)
    series_bias.attach(model)
    module = getattr(model, series_bias.BIAS_MODULE)
    names = dict(module.named_parameters())
    assert set(names) == {"count_lag", "lag_count", "lag_lag", "up", "down"}
    for p in names.values():
        assert p.shape == (model.config.num_layers, model.config.num_heads)
        assert p.dtype == torch.float32
        assert not p.any()


def test_blocks_batched_together_keep_their_own_places():
    a = series_bias.block_rows(2, 2, [-1, 0])
    rows = series_bias.batch_rows([a, a])
    place, parent, role = rows
    assert list(place) == [0, 1, 0, 1, 2, 3, 2, 3]
    assert list(parent) == [-1, 0, -1, 0, -1, 2, -1, 2]
    assert list(role) == [0, 0, 1, 1, 0, 0, 1, 1]


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
    place, parent, role = _group_rows(group)
    assert list(role) == [0] * len(HIER) + [1] * len(names)
    par = series_bias.parents_of(HIER)
    at = {p: i for i, p in enumerate(HIER)}
    assert list(place[: len(HIER)]) == list(range(len(HIER)))
    assert list(parent[: len(HIER)]) == par
    for k, name in enumerate(sorted(names)):
        i = at[name.rsplit("_lag_", 1)[0]]
        assert place[len(HIER) + k] == i
        assert parent[len(HIER) + k] == par[i]
