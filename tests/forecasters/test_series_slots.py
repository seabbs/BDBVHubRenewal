"""Per-series slots: a place's count and lag series share a learned slot."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("chronos")
pytest.importorskip("peft")

from bdbvhubrenewal.forecasters import slots  # noqa: E402

N_PLACES, LAG, T, H = 3, 2, 48, 16


def tiny_model(seed: int = 0):
    """A small, randomly initialised Chronos-2 with no dropout."""
    from chronos.chronos2.config import Chronos2CoreConfig
    from chronos.chronos2.model import Chronos2Model

    torch.manual_seed(seed)
    config = Chronos2CoreConfig(
        d_model=32,
        d_kv=8,
        d_ff=64,
        num_layers=2,
        num_heads=4,
        dropout_rate=0.0,
        rope_theta=10000.0,
        dense_act_fn="relu",
        feed_forward_proj="relu",
        chronos_config={
            "context_length": 256,
            "input_patch_size": 16,
            "input_patch_stride": 16,
            "max_output_patches": 4,
            "output_patch_size": 16,
            "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "use_arcsinh": True,
            "use_reg_token": True,
        },
    )
    return Chronos2Model(config).eval()


def toy_block(seed: int = 0) -> dict:
    """A ragged block: three places' counts, then each place's two lags."""
    rng = np.random.default_rng(seed)
    level = np.array([5.0, 50.0, 500.0])[:, None]
    target = rng.poisson(level, (N_PLACES, T + H)).astype(np.float32)
    lags = np.stack(
        [target[p] * share for p in range(N_PLACES) for share in (0.4, 0.8)]
    ).astype(np.float32)
    lags[:, T + 4 :] = np.nan
    context = np.concatenate([target, lags])
    return {
        "context": context,
        "n_targets": N_PLACES,
        "n_covariates": N_PLACES * LAG,
        "n_future_covariates": N_PLACES * LAG,
        "prediction_length": H,
    }


def swapped(block: dict) -> dict:
    """The block with the lag series of places 0 and 2 swapped."""
    x = block["context"].copy()
    rows = x[N_PLACES:].reshape(N_PLACES, LAG, -1)
    x[N_PLACES:] = rows[[2, 1, 0]].reshape(N_PLACES * LAG, -1)
    return {**block, "context": x}


def batch_of(block: dict) -> dict:
    """Chronos-2's validation batch of a block, as the trainer builds it."""
    from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode

    prepared = {
        **block,
        "context": torch.from_numpy(block["context"]),
        "future_covariates": torch.from_numpy(block["context"][:, T:]),
    }
    data = Chronos2Dataset(
        inputs=[prepared],
        context_length=T,
        prediction_length=H,
        batch_size=1000,
        output_patch_size=16,
        mode=DatasetMode.VALIDATION,
    )
    return next(iter(data))


def predict(model, block: dict, ids=None):
    """The model's quantiles of the block's targets, with slots `ids`."""
    batch = batch_of(block)
    with slots.assigned(model, ids), torch.no_grad():
        out = model(
            context=batch["context"],
            group_ids=batch["group_ids"],
            future_covariates=batch["future_covariates"],
            num_output_patches=batch["num_output_patches"],
        )
    return out.quantile_preds[:N_PLACES]


def lora():
    from peft import LoraConfig

    return LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=["self_attention.q", "self_attention.v"],
        modules_to_save=[slots.SLOT_MODULE],
    )


def trained(steps: int = 30):
    """A tiny model with slots and LoRA after a few steps on the toy block."""
    from peft import get_peft_model

    model = tiny_model()
    slots.attach(model, n_slots=8)
    peft_model = get_peft_model(model, lora())
    params = [p for p in peft_model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=0.05)
    rng = np.random.default_rng(1)
    block = toy_block()
    peft_model.train()
    for _ in range(steps):
        batch = batch_of(block)
        ids = slots.slot_ids(N_PLACES, N_PLACES * LAG, rng, n_slots=8)
        with slots.assigned(peft_model, ids):
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
    peft_model.eval()
    return peft_model


def test_rows_of_a_place_share_its_slot():
    places = slots.row_places(3, 6)
    assert places.tolist() == [0, 1, 2, 0, 0, 1, 1, 2, 2]
    ids = slots.slot_ids(3, 6, np.random.default_rng(0), n_slots=8)
    assert len(set(ids[:3])) == 3
    for p in range(3):
        assert set(ids[places == p]) == {ids[p]}


def test_slots_are_a_fresh_permutation_each_draw_and_fixed_by_seed():
    rng = np.random.default_rng(0)
    draws = {tuple(slots.slot_ids(3, 6, rng, n_slots=64)) for _ in range(20)}
    assert len(draws) > 1
    a = slots.slot_ids(3, 6, np.random.default_rng(slots.EVAL_SEED))
    b = slots.slot_ids(3, 6, np.random.default_rng(slots.EVAL_SEED))
    assert a.tolist() == b.tolist()


def test_more_places_than_slots_is_an_error():
    with pytest.raises(ValueError, match="slots"):
        slots.slot_ids(9, 0, np.random.default_rng(0), n_slots=8)


def test_zero_slots_give_bit_identical_output():
    block = toy_block()
    before = predict(tiny_model(), block)
    model = tiny_model()
    slots.attach(model, n_slots=8)
    ids = slots.slot_ids(N_PLACES, N_PLACES * LAG, np.random.default_rng(0), 8)
    assert torch.equal(predict(model, block, ids), before)
    assert torch.equal(predict(model, block), before)


def test_without_slots_swapping_lags_between_places_changes_nothing():
    model, block = tiny_model(), toy_block()
    a, b = predict(model, block), predict(model, swapped(block))
    assert torch.allclose(a, b, atol=1e-5)


def test_trained_slots_tie_lags_to_their_place():
    model, block = trained(), toy_block()
    ids = slots.slot_ids(
        N_PLACES, N_PLACES * LAG, np.random.default_rng(slots.EVAL_SEED), 8
    )
    a, b = predict(model, block, ids), predict(model, swapped(block), ids)
    assert (a - b).abs().max() > 1e-3


def test_slots_round_trip_through_the_adapter(tmp_path):
    model, block = trained(), toy_block()
    ids = slots.slot_ids(
        N_PLACES, N_PLACES * LAG, np.random.default_rng(slots.EVAL_SEED), 8
    )
    want = predict(model, block, ids)
    model.save_pretrained(tmp_path / "ckpt")
    assert slots.adapter_slots(tmp_path / "ckpt") == 8
    loaded = slots.load_adapter(tiny_model(), tmp_path / "ckpt")
    assert torch.allclose(predict(loaded, block, ids), want, atol=1e-5)


def test_an_adapter_without_slots_loads_as_before(tmp_path):
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(r=2, lora_alpha=4, target_modules=["self_attention.q"])
    get_peft_model(tiny_model(), config).save_pretrained(tmp_path / "ckpt")
    assert slots.adapter_slots(tmp_path / "ckpt") is None
    loaded = slots.load_adapter(tiny_model(), tmp_path / "ckpt")
    assert not slots.has_slots(loaded)


def test_inference_assigns_slots_by_place(monkeypatch, tmp_path):
    from chronos import Chronos2Pipeline

    from bdbvhubrenewal.forecasters import foundation

    model = trained()
    model.save_pretrained(tmp_path / "ckpt")
    loaded = slots.load_adapter(tiny_model(), tmp_path / "ckpt")
    monkeypatch.setattr(
        foundation,
        "_chronos",
        lambda _checkpoint=None: Chronos2Pipeline(model=loaded),
    )

    def group(block):
        x = block["context"]
        names = [f"p{p}_lag_{d}" for p in range(N_PLACES) for d in range(LAG)]
        lags = x[N_PLACES:]
        return {
            "target": x[:N_PLACES, :T],
            "past_covariates": dict(zip(names, lags[:, :T], strict=True)),
            "future_covariates": dict(zip(names, lags[:, T:], strict=True)),
        }

    block = toy_block()
    a = foundation.chronos_sampler([group(block)], H, 50, 1)[0]
    again = foundation.chronos_sampler([group(block)], H, 50, 1)[0]
    b = foundation.chronos_sampler([group(swapped(block))], H, 50, 1)[0]
    assert np.array_equal(a, again)
    assert not np.allclose(a, b)


def test_slots_are_made_on_the_models_device():
    model = tiny_model().to("meta")
    slots.attach(model, 4)
    for p in getattr(model, slots.SLOT_MODULE).parameters():
        assert p.device.type == "meta"


def test_slots_can_start_as_distinct_random_codes():
    model = tiny_model()
    slots.attach(model, 8, init_std=0.02, seed=3)
    w = getattr(model, slots.SLOT_MODULE).embedding.weight.detach()
    assert torch.all(w.abs().sum(1) > 0)
    assert len({tuple(row.tolist()) for row in w}) == 8
    assert abs(float(w.std()) - 0.02) < 0.005
    again = tiny_model()
    slots.attach(again, 8, init_std=0.02, seed=3)
    assert torch.equal(w, getattr(again, slots.SLOT_MODULE).embedding.weight)


# National first and names out of sort order: regions of places, a place
# by ages and an age total across places, as training blocks hold them.
HIER = ("CD", "R2", "R1", "R2-P01", "R1-P03", "A1", "P05", "P05-A1")


def test_each_place_links_to_its_nearest_parent_in_the_block():
    parents = slots.parents_of(HIER)
    named = {
        p: (HIER[i] if i >= 0 else None)
        for p, i in zip(HIER, parents, strict=True)
    }
    assert named == {
        "CD": None,
        "R2": "CD",
        "R1": "CD",
        "R2-P01": "R2",
        "R1-P03": "R1",
        "A1": "CD",
        "P05": "CD",
        "P05-A1": "P05",
    }
    # Germany's states and age groups, and BDBV's provinces and zones.
    assert slots.parents_of(("DE", "DE-BY", "DE-BB")) == [-1, 0, 0]
    assert slots.parents_of(("DE", "80+", "00-04")) == [-1, 0, 0]
    assert slots.parents_of(("CD", "CD-NK", "CD-NK-03", "CD-IT-01")) == [
        -1,
        0,
        1,
        0,
    ]
    # A block without its country links leaves to their own parents only.
    assert slots.parents_of(("R1-P03", "R1")) == [1, -1]


def test_a_parents_slot_is_what_its_children_carry():
    n_t, lags = len(HIER), 2
    ids = slots.slot_ids(
        n_t,
        n_t * lags,
        np.random.default_rng(0),
        16,
        parents=slots.parents_of(HIER),
    )
    own, parent = ids
    assert ids.shape == (2, n_t * (1 + lags))
    # Every row of a place carries its own slot and its parent's own slot.
    for p, q in enumerate(slots.parents_of(HIER)):
        rows = slots.row_places(n_t, n_t * lags) == p
        assert set(own[rows]) == {own[p]}
        assert set(parent[rows]) == {own[q] if q >= 0 else -1}
    assert len(set(own[:n_t])) == n_t


def test_a_place_adds_its_parents_slot_exactly():
    model = tiny_model()
    slots.attach(model, 8, init_std=0.02, seed=1)
    module = getattr(model, slots.SLOT_MODULE)
    table = module.embedding.weight.detach()
    out = module(torch.tensor([[3, 5], [5, -1]])).detach()
    assert torch.allclose(out[0], table[3] + table[5])
    assert torch.allclose(out[1], table[5])
    # The slot table is the only parameter.
    assert [n for n, _ in module.named_parameters()] == ["embedding.weight"]
    # Own slots alone, as before.
    assert torch.allclose(module(torch.tensor([3, 5])).detach(), table[[3, 5]])


def test_inference_slots_follow_chronos_row_order_by_name():
    from bdbvhubrenewal.forecasters.foundation import _group_slots

    lags = 11  # lag 10 sorts before lag 2
    names = [f"{p}_lag_{d}" for p in HIER for d in range(lags)]
    cov = {n: np.zeros(4) for n in names}
    group = {
        "target": np.zeros((len(HIER), 4)),
        "past_covariates": cov,
        "future_covariates": {n: np.zeros(2) for n in names},
        "places": HIER,
    }
    own, parent = _group_slots(group, 16)
    chosen = slots.slot_ids(
        len(HIER), 0, np.random.default_rng(slots.EVAL_SEED), 16
    )
    parents = slots.parents_of(HIER)
    at = {p: i for i, p in enumerate(HIER)}

    def want_parent(i):
        return chosen[parents[i]] if parents[i] >= 0 else -1

    assert list(own[: len(HIER)]) == list(chosen)
    assert list(parent[: len(HIER)]) == [
        want_parent(i) for i in range(len(HIER))
    ]
    for k, name in enumerate(sorted(names)):
        i = at[name.rsplit("_lag_", 1)[0]]
        assert own[len(HIER) + k] == chosen[i]
        assert parent[len(HIER) + k] == want_parent(i)


def test_slots_without_a_parent_link_ignore_parent_ids(tmp_path):
    from peft import get_peft_model

    from bdbvhubrenewal.forecasters.foundation import _group_slots

    model = tiny_model()
    slots.attach(model, 8, init_std=0.02, seed=1, parent=False)
    module = getattr(model, slots.SLOT_MODULE)
    table = module.embedding.weight.detach()
    out = module(torch.tensor([[3], [5]])).detach()
    assert torch.allclose(out[0], table[3])
    assert not slots.has_parent_link(model)
    # An adapter saved without the link loads without it.
    get_peft_model(model, lora()).save_pretrained(tmp_path / "plain")
    assert slots.adapter_parent_link(tmp_path / "plain") is False
    with_link = tiny_model()
    slots.attach(with_link, 8, init_std=0.02, seed=1)
    get_peft_model(with_link, lora()).save_pretrained(tmp_path / "linked")
    assert slots.adapter_parent_link(tmp_path / "linked") is True
    names = [f"{p}_lag_0" for p in ("CD", "CD-NK")]
    group = {
        "target": np.zeros((2, 4)),
        "past_covariates": {n: np.zeros(4) for n in names},
        "places": ("CD", "CD-NK"),
    }
    assert _group_slots(group, 8, parent=False).ndim == 1
