"""Per-series slots: a place's count and lag series share a learned slot."""

import json

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
    loaded = slots.load_adapter(tiny_model(), tmp_path / "ckpt", parent=False)
    assert torch.allclose(predict(loaded, block, ids), want, atol=1e-5)


def test_an_adapter_without_slots_loads_as_before(tmp_path):
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(r=2, lora_alpha=4, target_modules=["self_attention.q"])
    get_peft_model(tiny_model(), config).save_pretrained(tmp_path / "ckpt")
    assert slots.adapter_slots(tmp_path / "ckpt") is None
    loaded = slots.load_adapter(tiny_model(), tmp_path / "ckpt", parent=False)
    assert not slots.has_slots(loaded)


def test_inference_assigns_slots_by_place(monkeypatch, tmp_path):
    from chronos import Chronos2Pipeline

    from bdbvhubrenewal.forecasters import foundation

    model = trained()
    model.save_pretrained(tmp_path / "ckpt")
    loaded = slots.load_adapter(tiny_model(), tmp_path / "ckpt", parent=False)
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


@pytest.mark.parametrize("drop_lags_of", [None, "CD-EQ"])
def test_inference_slots_follow_chronos_row_order(
    monkeypatch, tmp_path, drop_lags_of
):
    # Chronos-2's dict path stacks covariates by sorted name (past-only
    # first, then known-future), not in input order. With hub names "-"
    # sorts before "_", so "CD_lag_*" comes after "CD-KN_lag_*": each lag
    # row must still get its own place's slot. A place may have no lags.
    from contextlib import contextmanager

    from chronos import Chronos2Pipeline

    from bdbvhubrenewal.forecasters import foundation

    model = trained()
    model.save_pretrained(tmp_path / "ckpt")
    loaded = slots.load_adapter(tiny_model(), tmp_path / "ckpt", parent=False)
    monkeypatch.setattr(
        foundation,
        "_chronos",
        lambda _checkpoint=None: Chronos2Pipeline(model=loaded),
    )
    seen = []
    real = slots.assigned

    @contextmanager
    def spy(m, ids):
        seen.append(np.asarray(ids))
        with real(m, ids):
            yield

    monkeypatch.setattr(slots, "assigned", spy)
    places = ["CD", "CD-KN", "CD-EQ"]
    x = toy_block()["context"]
    names = [f"{p}_lag_{d}" for p in places for d in range(LAG)]
    rows = dict(zip(names, x[N_PLACES:], strict=True))
    if drop_lags_of:
        rows = {
            k: v
            for k, v in rows.items()
            if not k.startswith(drop_lags_of + "_")
        }
    group = {
        "target": x[:N_PLACES, :T],
        "past_covariates": {k: v[:T] for k, v in rows.items()},
        "future_covariates": {k: v[T:] for k, v in rows.items()},
        "places": places,
    }
    foundation.chronos_sampler([group], H, 5, 1)
    ids = seen[-1]
    target_slot = dict(zip(places, ids[:N_PLACES], strict=True))
    for k, name in enumerate(sorted(rows)):
        assert ids[N_PLACES + k] == target_slot[name.rsplit("_lag_", 1)[0]]
    assert len(set(target_slot.values())) == N_PLACES


def test_forecast_groups_name_their_places():
    # The forecasting path gives the sampler each group's target places,
    # in target order, as the slots need them.
    from bdbvhubrenewal.forecasters import foundation

    tri = foundation.HubTriangle(
        times=["a", "b", "c", "d"],
        final=np.ones((2, 4)),
        partial=np.ones((2, 4, 2)),
        last=3,
    )
    group = foundation.layout_block(tri, "ragged", ["CD", "CD-KN"], 2)
    assert group["places"] == ["CD", "CD-KN"]


def test_slots_are_made_on_the_models_device():
    model = tiny_model().to("meta")
    slots.attach(model, 4)
    for p in getattr(model, slots.SLOT_MODULE).parameters():
        assert p.device.type == "meta"


def save_with_settings(model, where, parent: bool) -> None:
    """Save an adapter with the settings.json a checkpoint has beside it."""
    model.save_pretrained(where)
    slot = {
        "n": 8,
        **({"parent": {"link": "own plus parent"}} if parent else {}),
    }
    (where / "settings.json").write_text(json.dumps({"series_slots": slot}))


def linked_adapter(tmp_path):
    """Save an adapter whose slots carry the parent link; return it and ids."""
    from peft import get_peft_model

    model = tiny_model()
    slots.attach(model, 8, init_std=0.02, seed=1)
    peft_model = get_peft_model(model, lora()).eval()
    save_with_settings(peft_model, tmp_path / "linked", parent=True)
    ids = slots.slot_ids(
        N_PLACES,
        N_PLACES * LAG,
        np.random.default_rng(slots.EVAL_SEED),
        8,
        parents=slots.parents_of(["CD", "CD-NK", "CD-NK-01"]),
    )
    return peft_model, ids


def test_a_parent_linked_adapter_forecasts_with_its_parents(tmp_path):
    model, ids = linked_adapter(tmp_path)
    assert ids.shape == (2, N_PLACES * (1 + LAG))
    block = toy_block()
    want = predict(model, block, ids)
    loaded = slots.load_adapter(tiny_model(), tmp_path / "linked")
    assert slots.has_parent_link(loaded)
    assert not any("parent_scale" in n for n, _ in loaded.named_parameters())
    assert torch.allclose(predict(loaded, block, ids), want, atol=1e-5)
    # Own plus parent exactly: the parent term changes the forecast.
    assert not torch.allclose(predict(loaded, block, ids[0]), want, atol=1e-5)
    module = getattr(loaded, slots.SLOT_MODULE)
    table = module.embedding.weight.detach()
    out = module(torch.tensor([[3, 4], [5, -1]])).detach()
    assert torch.allclose(out[0], table[3] + table[5])
    assert torch.allclose(out[1], table[4])


def test_inference_gives_a_linked_model_its_parents(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from chronos import Chronos2Pipeline

    from bdbvhubrenewal.forecasters import foundation

    linked_adapter(tmp_path)
    loaded = slots.load_adapter(tiny_model(), tmp_path / "linked")
    monkeypatch.setattr(
        foundation,
        "_chronos",
        lambda _checkpoint=None: Chronos2Pipeline(model=loaded),
    )
    seen = []
    real = slots.assigned

    @contextmanager
    def spy(m, ids):
        seen.append(np.asarray(ids))
        with real(m, ids):
            yield

    monkeypatch.setattr(slots, "assigned", spy)
    places = ["CD", "CD-NK", "CD-NK-01"]
    x = toy_block()["context"]
    names = [f"{p}_lag_{d}" for p in places for d in range(LAG)]
    rows = dict(zip(names, x[N_PLACES:], strict=True))
    group = {
        "target": x[:N_PLACES, :T],
        "past_covariates": {k: v[:T] for k, v in rows.items()},
        "future_covariates": {k: v[T:] for k, v in rows.items()},
        "places": places,
    }
    foundation.chronos_sampler([group], H, 5, 1)
    own, parent = seen[-1]
    # CD has no parent; CD-NK's is CD; CD-NK-01's is CD-NK.
    assert parent[0] == -1
    assert parent[1] == own[0]
    assert parent[2] == own[1]


def test_an_adapter_without_the_parent_link_still_loads(tmp_path):
    from peft import get_peft_model

    model = tiny_model()
    slots.attach(model, 8, init_std=0.02, seed=1, parent=False)
    save_with_settings(
        get_peft_model(model, lora()), tmp_path / "plain", parent=False
    )
    assert slots.adapter_parent_link(tmp_path / "plain") is False
    loaded = slots.load_adapter(tiny_model(), tmp_path / "plain")
    assert slots.has_slots(loaded)
    assert not slots.has_parent_link(loaded)
    # Parent ids are ignored by slots trained without the link.
    block = toy_block()
    ids = slots.slot_ids(
        N_PLACES,
        N_PLACES * LAG,
        np.random.default_rng(0),
        8,
        parents=[-1, 0, 1],
    )
    assert torch.equal(
        predict(loaded, block, ids), predict(loaded, block, ids[0])
    )


def test_a_slots_adapter_with_no_settings_is_an_error(tmp_path):
    linked_adapter(tmp_path)
    (tmp_path / "linked" / "settings.json").unlink()
    with pytest.raises(ValueError, match=r"settings\.json"):
        slots.load_adapter(tiny_model(), tmp_path / "linked")


def test_an_adapter_weight_the_model_drops_is_an_error(tmp_path, monkeypatch):
    linked_adapter(tmp_path)
    # A loader that builds no slots would drop the slot weights; that must
    # fail, not load silently.
    monkeypatch.setattr(slots, "adapter_slots", lambda _a: None)
    with pytest.raises(ValueError, match="has no place"):
        slots.load_adapter(tiny_model(), tmp_path / "linked")
