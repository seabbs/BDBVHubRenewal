"""The slots check on an attention-bias checkpoint: swaps through the ids.

The lag swap gives a place's lag rows another place's id; the parent swap
gives places a wrong parent. The content stays as given, so any change in
the output is the bias reading the ids.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("chronos")
pytest.importorskip("peft")

sys.path.insert(0, str(Path(__file__).parent / "forecasters"))
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import covariate_ablation as ca  # noqa: E402
import slots_check as sc  # noqa: E402
from chronos_toy import LAG, N_PLACES, H, T, tiny_model, toy_block  # noqa: E402

from bdbvhubrenewal.forecasters import series_bias  # noqa: E402

PLACES = ("CD", "CD-NK", "CD-NK-01")
"""A country, a province and its zone."""


@pytest.fixture(autouse=True)
def short_context(monkeypatch):
    monkeypatch.setattr(ca.ft, "CONTEXT", T)


def block() -> dict:
    """Places that differ in shape, so whose lags sit where matters."""
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
        "places": PLACES,
    }


def biased(scale: float = 0.5):
    """A tiny model with nonzero biases of every kind."""
    model = tiny_model()
    series_bias.attach(model)
    module = getattr(model, series_bias.BIAS_MODULE)
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(scale * torch.randn(p.shape, generator=gen))
    return model


def test_bias_ids_follow_places_and_parents():
    rows = sc.model_ids(biased(), block())
    place, parent = rows
    assert list(place[:N_PLACES]) == [0, 1, 2]
    assert list(parent[:N_PLACES]) == [-1, 0, 1]
    lags = series_bias.row_places(N_PLACES, N_PLACES * LAG)
    assert np.array_equal(place, lags)


def test_wrong_place_moves_only_lag_rows_to_other_places():
    right = sc.model_ids(biased(), block())
    wrong = sc.model_ids(
        biased(), block(), wrong="wrong_place", rng=np.random.default_rng(0)
    )
    # The targets keep their place and parent.
    assert np.array_equal(wrong[:, :N_PLACES], right[:, :N_PLACES])
    for r in range(N_PLACES, right.shape[1]):
        assert wrong[0, r] != right[0, r]
        # A moved row carries its new place's parent.
        assert wrong[1, r] == right[1, wrong[0, r]]


def test_wrong_parent_moves_only_parents():
    right = sc.model_ids(biased(), block())
    wrong = sc.model_ids(
        biased(), block(), wrong="wrong_parent", rng=np.random.default_rng(0)
    )
    assert np.array_equal(wrong[0], right[0])
    assert wrong[1, 0] == -1
    # The zone's parent becomes the country, the province's the zone.
    assert wrong[1, 2] == 0
    assert wrong[1, 1] == 2


@pytest.mark.parametrize("cond", ["wrong_place", "wrong_parent"])
def test_wrong_ids_change_a_bias_models_output(cond):
    model = biased()
    b = block()
    _, given = sc.predict(model, b)
    _, moved = sc.predict(model, b, wrong=cond, rng=np.random.default_rng(0))
    assert sc.output_change(given, moved, LAG, H)["nowcast_max"] > 1e-4


def test_zero_bias_ignores_wrong_ids():
    model = tiny_model()
    series_bias.attach(model)
    b = block()
    _, given = sc.predict(model, b)
    for cond in ("wrong_place", "wrong_parent"):
        _, moved = sc.predict(
            model, b, wrong=cond, rng=np.random.default_rng(0)
        )
        assert torch.equal(given, moved)


def test_id_swaps_are_scored_for_the_bias_model_only():
    assert {"wrong_place", "wrong_parent"} <= set(sc.CONDITIONS)
    for phase in ("headline", "rest", "all"):
        work = sc.phase_work(phase, True)
        for m, c in work:
            if c in sc.ID_SWAPS:
                assert m == "slots"
    assert ("slots", "wrong_place") in sc.phase_work("headline", True)
    assert ("slots", "wrong_parent") in sc.phase_work("headline", True)


def test_a_bias_checkpoint_is_not_held(tmp_path):
    import json

    (tmp_path / "hold").write_text("")
    (tmp_path / "settings.json").write_text(
        json.dumps({"series_bias": {"init": "zero"}})
    )
    assert not sc.held(tmp_path, hold=tmp_path / "hold")


def test_headline_reports_the_id_swaps():
    import pandas as pd

    row = {
        "model": "slots",
        "window": "nowcast",
        "ratio": 1.2,
        "ci_low": 1.1,
        "ci_high": 1.3,
        "n_blocks": 40,
        "n_above_floor": 30,
        "n_unchanged": 0,
        "slot_sd": np.nan,
        "against": "given",
        "loss": 0.1,
    }
    s = pd.DataFrame(
        [
            {**row, "set": "sims multi-place", "condition": "swap_places"},
            {**row, "set": "sims multi-place", "condition": "wrong_place"},
            {**row, "set": "germany states", "condition": "wrong_parent"},
        ]
    )
    text = sc.headline(s)
    assert "sims multi-place nowcast wrong_place/given: slots x1.20" in text
    assert "germany states nowcast wrong_parent/given" in text
