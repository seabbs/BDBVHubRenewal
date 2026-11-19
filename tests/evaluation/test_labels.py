import pytest

from bdbvhubrenewal.evaluation.labels import (
    MODEL_LABELS,
    MODEL_STYLES,
    WIS_PART_SHADES,
    model_label,
    tint,
)

MAIN = {
    "fm_chronos_daily_ragged_ft": "Fine-tuned (delays)",
    "fm_chronos_daily_ragged": "Zero-shot (delays)",
    "fm_chronos_daily_complete": "Zero-shot (complete data only)",
    "randomwalknowcast": "Random walk nowcast",
    "randomwalk": "Random walk",
}


@pytest.mark.parametrize(("model", "label"), MAIN.items())
def test_main_models_have_sams_names(model, label) -> None:
    assert model_label(model) == label


def test_every_named_model_has_a_style_and_names_are_unique() -> None:
    assert set(MODEL_LABELS) == set(MODEL_STYLES)
    assert len(set(MODEL_LABELS.values())) == len(MODEL_LABELS)


def test_main_models_have_distinct_colours() -> None:
    # Bars have no dashes, so colour alone tells the main models apart.
    colours = [MODEL_STYLES[m][0] for m in MAIN]
    assert len(set(colours)) == len(colours)


def test_checkpoints_real_data_and_hub_teams() -> None:
    assert (
        model_label("fm_chronos_daily_ragged_ft@2000")
        == "Fine-tuned (delays), 2,000 outbreaks"
    )
    assert (
        model_label("fm_chronos_daily_ragged_real")
        == "Fine-tuned on real data (delays)"
    )
    assert model_label("OHTJHU-someteam") == "OHTJHU-someteam"


def test_wis_parts_are_shades_darkest_first() -> None:
    shades = list(WIS_PART_SHADES.values())
    assert list(WIS_PART_SHADES) == [
        "dispersion",
        "overprediction",
        "underprediction",
    ]
    assert shades == sorted(shades, reverse=True)
    assert tint("#2a78d6", 1.0) == "#2a78d6"
    assert tint("#2a78d6", 0.0) == "#ffffff"
