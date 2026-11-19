import numpy as np
import pytest

from bdbvhubrenewal.evaluation.labels import (
    MODEL_LABELS,
    MODEL_STYLES,
    WIS_PART_FILLS,
    WIS_PART_STACK,
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
def test_main_models_display_names(model, label) -> None:
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


@pytest.mark.parametrize(
    "model", ["epinowcast_real", "randomwalk_real", "hub_ensemble_real"]
)
def test_real_data_names_only_models_with_a_variant(model) -> None:
    # Only labels with a "(...)" variant name a real-data fine-tune.
    assert model_label(model) == model


def test_wis_parts_stack_and_differ_by_fill() -> None:
    assert WIS_PART_STACK == ("underprediction", "dispersion", "overprediction")
    assert WIS_PART_FILLS == {
        "underprediction": "hatch-down",
        "dispersion": "solid",
        "overprediction": "hatch-up",
    }
    assert tint("#2a78d6", 1.0) == "#2a78d6"
    assert tint("#2a78d6", 0.0) == "#ffffff"


# Colour checks: CIEDE2000 between the shared colours under normal vision
# and simulated colour-blindness (Machado et al. 2009, severity 1), and
# WCAG contrast against the dashboard's light and dark backgrounds.
_CVD = {
    "normal": np.eye(3),
    "deutan": [
        [0.367322, 0.860646, -0.227968],
        [0.280085, 0.672501, 0.047413],
        [-0.011820, 0.042940, 0.968881],
    ],
    "protan": [
        [0.152286, 1.052583, -0.204868],
        [0.114503, 0.786281, 0.099216],
        [-0.003882, -0.048116, 1.051998],
    ],
    "tritan": [
        [1.255528, -0.076749, -0.178779],
        [-0.078411, 0.930809, 0.147602],
        [0.004733, 0.691367, 0.303900],
    ],
}
_BACKGROUNDS = {"light": "#ffffff", "dark": "#14171c"}


def _linear(colour: str) -> np.ndarray:
    c = np.array([int(colour[i : i + 2], 16) for i in (1, 3, 5)]) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lab(rgb: np.ndarray) -> np.ndarray:
    xyz = np.array(
        [
            [0.4124, 0.3576, 0.1805],
            [0.2126, 0.7152, 0.0722],
            [0.0193, 0.1192, 0.9505],
        ]
    ) @ np.clip(rgb, 0, 1)
    xyz = xyz / np.array([0.95047, 1, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def _de2000(a: np.ndarray, b: np.ndarray) -> float:
    (l1, a1, b1), (l2, a2, b2) = a, b
    cb = (np.hypot(a1, b1) + np.hypot(a2, b2)) / 2
    g = 0.5 * (1 - np.sqrt(cb**7 / (cb**7 + 25**7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1, c2 = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1 = np.degrees(np.arctan2(b1, a1p)) % 360
    h2 = np.degrees(np.arctan2(b2, a2p)) % 360
    dh = 0.0 if c1 * c2 == 0 else (h2 - h1 + 180) % 360 - 180
    dl, dc = l2 - l1, c2 - c1
    big_dh = 2 * np.sqrt(c1 * c2) * np.sin(np.radians(dh / 2))
    lb, cbp = (l1 + l2) / 2, (c1 + c2) / 2
    hb = (h1 + h2) / 2 if abs(h1 - h2) <= 180 else (h1 + h2 + 360) / 2
    t = (
        1
        - 0.17 * np.cos(np.radians(hb - 30))
        + 0.24 * np.cos(np.radians(2 * hb))
        + 0.32 * np.cos(np.radians(3 * hb + 6))
        - 0.20 * np.cos(np.radians(4 * hb - 63))
    )
    sl = 1 + 0.015 * (lb - 50) ** 2 / np.sqrt(20 + (lb - 50) ** 2)
    sc, sh = 1 + 0.045 * cbp, 1 + 0.015 * cbp * t
    rot = 30 * np.exp(-(((hb - 275) / 25) ** 2))
    rt = -np.sin(np.radians(2 * rot)) * 2 * np.sqrt(cbp**7 / (cbp**7 + 25**7))
    return float(
        np.sqrt(
            (dl / sl) ** 2
            + (dc / sc) ** 2
            + (big_dh / sh) ** 2
            + rt * (dc / sc) * (big_dh / sh)
        )
    )


def _contrast(a: str, b: str) -> float:
    lum = sorted(
        float(np.array([0.2126, 0.7152, 0.0722]) @ _linear(c)) for c in (a, b)
    )
    return (lum[1] + 0.05) / (lum[0] + 0.05)


def test_zero_shot_is_green_and_complete_data_zero_shot_purple() -> None:
    # Green as the old dashboard's zero-shot, not pink.
    def hue(colour: str) -> float:
        _, a, b = _lab(_linear(colour))
        return float(np.degrees(np.arctan2(b, a)) % 360)

    assert 120 < hue(MODEL_STYLES["fm_chronos_daily_ragged"][0]) < 160
    assert 290 < hue(MODEL_STYLES["fm_chronos_daily_complete"][0]) < 330


@pytest.mark.parametrize("vision", sorted(_CVD))
def test_shared_colours_are_distinct_with_colour_blindness(vision) -> None:
    # Models drawn alike share a colour on purpose: the hub baselines (one
    # per study) and the weekly variants (told apart by dash).
    colours = sorted({c for c, _ in MODEL_STYLES.values()})
    lab = {c: _lab(np.array(_CVD[vision]) @ _linear(c)) for c in colours}
    worst = min(
        (_de2000(lab[a], lab[b]), a, b)
        for i, a in enumerate(colours)
        for b in colours[i + 1 :]
    )
    assert worst[0] >= 5, worst


@pytest.mark.parametrize("theme", sorted(_BACKGROUNDS))
def test_main_models_stand_out_on_light_and_dark(theme) -> None:
    for model in MAIN:
        colour = MODEL_STYLES[model][0]
        assert _contrast(colour, _BACKGROUNDS[theme]) >= 3, (model, colour)


def _oklab(rgb: np.ndarray) -> np.ndarray:
    m1 = np.array(
        [
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ]
    )
    m2 = np.array(
        [
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660],
        ]
    )
    return m2 @ np.cbrt(m1 @ np.clip(rgb, 0, 1))


def test_zero_shots_meet_the_palette_validator_against_the_main_models():
    # Zero-shot (green) and Zero-shot complete data (purple) are
    # clearly distinct from Fine-tuned (delays) and Random walk nowcast:
    # the dataviz validator's thresholds, OKLab Delta E x 100 of at least 8
    # under simulated deuteranopia and protanopia and 15 under normal
    # vision, for every pair of the four.
    models = [
        "fm_chronos_daily_ragged_ft",
        "randomwalknowcast",
        "fm_chronos_daily_ragged",
        "fm_chronos_daily_complete",
    ]
    colours = [MODEL_STYLES[m][0] for m in models]
    for vision, floor in (("normal", 15), ("deutan", 8), ("protan", 8)):
        lab = [_oklab(np.array(_CVD[vision]) @ _linear(c)) for c in colours]
        worst = min(
            100 * float(np.linalg.norm(lab[i] - lab[j]))
            for i in range(4)
            for j in range(i + 1, 4)
        )
        assert worst >= floor, (vision, worst)
    # Line styles differ too: complete data dashed, delays dotted.
    dashes = [MODEL_STYLES[m][1] for m in models[::2]] + [
        MODEL_STYLES["fm_chronos_daily_complete"][1]
    ]
    assert dashes == ["solid", "dotted", "dashed"]


def test_dashboard_colour_maps_are_shared_and_distinct() -> None:
    import sys
    from pathlib import Path

    from bdbvhubrenewal.evaluation import labels

    # Every relation a run may train; layouts as the trainer's r6 design.
    assert list(labels.RELATION_COLOURS) == ["self", "same", "up", "down"]
    sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
    import finetune_chronos as ft

    assert set(labels.LAYOUT_COLOURS) == set(ft.DESIGNS["r6"]["source_shares"])
    for colours in (
        labels.RELATION_COLOURS,
        labels.LAYOUT_COLOURS,
        labels.LEVEL_COLOURS,
    ):
        assert len(set(colours.values())) == len(colours)
        assert set(colours.values()) <= {*labels.CATEGORICAL, labels.NEUTRAL}
    # Earlier runs' shades are the fine-tuned model's blues, apart from it.
    ft_blue = labels.MODEL_STYLES["fm_chronos_daily_ragged_ft"][0]
    assert ft_blue not in labels.RUN_SHADES
