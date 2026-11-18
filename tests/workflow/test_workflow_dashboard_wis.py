"""Mean WIS split into its components, from the shared score table."""

from flowlib.dashboard import paperstyle, runs, scores, wis

RULE = "own-quantile-wis_crps-1000-common"


def _row(model, checkpoint, n="", parts=(0.5, 0.2, 0.3), **kw):
    disp, under, over = parts
    row = {
        "rule": RULE,
        "study": "germany",
        "level": "national",
        "model": model,
        "checkpoint": checkpoint,
        "outbreaks": n,
        "wis": f"{disp + under + over}",
        "dispersion": f"{disp}",
        "underprediction": f"{under}",
        "overprediction": f"{over}",
    }
    return row | kw


def _which():
    rs = runs.load_runs([], "slots=slots-,label=live,colour=#111", [])
    refs = runs.load_references(
        [
            {
                "key": "rw",
                "label": "randomwalknowcast",
                "colour": "#666",
                "model": "^randomwalknowcast$",
            }
        ],
        [],
    )
    return rs, refs


ROWS = [
    _row("randomwalknowcast", "zero-shot"),
    _row("fm_chronos_daily_ragged_ft", "slots-bbbbbb", n="500"),
    _row("fm_chronos_daily_ragged_ft", "slots-aaaaaa", n="250"),
    # Another source, another rule, the held-out layouts, another target
    # and a row scored before the components: none is drawn.
    _row("fm_chronos_weekly_ragged_ft", "slots-cccccc", n="250"),
    _row("fm_chronos_daily_ragged_ft", "slots-aaaaaa", n="250", rule="x"),
    _row("fm_chronos_daily_ragged_ft", "slots-aaaaaa", n="250", layout="l"),
    _row("fm_chronos_daily_ragged_ft", "slots-dddddd", n="750", target="t"),
    _row("fm_chronos_daily_ragged_ft", "slots-eeeeee", n="1000")
    | {"dispersion": ""},
]


def test_bars_are_checkpoints_by_size_then_references():
    rs, refs = _which()
    got = scores.wis_bars(ROWS, rs, refs, RULE, source="daily")
    bars = got[("germany", "national")]
    assert [b["label"] for b in bars] == [
        "live n250",
        "live n500",
        "randomwalknowcast",
    ]
    assert bars[0]["colour"] == "#111"
    assert bars[0]["wis"] == 1.0
    assert bars[0]["parts"] == {
        "dispersion": 0.5,
        "underprediction": 0.2,
        "overprediction": 0.3,
    }


def test_no_bars_without_components():
    rs, refs = _which()
    old = [{k: v for k, v in r.items() if k not in wis.PARTS} for r in ROWS]
    assert scores.wis_bars(old, rs, refs, RULE, source="daily") == {}
    assert wis.section({}, {}) == ""


def test_the_section_draws_each_part_in_the_paper_s_colour():
    rs, refs = _which()
    bars = scores.wis_bars(ROWS, rs, refs, RULE, source="daily")
    view = {"studies": {"germany": "Germany"}, "levels": ["national"]}
    out = wis.section(bars, view)
    assert "Germany, national" in out
    # Each part a shade of its model's colour, as the paper's bars.
    assert wis.PARTS == ("dispersion", "overprediction", "underprediction")
    for part in wis.PARTS:
        assert paperstyle.part_colour("#111", part) in out
    assert paperstyle.part_colour("#111", "dispersion") == "#111111"
    assert paperstyle.part_colour("#000000", "underprediction") == "#b3b3b3"
    # Stacked in the paper's order.
    first = out.index("<rect", out.index("live n250"))
    assert out.index("dispersion", first) < out.index("overprediction", first)
    assert out.index("overprediction", first) < out.index(
        "underprediction", first
    )
    assert out.count("class=card") == 1


def test_a_bar_scored_on_other_forecasts_is_flagged():
    rs, refs = _which()
    rows = [r | {"n": "120"} for r in ROWS[:3]]
    rows[1] |= {"n": "90"}  # n500, from a resumed or partial unit
    bars = scores.wis_bars(rows, rs, refs, RULE, source="daily")
    assert [b["cells"] for b in bars[("germany", "national")]] == [
        120,
        90,
        120,
    ]
    out = wis.section(bars, {})
    assert "live n500 *" in out
    assert "live n250 *" not in out
    assert "other forecasts" in out
