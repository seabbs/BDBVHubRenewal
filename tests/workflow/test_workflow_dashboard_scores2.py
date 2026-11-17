"""Score panels: target, metric and layout per row; studies from config."""

from flowlib.dashboard import runs, scores

RULE = "R"


def _row(
    study,
    model,
    ck,
    n="",
    target="",
    wis=None,
    crps=None,
    level="national",
    layout="",
):
    return {
        "rule": RULE,
        "study": study,
        "level": level,
        "model": model,
        "checkpoint": ck,
        "outbreaks": n,
        "target": target,
        "rel_wis": "" if wis is None else wis,
        "rel_crps": "" if crps is None else crps,
        "layout": layout,
    }


def _setup():
    rs = runs.load_runs([], "s=slots-", [])
    refs = runs.load_references(
        [
            {
                "key": "zero",
                "model": "^(?!randomwalknowcast$)",
                "label": "z",
                "checkpoint": "zero-shot",
                "role": "start",
            }
        ],
        [],
    )
    return rs, refs


VIEW = {
    "studies": {
        "germany": "Germany",
        "ebola": "BDBV",
        "held_out": "held-out synthetic",
    },
    "levels": ["national", "states", "age", "provinces"],
    "defaults": {
        "target": "7day",
        "metric": "wis",
        "source": "daily",
        "level": "national",
    },
}


def test_rows_carry_target_and_metric_and_old_rows_are_7day():
    rs, refs = _setup()
    rows = [
        _row(
            "germany",
            "fm_chronos_daily_ragged_ft",
            "slots-a1b2c3",
            "250",
            target="daily",
            wis="0.9",
        ),
        _row(
            "ebola",
            "fm_chronos_weekly_ragged_ft",
            "slots-a1b2c3",
            "250",
            crps="1.1",
        ),
    ]
    got = scores.payload(rows, rs, refs, RULE, local=[], view=VIEW)
    assert [
        "germany",
        "national",
        "daily",
        "s",
        250,
        "a1b2c3",
        0.9,
        "daily",
        "wis",
    ] in got["rows"]
    assert [
        "ebola",
        "national",
        "weekly",
        "s",
        250,
        "a1b2c3",
        1.1,
        "7day",
        "crps",
    ] in got["rows"]
    assert got["targets"] == {"germany": ["daily"], "ebola": ["7day"]}
    assert got["metrics"] == {"germany": ["wis"], "ebola": ["crps"]}


def test_a_row_with_both_scores_gives_two_rows():
    rs, refs = _setup()
    rows = [
        _row(
            "ebola",
            "fm_chronos_daily_ragged_ft",
            "slots-a1b2c3",
            "1",
            wis="0.8",
            crps="0.7",
        )
    ]
    got = scores.payload(rows, rs, refs, RULE, local=[], view=VIEW)
    assert {r[8] for r in got["rows"]} == {"wis", "crps"}


def test_held_out_rows_are_by_layout():
    rs, refs = _setup()
    rows = [
        _row(
            "held_out",
            "fm_chronos_daily_ragged_ft",
            "slots-a1b2c3",
            "250",
            wis="0.7",
            level="all",
            layout="3day",
        )
    ]
    got = scores.payload(rows, rs, refs, RULE, local=[], view=VIEW)
    assert got["rows"][0][2] == "3day"
    assert got["vlabel"]["held_out"] == "layout"
    assert got["vbase"]["3day"] == "daily"
    assert got["variants"]["held_out"] == ["3day"]


def test_studies_and_levels_are_ordered_and_labelled_by_config():
    rs, refs = _setup()
    rows = [
        _row(
            st,
            "fm_chronos_daily_ragged_ft",
            "slots-a1b2c3",
            "1",
            wis="1",
            level=lv,
        )
        for st, lv in [
            ("held_out", "all"),
            ("ebola", "provinces"),
            ("germany", "age"),
            ("germany", "national"),
        ]
    ]
    got = scores.payload(rows, rs, refs, RULE, local=[], view=VIEW)
    assert got["studies"] == ["germany", "ebola", "held_out"]
    assert got["studylab"]["held_out"] == "held-out synthetic"
    assert got["levels"]["germany"] == ["national", "age"]
    assert got["view_defaults"] == VIEW["defaults"]


def test_excluded_checkpoints_are_left_out():
    rs, refs = _setup()
    rows = [
        _row(
            "germany",
            "fm_chronos_daily_ragged_ft",
            "slots-a1b2c3",
            "1",
            wis="1",
        )
    ]
    got = scores.payload(
        rows, rs, refs, RULE, local=[], view=VIEW, excluded={"a1b2c3"}
    )
    assert got["rows"] == []
