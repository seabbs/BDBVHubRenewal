"""Score panels read whichever runs and references are configured."""

from flowlib.dashboard import runs, scores

RULE = "own-quantile-wis_crps-1000-common"


def _row(model, checkpoint, study="germany", n="", value="0.9", rule=RULE):
    col = "rel_wis" if study == "germany" else "rel_crps"
    return {
        "rule": rule,
        "study": study,
        "level": "national",
        "model": model,
        "checkpoint": checkpoint,
        "outbreaks": n,
        col: value,
    }


def _which():
    rs = runs.load_runs(
        [], "slots=slots-,colour=#111;realdata=realdata-ft-,colour=#222", []
    )
    refs = runs.load_references(
        [
            {
                "key": "zero",
                "label": "zero-shot ragged",
                "colour": "#0a0",
                "model": "^(?!randomwalknowcast$)(?!.*_complete$)",
                "checkpoint": "zero-shot",
            },
            {
                "key": "rw",
                "label": "randomwalknowcast",
                "colour": "#666",
                "model": "^randomwalknowcast$",
            },
        ],
        [],
    )
    return rs, refs


def test_rows_go_to_their_run_or_reference():
    rs, refs = _which()
    rows = [
        _row("fm_chronos_daily_ragged_ft", "slots-aaaaaa", n="250"),
        _row(
            "fm_chronos_weekly_ragged_ft",
            "realdata-ft-bbbbbb",
            "ebola",
            n="0",
            value="1.1",
        ),
        _row("fm_chronos_daily_ragged", "zero-shot"),
        _row("randomwalknowcast", "zero-shot"),
        _row("fm_chronos_daily_ragged_ft", "cccccc"),  # no run claims it
        _row("fm_chronos_daily_ragged_ft", "slots-aaaaaa", rule="other"),
    ]
    got = scores.payload(rows, rs, refs, RULE, local=[])
    series = {(r[0], r[3]) for r in got["rows"]}
    assert series == {
        ("germany", "slots"),
        ("ebola", "realdata"),
        ("germany", "zero"),
    }
    # randomwalknowcast has no source, so it is not a series of its own.
    assert all(r[2] in ("daily", "weekly") for r in got["rows"])


def test_series_are_the_runs_then_the_references_with_their_colours():
    rs, refs = _which()
    got = scores.payload([], rs, refs, RULE, local=[])
    assert [s["key"] for s in got["series"]] == [
        "slots",
        "realdata",
        "zero",
        "rw",
    ]
    assert got["col"] == {
        "slots": "#111",
        "realdata": "#222",
        "zero": "#0a0",
        "rw": "#666",
    }


def test_checkpoints_saved_but_not_yet_scored_are_pending():
    rs, refs = _which()
    rows = [_row("fm_chronos_daily_ragged_ft", "slots-aaaaaa", n="250")]
    local = [("slots", 250, "aaaaaa"), ("slots", 500, "dddddd")]
    got = scores.payload(rows, rs, refs, RULE, local=local)
    assert got["pending"] == [{"run": "slots", "n": 500, "hash": "dddddd"}]


def test_the_first_scored_run_and_the_references_are_shown_first():
    rs, refs = _which()
    rows = [_row("fm_chronos_daily_ragged_ft", "realdata-ft-bbbbbb", n="0")]
    got = scores.payload(rows, rs, refs, RULE, local=[])
    assert got["defaults"] == ["realdata", "zero", "rw"]


def test_reference_roles_go_to_the_page():
    rs = runs.load_runs([], "a=a-", [])
    refs = runs.load_references(
        [
            {
                "key": "base",
                "label": "b",
                "colour": "#1",
                "model": "^rw$",
                "role": "baseline",
            },
            {
                "key": "zs",
                "label": "z",
                "colour": "#2",
                "model": "x",
                "checkpoint": "zero-shot",
                "role": "start",
            },
            {"key": "other", "label": "o", "colour": "#3", "model": "y"},
        ],
        ["extra=z,role=start"],
    )
    got = scores.payload([], rs, refs, RULE, local=[])
    assert got["roles"] == {
        "base": "baseline",
        "zs": "start",
        "other": "other",
        "extra": "start",
    }
