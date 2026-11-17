"""Dashboard runs and references come from configuration."""

import json

import pytest
from flowlib.dashboard import runs


def test_a_run_spec_with_a_folder(tmp_path):
    r = runs.parse_run(f"slots={tmp_path},label=slots run,colour=#2563eb,live")
    assert r.key == "slots"
    assert r.folder == tmp_path
    assert r.prefix is None
    assert (r.label, r.colour, r.live) == ("slots run", "#2563eb", True)


def test_a_run_spec_with_a_release_prefix():
    r = runs.parse_run("realdata=realdata-ft-")
    assert r.prefix == "realdata-ft-"
    assert r.folder is None
    assert r.label == "realdata"


def test_hashes_and_sizes_are_data():
    r = runs.parse_run(
        "noslots=chronos-ft-,hashes=de129103cacc+7962fd718a47,"
        "n=de129103cacc:1000+7962fd718a47:2000,base=slots"
    )
    assert r.hashes == ("de129103cacc", "7962fd718a47")
    assert r.n == {"de129103cacc": 1000, "7962fd718a47": 2000}
    assert r.base == "slots"


def test_a_bad_spec_says_what_is_wrong():
    with pytest.raises(ValueError, match="name=SOURCE"):
        runs.parse_run("no-equals")
    with pytest.raises(ValueError, match="unknown"):
        runs.parse_run("a=b-,shade=3")


def test_runs_join_config_then_machine_then_command_line(tmp_path):
    config = [{"key": "realdata", "source": "realdata-ft-", "colour": "#111"}]
    env = f"slots={tmp_path},live;realdata=realdata-ft-,colour=#222"
    cli = ["noslots=chronos-ft-,hashes=abc123"]
    got = runs.load_runs(config, env, cli)
    assert [r.key for r in got] == ["realdata", "slots", "noslots"]
    # A later layer replaces a run of the same name.
    assert got[0].colour == "#222"
    assert sum(r.live for r in got) == 1


def test_one_run_is_live_by_default(tmp_path):
    got = runs.load_runs([], f"a={tmp_path};b=x-", [])
    assert [r.live for r in got] == [True, False]


def test_a_checkpoint_label_maps_to_its_run(tmp_path):
    train = tmp_path / "train" / "n250"
    train.mkdir(parents=True)
    (train / "settings.json").write_text(json.dumps({"hash": "7d7c5d9f247e"}))
    got = runs.load_runs(
        [],
        f"slots={tmp_path},prefix=slots-;"
        "zeroinit=slots-zeroinit-,hashes=06dbdcc2b48d;"
        "noslots=chronos-ft-,hashes=de129103cacc;realdata=realdata-ft-",
        [],
    )
    which = runs.Classifier(got)
    assert which.run_of("slots-7d7c5d9f247e") == "slots"
    assert which.run_of("chronos-slots-7d7c5d9f247e") == "slots"
    assert which.run_of("06dbdcc2b48d") == "zeroinit"
    assert which.run_of("de129103cacc") == "noslots"
    assert which.run_of("realdata-ft-0a1b2c3d4e5f") == "realdata"
    assert which.run_of("ffffffffffff") is None


def test_references_match_rows():
    refs = runs.load_references(
        [
            {
                "key": "zero",
                "label": "zero-shot ragged",
                "colour": "#16a34a",
                "model": "^(?!randomwalknowcast$)(?!.*_complete$)",
                "checkpoint": "zero-shot",
            },
            {
                "key": "rw",
                "label": "randomwalknowcast",
                "colour": "#6b7280",
                "model": "^randomwalknowcast$",
            },
        ],
        ["complete=_complete$,checkpoint=zero-shot,colour=#9333ea"],
    )
    assert [r.key for r in refs] == ["zero", "rw", "complete"]
    which = runs.Classifier([], refs)
    row = {"model": "fm_chronos_daily_ragged_ft", "checkpoint": "zero-shot"}
    assert which.series(row) == "zero"
    row = {"model": "fm_chronos_daily_complete", "checkpoint": "zero-shot"}
    assert which.series(row) == "complete"
    row = {"model": "randomwalknowcast", "checkpoint": "abc"}
    assert which.series(row) == "rw"
