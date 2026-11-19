"""Run settings come from an uncommitted local file, merged over config."""

import json

import pytest

pytest.importorskip("matplotlib")

from flowlib.dashboard import context, runs

BASE = {
    "finetune": {
        "outbreaks": 8000,
        "learning_rate": 1e-5,
        "prefixes": [250, 8000],
    },
    "dashboard": {
        "runs": [],
        "references": [],
        "rule": "R",
        "mirror": "output/dashboard/mirror",
        "exclude": {"hashes": ["aaaaaa"]},
    },
}


def _local(tmp_path, text):
    path = tmp_path / "dashboard.local.yaml"
    path.write_text(text)
    return {
        "DASHBOARD_CONFIG": str(path),
        "REMOTE": "",
        "SCORE_CACHE": str(tmp_path / "s"),
        "MAILBOX": "",
    }


def test_the_local_file_adds_runs_and_replaces_settings(tmp_path):
    local = _local(
        tmp_path,
        f"""
mirror: {tmp_path}/cache
runs:
  - key: bias
    label: attention bias
    source: {tmp_path}/run
    hue: 217
    live: true
""",
    )
    ctx = context.build(BASE, local, root=tmp_path)
    assert [r.key for r in ctx.runs] == ["bias"]
    # The live run shows the shared display name, not its label.
    assert ctx.runs[0].label == "Fine-tuned (delays)"
    assert ctx.runs[0].hue == 217
    assert ctx.source.root == tmp_path / "cache"


def test_a_missing_local_file_is_no_error(tmp_path):
    local = _local(tmp_path, "runs: []\n")
    local["DASHBOARD_CONFIG"] = str(tmp_path / "none.yaml")
    local["DASHBOARD_RUNS"] = f"a={tmp_path}"
    assert [r.key for r in context.build(BASE, local).runs] == ["a"]


def test_the_default_local_file_is_found_under_config(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "dashboard.local.yaml").write_text(
        f"runs: [{{key: x, source: {tmp_path}}}]\n"
    )
    local = {"REMOTE": "", "SCORE_CACHE": "/s", "MAILBOX": ""}
    assert context.build(BASE, local, root=tmp_path).runs[0].key == "x"


def test_excluded_checkpoints_belong_to_no_run(tmp_path):
    hidden = tmp_path / "labels" / "train" / "n250"
    hidden.mkdir(parents=True)
    (hidden / "settings.json").write_text(json.dumps({"hash": "bbbbbb"}))
    local = _local(
        tmp_path,
        f"""
runs: [{{key: s, source: slots-}}]
exclude: {{hashes: [cccccc], dirs: [{tmp_path}/labels]}}
""",
    )
    ctx = context.build(BASE, local, root=tmp_path)
    which = runs.Classifier(ctx.runs, ctx.references, ctx.excluded)
    assert ctx.excluded == {"aaaaaa", "bbbbbb", "cccccc"}
    assert which.run_of("slots-aaaaaa") is None
    assert which.run_of("slots-bbbbbb") is None
    assert which.run_of("slots-dddddd") == "s"


def test_a_hue_shades_a_run_s_checkpoints_by_size():
    r = runs.make_run("a", "a-", hue=217)
    assert runs.shade(r, 0, 1) == "hsl(217,80%,46%)"
    assert runs.shade(r, 0, 3) == "hsl(217,80%,66%)"
    assert runs.shade(r, 2, 3) == "hsl(217,80%,38%)"
    plain = runs.make_run("b", "b-", colour="#123456")
    assert runs.shade(plain, 1, 3) == "#123456"


def test_the_committed_example_is_valid(tmp_path):
    from pathlib import Path

    import yaml

    example = Path(__file__).parents[2] / "config" / "dashboard.example.yaml"
    data = yaml.safe_load(example.read_text())
    assert {"runs", "exclude", "datasets", "mirror"} <= set(data)
    local = {
        "DASHBOARD_CONFIG": str(example),
        "REMOTE": "",
        "SCORE_CACHE": "/s",
        "MAILBOX": "",
    }
    ctx = context.build(BASE, local, root=tmp_path)
    assert sum(r.live for r in ctx.runs) == 1


def test_local_datasets_merge_into_the_committed_ones_by_key(tmp_path):
    base = {
        **BASE,
        "dashboard": {
            **BASE["dashboard"],
            "datasets": [
                {"key": "germany", "label": "Germany", "format": "quantiles"},
                {
                    "key": "bdbv",
                    "label": "BDBV",
                    "format": "samples",
                    "truth_source": "data/target.csv",
                },
            ],
        },
    }
    local = _local(
        tmp_path,
        """
runs: [{key: s, source: slots-}]
datasets:
  - {key: bdbv, source: "h:cache/bdbv/", truth_source: "h:repo/target.csv"}
""",
    )
    ctx = context.build(base, local, root=tmp_path)
    by_key = {d["key"]: d for d in ctx.datasets}
    assert list(by_key) == ["germany", "bdbv"]
    assert by_key["bdbv"]["format"] == "samples"
    assert by_key["bdbv"]["source"] == "h:cache/bdbv/"
    assert ctx.source.truth("bdbv") == tmp_path / "output/dashboard/mirror" / (
        "bdbv-truth.csv"
    )


def test_bdbv_is_held_out_of_every_panel_even_from_the_local_file(tmp_path):
    # BDBV (Ebola) is entirely held out for the paper; no panel,
    # selector, table or mailbox line shows it.
    from pathlib import Path

    import yaml
    from flowlib.dashboard import page

    shipped = yaml.safe_load(
        (Path(__file__).parents[2] / "workflow" / "config.yaml").read_text()
    )["dashboard"]
    assert [d["key"] for d in shipped["datasets"]] == ["germany", "held_out"]
    assert "ebola" not in shipped["studies"]
    box = tmp_path / "box.md"
    box.write_text(
        "slots BDBV unit scored, CRPS 1.2\n"
        "slots ebola rows\n"
        "slots [hostname bdbv-scoring] German unit 286 s\n"
    )
    cfg = {
        **BASE,
        "dashboard": {
            **BASE["dashboard"],
            "studies": {"germany": "Germany", "ebola": "BDBV"},
            "held_out": shipped["held_out"],
            "mail_filter": "slots",
        },
    }
    local = _local(
        tmp_path,
        f"""
runs:
  - key: r
    source: {tmp_path}/run
datasets:
  - key: bdbv
    label: BDBV (Ebola)
    format: samples
  - key: germany
    label: Germany
    format: quantiles
""",
    )
    local["MAILBOX"] = str(box)
    ctx = context.build(cfg, local, root=tmp_path)
    assert [d["key"] for d in ctx.datasets] == ["germany"]
    assert "ebola" not in ctx.view["studies"]
    assert ctx.hidden_studies == {"ebola"}
    mail = page.mail_section(ctx)
    assert "BDBV unit" not in mail
    assert "ebola rows" not in mail
    # Agent names are not BDBV lines.
    assert "bdbv-scoring" in mail
