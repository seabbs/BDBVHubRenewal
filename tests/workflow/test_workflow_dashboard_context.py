"""A dashboard is configured from workflow config, the machine and the CLI."""

import pytest

pytest.importorskip("matplotlib")

from flowlib.dashboard import context

CONFIG = {
    "finetune": {"outbreaks": 8000, "learning_rate": 1e-5, "lr_steps": 36500},
    "dashboard": {
        "runs": [
            {"key": "realdata", "source": "realdata-ft-", "colour": "#222"}
        ],
        "references": [
            {
                "key": "rw",
                "label": "randomwalknowcast",
                "colour": "#666",
                "model": "^randomwalknowcast$",
            }
        ],
        "rule": "R",
        "mail_filter": "slots|ckpt",
        "mirror": "output/dashboard/mirror",
        "datasets": [
            {
                "key": "germany",
                "label": "Germany",
                "truth": "t",
                "groups": [{"name": "national", "match": "^DE$"}],
            }
        ],
    },
}


def test_runs_come_from_all_three_layers(tmp_path):
    local = {
        "DASHBOARD_RUNS": f"slots={tmp_path},live",
        "REMOTE": "",
        "SCORE_CACHE": str(tmp_path / "s"),
        "MAILBOX": str(tmp_path / "m"),
    }
    ctx = context.build(CONFIG, local, runs=["extra=x-ft-"], refs=[])
    assert [r.key for r in ctx.runs] == ["realdata", "slots", "extra"]
    assert next(r for r in ctx.runs if r.live).key == "slots"
    assert [r.key for r in ctx.references] == ["rw"]
    assert ctx.rule == "R"
    assert ctx.outbreaks == 8000
    assert (
        ctx.source.scores == tmp_path / "s" / "tables" / "checkpoint-scores.csv"
    )
    assert ctx.mailbox == tmp_path / "m"


def test_a_remote_machine_reads_a_mirror(tmp_path):
    local = {
        "DASHBOARD_RUNS": "",
        "REMOTE": "h",
        "REMOTE_ROOT": "r",
        "SCORE_CACHE": "/x",
        "MAILBOX": "",
    }
    ctx = context.build(CONFIG, local, runs=[], refs=[], root=tmp_path)
    assert ctx.source.root == tmp_path / "output/dashboard/mirror"
    assert ctx.mailbox is None


def test_no_runs_at_all_is_an_error():
    cfg = {**CONFIG, "dashboard": {**CONFIG["dashboard"], "runs": []}}
    local = {
        "DASHBOARD_RUNS": "",
        "REMOTE": "",
        "SCORE_CACHE": "/x",
        "MAILBOX": "",
    }
    with pytest.raises(SystemExit, match="no runs"):
        context.build(cfg, local, runs=[], refs=[])


def test_a_reference_can_take_its_paper_colour(tmp_path):
    from flowlib.dashboard import paperstyle

    cfg = {
        **CONFIG,
        "dashboard": {
            **CONFIG["dashboard"],
            "references": [
                {
                    "key": "rw",
                    "label": "randomwalknowcast",
                    "model": "^randomwalknowcast$",
                    "paper_model": "randomwalknowcast",
                }
            ],
        },
    }
    local = {
        "DASHBOARD_RUNS": f"a={tmp_path}",
        "REMOTE": "",
        "SCORE_CACHE": "/x",
        "MAILBOX": "",
    }
    ctx = context.build(cfg, local)
    assert ctx.references[0].colour == paperstyle.model_colour(
        "randomwalknowcast"
    )
