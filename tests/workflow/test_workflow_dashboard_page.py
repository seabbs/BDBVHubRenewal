"""The dashboard page, built from a fixture machine."""

import csv
import gzip
import json

import pytest

pytest.importorskip("matplotlib")

from flowlib.dashboard import mirror, page, paperstyle, runs

LINE = "2026-11-16 16:{m:02d}:00 [train] step {s}: loss {l}, held-out (x) {h}"


def _machine(tmp_path):
    run = tmp_path / "run"
    (run / "train" / "n250" / "finetuned-ckpt").mkdir(parents=True)
    (run / "train" / "n250" / "settings.json").write_text(
        json.dumps(
            {
                "hash": "aaaaaa",
                "n_outbreaks": 250,
                "steps": 1140,
                "held_out_loss": 0.41,
                "held_out_loss_start": 0.47,
                "held_out_loss_by_source": {"weekly": 0.2, "daily": 0.5},
            }
        )
    )
    (run / "train" / "history.json").write_text(
        json.dumps(
            [{"step": 0, "held_out_loss": 0.47, "by_source": {"weekly": 0.25}}]
        )
    )
    (run / "train.log").write_text(
        "\n".join(
            LINE.format(m=i, s=10 * i, l=0.5 - i / 100, h=0.45)
            for i in range(30)
        )
        + "\n"
    )
    scores_dir = tmp_path / "scores" / "tables"
    scores_dir.mkdir(parents=True)
    with (scores_dir / "checkpoint-scores.csv").open("w") as f:
        w = csv.DictWriter(
            f,
            [
                "rule",
                "study",
                "level",
                "model",
                "checkpoint",
                "outbreaks",
                "rel_wis",
                "rel_crps",
                "loss",
                "loss_zero_shot",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "rule": "R",
                "study": "germany",
                "level": "national",
                "model": "fm_chronos_daily_ragged_ft",
                "checkpoint": "slots-aaaaaa",
                "outbreaks": 250,
                "rel_wis": 0.9,
            }
        )
        w.writerow(
            {
                "rule": "R",
                "study": "held_out",
                "level": "all",
                "model": "",
                "checkpoint": "slots-aaaaaa",
                "outbreaks": 250,
                "loss": 0.41,
                "loss_zero_shot": 0.47,
            }
        )
    ger = tmp_path / "scores" / "germany" / "slots-aaaaaa"
    ger.mkdir(parents=True)
    with gzip.open(ger / "2022-02-09.csv.gz", "wt") as f:
        w = csv.DictWriter(
            f,
            [
                "model",
                "location",
                "target_end_date",
                "quantile",
                "value",
                "truth",
                "truth_vintage",
            ],
        )
        w.writeheader()
        for i, q in enumerate(
            ("0.025", "0.1", "0.25", "0.5", "0.75", "0.9", "0.975")
        ):
            w.writerow(
                {
                    "model": "fm_chronos_daily_ragged_ft",
                    "location": "DE",
                    "target_end_date": "2022-02-01",
                    "quantile": q,
                    "value": 8 + i,
                    "truth": 10,
                    "truth_vintage": "2022-03-01",
                }
            )
    box = tmp_path / "box.md"
    box.write_text("2026-11-16 16:00 x: slots n250 published\nother\n")
    rs = runs.load_runs(
        [], f"mine={run},label=my run,prefix=slots-,colour=#123456", []
    )
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
    return page.Context(
        runs=rs,
        references=refs,
        source=mirror.Source(
            {"REMOTE": "", "SCORE_CACHE": str(tmp_path / "scores")},
            tmp_path / "m",
        ),
        rule="R",
        outbreaks=8000,
        learning_rate=1e-5,
        lr_steps=36500,
        style=paperstyle.page_style(),
        mailbox=box,
        mail_filter="slots",
        once=True,
    )


def test_every_section_is_built_for_the_configured_runs(tmp_path):
    out = page.build(_machine(tmp_path))
    for h in (
        "Status",
        "Loss plots",
        "Held-out loss by checkpoint",
        "Performance on real data",
        "German nowcasts by checkpoint",
        "Recent activity",
    ):
        assert f"<h2>{h}" in out
    assert "Chronos-2 fine-tuning: my run" in out
    assert "#123456" in out
    assert (
        "n250 of 8000" not in out
    )  # the live step is 290, n is steps / 4.5625
    assert "of 8000 simulations trained" in out
    assert "slots n250 published" in out
    assert "other" not in out.split("Recent activity")[1]
    assert "id=gdata" in out
    assert '"levels":[0.025,0.25,0.5,0.75,0.975]' in out
    # No hard-coded runs or hosts.
    assert "no slots" not in out
    assert "archie" not in out


def test_a_static_page_does_not_reload(tmp_path):
    ctx = _machine(tmp_path)
    assert "location.reload" not in page.build(ctx)
    ctx.once = False
    assert "location.reload" in page.build(ctx) or "reload" in page.build(ctx)
