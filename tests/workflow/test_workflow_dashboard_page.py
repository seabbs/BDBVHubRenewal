"""The dashboard page, built from a fixture machine."""

import csv
import gzip
import json

import pytest

pytest.importorskip("matplotlib")

from flowlib.dashboard import mirror, page, runs

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
        mailbox=box,
        mail_filter="slots",
        once=True,
        datasets=[
            {
                "key": "germany",
                "label": "Germany",
                "format": "quantiles",
                "metric": "WIS",
                "truth": "t",
                "groups": [{"name": "national", "match": "^DE$"}],
            }
        ],
        view={
            "studies": {"germany": "Germany"},
            "levels": ["national"],
            "defaults": {"target": "7day", "source": "daily"},
        },
    )


def test_every_section_is_built_for_the_configured_runs(tmp_path):
    out = page.build(_machine(tmp_path))
    for h in (
        "Status",
        "Loss",
        "Attention bias",
        "What fine-tuning learns",
        "Nowcasts",
        "Recent activity",
        "Tables",
    ):
        assert f"<h2>{h}</h2>" in out
    assert "<b>my run run</b>" in out
    for anchor in (
        "status",
        "loss",
        "bias",
        "learning",
        "nowcasts",
        "mailbox",
        "tables",
    ):
        assert f'<a href="#{anchor}">' in out
    assert "#123456" in out
    assert (
        "n250 of 8000" not in out
    )  # the live step is 290, n is steps / 4.5625
    assert "of 8000 simulations trained" in out
    assert "slots n250 published" in out
    assert "other" not in out.split("Recent activity")[1]
    assert "id=gdata" in out
    assert '"key":"germany"' in out
    assert "id=ldata" in out
    assert "id=bdata" in out
    assert "id=seldata" in out
    assert '<option value="daily" selected>daily</option>' in out
    # No hard-coded runs or hosts.
    assert "no slots" not in out
    assert "archie" not in out


def test_the_page_never_reloads_and_a_served_one_updates_in_place(tmp_path):
    ctx = _machine(tmp_path)
    assert "location.reload" not in page.build(ctx)
    assert "/sections.json" not in page.build(ctx)
    ctx.once = False
    built = page.build(ctx)
    assert "location.reload" not in built
    assert "/sections.json" in built


def test_sections_are_the_page_s_parts(tmp_path):
    ctx = _machine(tmp_path)
    got = page.sections(ctx)
    assert {"bar", "status", "loss", "bias", "nowcasts", "tables"} <= set(got)
    built = page.build(ctx)
    for key in ("loss", "bias", "tables"):
        assert got[key] in built


def test_checkpoint_sizes_also_come_from_the_score_table(tmp_path):
    ctx = _machine(tmp_path)
    # A run known only by its release prefix, sized in the score table.
    ctx.runs = list(ctx.runs) + list(runs.load_runs([], "other=other-ft-", []))
    sizes = page.checkpoint_sizes(
        ctx,
        {r.key: {"ck": page._checkpoints(r)} for r in ctx.runs},
        [{"checkpoint": "other-ft-bbbbbb", "outbreaks": "500"}],
    )
    assert sizes["bbbbbb"] == 500
    assert sizes["aaaaaa"] == 250


def test_the_mailbox_section_is_hidden_without_a_mailbox(tmp_path):
    ctx = _machine(tmp_path)
    ctx.mailbox = None
    assert "Recent activity" not in page.build(ctx)


def test_samples_are_summarised_outside_page_builds(tmp_path, monkeypatch):
    from flowlib.dashboard import nowcasts

    ctx = _machine(tmp_path)
    ctx.datasets = [{"key": "b", "format": "samples", "groups": []}]
    calls = []
    monkeypatch.setattr(
        nowcasts, "summarise_samples", lambda *a, **_k: calls.append(a) or 14
    )
    truth = tmp_path / "truth.csv"
    truth.write_text("kind\n")
    (tmp_path / "scores" / "b").mkdir(parents=True)
    monkeypatch.setattr(ctx.source, "truth", lambda _key: truth)
    page.build(ctx)
    assert calls == []
    ctx.summarise()
    ctx.summarise()  # nothing changed: the summaries are not redone
    assert len(calls) == 1
    assert ctx.delays["b"] == 14


def test_checkpoint_tables_have_a_column_per_source(tmp_path):
    ctx = _machine(tmp_path)
    out = page.build(ctx)
    tables = out.split("<h2>Tables</h2>")[1]
    assert "<th>weekly<th>daily" in tables or "<th>daily<th>weekly" in tables
    assert "<td>0.2000" in tables


def test_the_run_size_comes_from_the_config_without_a_trainer(tmp_path):
    # Under Snakemake there is no trainer command line to read.
    ctx = _machine(tmp_path)
    ctx.outbreaks = 8400
    ctx.prefixes = (250, 500, 1000, 2000, 4000, 8400)
    out = page.build(ctx)
    assert "of 8400 simulations trained" in out
    assert "of 8000" not in out
