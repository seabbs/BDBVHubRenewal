"""The dashboard page, built from a fixture machine."""

import csv
import gzip
import json
import re
from datetime import datetime, timedelta

import pytest

pytest.importorskip("matplotlib")

from flowlib.dashboard import logs, mirror, page, runs

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
        prefixes=(250, 500, 750, 1000, 2000, 4000, 8000),
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
    assert "<b>my run</b>" in out
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


def test_run_sizes_drop_prefixes_past_the_run_and_end_at_it():
    assert page.run_sizes((250, 8000, 9000), 8400) == [250, 8000, 8400]
    assert page.run_sizes((250, 500), 500) == [250, 500]


def test_the_run_size_comes_from_the_config_without_a_trainer(tmp_path):
    ctx = _machine(tmp_path)
    # The fixture's run has no trainer command line to name its prefixes.
    assert logs.trainer_args(ctx.runs[0].folder) == ""
    ctx.outbreaks = 8400
    ctx.prefixes = (250, 500, 1000, 2000, 4000, 8000)
    out = page.build(ctx)
    assert "of 8400 simulations trained" in out
    assert "of 8000" not in out
    planned = page.run_sizes(ctx.prefixes, ctx.outbreaks)
    st = page.live_status(ctx.runs[0], ctx.log(ctx.runs[0]), planned)
    assert st.params["prefixes"][-2:] == [8000, 8400]


def test_wis_has_no_section_of_its_own(tmp_path):
    # Sam 11:05: each model's WIS is in the nowcasts' score card only.
    ctx = _machine(tmp_path)
    table = tmp_path / "scores" / "tables" / "checkpoint-scores.csv"
    with table.open() as f:
        rows = list(csv.DictReader(f))
    parts = ("wis", "dispersion", "underprediction", "overprediction")
    rows[0] |= dict(zip(parts, ("1.0", "0.5", "0.2", "0.3"), strict=True))
    with table.open("w") as f:
        w = csv.DictWriter(f, [*rows[0]])
        w.writeheader()
        w.writerows(rows)
    out = page.build(ctx)
    assert "<h2>Mean WIS</h2>" not in out
    assert '<a href="#wis">' not in out
    assert "wis" not in page.sections(ctx)


def test_a_segment_s_empty_folder_is_not_a_checkpoint_being_scored(tmp_path):
    ctx = _machine(tmp_path)
    folder = ctx.runs[0].folder
    # Snakemake makes the next segment's output folders when it starts.
    (folder / "train" / "n500" / "finetuned-ckpt").mkdir(parents=True)
    out = page.build(ctx)
    assert "scoring since" not in out
    bar = out.split("id=livebar>", 1)[1].split("</span></span>", 1)[0]
    # Simulations done of the next prefix, and when it is due.
    assert re.search(r"n\d+ / n500", bar)
    assert "ETA" in bar


def test_a_written_checkpoint_is_scored_as_itself_not_as_the_next(tmp_path):
    ctx = _machine(tmp_path)
    ckpt = ctx.runs[0].folder / "train" / "n500" / "finetuned-ckpt"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_model.safetensors").write_bytes(b"")
    out = page.build(ctx)
    assert re.search(r"n500 scoring since \d\d:\d\d", out)
    assert re.search(r"n\d+ / n750", out)


def test_the_bar_s_clock_is_spaced_from_the_status(tmp_path):
    bar = page.sections(_machine(tmp_path))["bar"]
    assert "</span> <span class=mut>" in bar


def test_the_learning_rate_falls_over_simulations_not_constant(tmp_path):
    ctx = _machine(tmp_path)
    folder = ctx.runs[0].folder
    s = json.loads((folder / "train/n250/settings.json").read_text())
    s |= {
        "learning_rate": 1e-5,
        "lr_decay": "linear to zero over the outbreaks",
    }
    (folder / "train/n250/settings.json").write_text(json.dumps(s))
    ctx.outbreaks = 1000
    ctx.prefixes = (250, 500, 1000)
    out = page.build(ctx)
    assert "linear decay to 0 at 1000 simulations" in out
    assert ">constant<" not in out
    assert "learning rate vs simulations trained" in out


def test_the_last_log_line_is_the_running_segment_s(tmp_path):
    ctx = _machine(tmp_path)
    folder = ctx.runs[0].folder
    now = datetime.now().replace(microsecond=0)
    (folder / "train" / "n500.log").write_text(
        f"{now:%Y-%m-%d %H:%M:%S} [train] resumed at step 1140, "
        "outbreaks 251-\n"
    )
    out = page.build(ctx)
    tile = out.split("Last log line", 1)[1].split("</div></div>", 1)[0]
    assert f"{now:%H:%M:%S}" in tile
    assert "STALE" not in tile
    assert "resumed at step 1140" in tile


def test_the_next_checkpoint_names_its_simulations_and_planned_step(tmp_path):
    ctx = _machine(tmp_path)
    folder = ctx.runs[0].folder
    (folder / "train" / "n500.log").write_text(
        "2026-11-16 16:30:00 [train] outbreaks 251-500: 9 blocks, "
        "99 series, 1000 steps\n"
    )
    out = page.build(ctx)
    tile = out.split("Next checkpoint", 1)[1].split("</div></div>", 1)[0]
    assert ">n500<" in tile
    # 250 checkpointed at step 1140; n500's chunk planned as 1000 steps.
    assert "of 500 simulations" in tile
    assert "step 2140" in tile
    assert "planned" in tile


def test_every_configured_dataset_is_in_the_selector(tmp_path):
    # Sam 11:05: Germany, BDBV and the held-out simulations, each listed
    # whether or not it has nowcasts yet.
    ctx = _machine(tmp_path)
    ctx.datasets = [
        *ctx.datasets,
        {"key": "bdbv", "label": "BDBV", "format": "samples"},
        {
            "key": "held_out",
            "label": "Held-out simulations",
            "format": "quantiles",
            "missing": "the scoring workflow writes none yet",
        },
    ]
    sel = page.sections(ctx)["nowcasts"]
    data = json.loads(sel.split("id=gdata>", 1)[1].split("</script>", 1)[0])
    got = [
        (d["key"], d["label"], bool(d.get("empty"))) for d in data["datasets"]
    ]
    assert ("bdbv", "BDBV", True) in got
    assert ("held_out", "Held-out simulations", True) in got
    held = next(d for d in data["datasets"] if d["key"] == "held_out")
    assert "the scoring workflow writes none yet" in held["missing"]


def test_speed_and_etas_come_from_the_status_file_when_steps_are_sparse(
    tmp_path,
):
    # One logged step in the last 30 minutes (evaluations every 400 steps)
    # and the trainer's status: a rate, simulations a minute and ETAs.
    ctx = _machine(tmp_path)
    folder = ctx.runs[0].folder
    now = datetime.now().replace(microsecond=0)
    (folder / "train.log").write_text(
        LINE.format(m=0, s=800, l=0.5, h=0.45).replace(
            "2026-11-16 16:00:00",
            f"{now - timedelta(minutes=40):%Y-%m-%d %H:%M:%S}",
        )
        + "\n"
    )
    (folder / "train" / "stream-status.json").write_text(
        json.dumps(
            {
                "stage": "training",
                "step": 1200,
                "outbreaks": 250,
                "time": f"{now - timedelta(minutes=1):%Y-%m-%d %H:%M:%S}",
            }
        )
    )
    out = page.build(ctx)
    tile = out.split("Steps / s", 1)[1].split("</div></div>", 1)[0]
    assert "n/a" not in tile
    assert "simulations/min" in tile
    eta = out.split("ETA n8000", 1)[1].split("</div></div>", 1)[0]
    assert "n/a" not in eta


def test_the_loss_panel_offers_each_loss_type(tmp_path):
    out = page.build(_machine(tmp_path))
    for k in ("total", "nowcast", "forecast", "layout", "level"):
        assert f"data-lt={k}>" in out
