"""Where the dashboard reads scores, forecasts, truth and notes from."""

import subprocess
from pathlib import Path

from flowlib.dashboard import mirror

DATASETS = [{"key": "germany"}, {"key": "bdbv", "truth_source": "data/t.csv"}]


def test_by_default_the_scoring_cache_is_read_in_place(tmp_path):
    cfg = {"REMOTE": "", "SCORE_CACHE": str(tmp_path / "s"), "MAILBOX": ""}
    src = mirror.Source(cfg, tmp_path / "m", datasets=DATASETS, root=tmp_path)
    assert src.scores == tmp_path / "s" / "tables" / "checkpoint-scores.csv"
    assert src.dataset("germany") == tmp_path / "s" / "germany"
    assert src.truth("bdbv") == tmp_path / "data" / "t.csv"
    assert src.truth("germany") is None
    assert src.mailbox is None
    assert src.commands() == []


def test_remote_sources_are_mirrored_read_only(tmp_path):
    datasets = [
        {"key": "germany", "source": "h:cache/germany/"},
        {
            "key": "bdbv",
            "source": "h:cache/bdbv/",
            "truth_source": "h:repo/data/target.csv",
        },
    ]
    src = mirror.Source(
        {"REMOTE": "", "SCORE_CACHE": "/x", "MAILBOX": ""},
        tmp_path / "m",
        datasets=datasets,
        scores="h:repo/evidence/scores.csv",
        mailbox={"source": "h:box.md", "lines": 30},
        root=tmp_path,
    )
    assert src.dataset("germany") == tmp_path / "m" / "germany"
    assert src.scores == tmp_path / "m" / "checkpoint-scores.csv"
    assert src.truth("bdbv") == tmp_path / "m" / "bdbv-truth.csv"
    assert src.mailbox == tmp_path / "m" / "mailbox.md"
    cmds = src.commands()
    assert [
        *mirror.RSYNC,
        "h:cache/germany/",
        f"{tmp_path / 'm' / 'germany'}/",
    ] in cmds
    assert [
        *mirror.RSYNC,
        "h:box.md",
        str(tmp_path / "m" / "mailbox.md"),
    ] in cmds
    assert len(cmds) == 5


def test_local_sources_are_read_where_they_are(tmp_path):
    src = mirror.Source(
        {"REMOTE": "", "SCORE_CACHE": "/x", "MAILBOX": "/notes.md"},
        tmp_path / "m",
        datasets=[{"key": "g", "source": "out/g"}],
        root=tmp_path,
    )
    assert src.dataset("g") == tmp_path / "out" / "g"
    assert src.mailbox == Path("/notes.md")
    assert src.commands() == []


def test_the_old_remote_keys_still_mirror_the_scoring_cache(tmp_path):
    cfg = {
        "REMOTE": "h",
        "REMOTE_ROOT": "r",
        "SCORE_CACHE": "/x",
        "REMOTE_MAILBOX": "box.md",
        "MAILBOX": "",
    }
    src = mirror.Source(
        cfg, tmp_path, datasets=[{"key": "germany"}], root=tmp_path
    )
    assert src.dataset("germany") == tmp_path / "germany"
    assert [
        *mirror.RSYNC,
        "h:r/scores/germany/",
        f"{tmp_path / 'germany'}/",
    ] in src.commands()
    assert src.mailbox == tmp_path / "mailbox.md"


def test_refresh_reports_each_failed_copy(tmp_path, monkeypatch):
    src = mirror.Source(
        {"REMOTE": "", "SCORE_CACHE": "/x", "MAILBOX": ""},
        tmp_path,
        datasets=[{"key": "g", "source": "h:g/"}],
        mailbox={"source": "h:box.md"},
        root=tmp_path,
    )

    def run(cmd, **_):
        code = 23 if cmd[-1].endswith("mailbox.md") else 0
        return subprocess.CompletedProcess(cmd, code, "", "no such file")

    monkeypatch.setattr(mirror.subprocess, "run", run)
    status = src.refresh()
    assert status.startswith("mirrored")
    assert "mailbox.md failed" in status
    assert src.failed


def test_a_samples_dataset_is_mirrored_apart_from_its_summaries(tmp_path):
    datasets = [{"key": "bdbv", "format": "samples", "source": "h:cache/bdbv/"}]
    src = mirror.Source(
        {"REMOTE": "", "SCORE_CACHE": "/x", "MAILBOX": ""},
        tmp_path,
        datasets=datasets,
        root=tmp_path,
    )
    assert src.dataset("bdbv") == tmp_path / "bdbv-samples"


def test_notes_and_tables_may_be_remote(tmp_path):
    src = mirror.Source(
        {"REMOTE": "", "SCORE_CACHE": "/x", "MAILBOX": ""},
        tmp_path / "m",
        root=tmp_path,
        extras=["h:review/notes.md", "out/t.csv"],
    )
    assert src.extra(0) == tmp_path / "m" / "extra-0-notes.md"
    assert src.extra(1) == tmp_path / "out" / "t.csv"
    assert [
        *mirror.RSYNC,
        "h:review/notes.md",
        str(tmp_path / "m" / "extra-0-notes.md"),
    ] in src.commands()
