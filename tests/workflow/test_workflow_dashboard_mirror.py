"""Scores reach the dashboard from this machine or a read-only mirror."""

from pathlib import Path

from flowlib.dashboard import mirror


def test_with_no_remote_the_local_scores_are_read(tmp_path):
    cfg = {"REMOTE": "", "SCORE_CACHE": str(tmp_path / "scores")}
    src = mirror.Source(cfg, tmp_path / "mirror")
    assert src.root == tmp_path / "scores"
    assert src.command() is None


def test_a_remote_is_mirrored_with_rsync_read_only(tmp_path):
    cfg = {
        "REMOTE": "sims-host",
        "REMOTE_ROOT": "bdbv-rounds",
        "SCORE_CACHE": "/unused",
    }
    src = mirror.Source(cfg, tmp_path / "mirror")
    assert src.root == tmp_path / "mirror"
    cmd = src.command()
    assert cmd[0] == "rsync"
    assert "sims-host:bdbv-rounds/scores/" in cmd
    assert str(tmp_path / "mirror") + "/" in cmd
    # Only the score table and the German forecast caches.
    assert "--include=tables/checkpoint-scores.csv" in cmd
    assert "--include=germany/***" in cmd
    assert cmd[-3:-2] == ["--exclude=*"] or "--exclude=*" in cmd


def test_the_remote_scores_path_can_be_set(tmp_path):
    cfg = {
        "REMOTE": "h",
        "REMOTE_ROOT": "r",
        "REMOTE_SCORE_CACHE": "ckpt-scores/cache",
        "SCORE_CACHE": "/x",
    }
    assert "h:ckpt-scores/cache/" in mirror.Source(cfg, tmp_path).command()


def test_paths_under_the_root(tmp_path):
    src = mirror.Source({"REMOTE": "", "SCORE_CACHE": str(tmp_path)}, Path("m"))
    assert src.scores == tmp_path / "tables" / "checkpoint-scores.csv"
    assert src.germany == tmp_path / "germany"


def test_a_remote_mailbox_is_mirrored_too(tmp_path):
    cfg = {
        "REMOTE": "h",
        "REMOTE_ROOT": "r",
        "REMOTE_MAILBOX": "box.md",
        "SCORE_CACHE": "/x",
        "MAILBOX": "/local/box.md",
    }
    src = mirror.Source(cfg, tmp_path)
    assert src.mailbox == tmp_path / "mailbox.md"
    assert src.mailbox_command() == [
        "rsync",
        "-a",
        "--timeout=60",
        "h:box.md",
        str(tmp_path / "mailbox.md"),
    ]


def test_a_local_mailbox_is_read_where_it_is(tmp_path):
    cfg = {"REMOTE": "", "SCORE_CACHE": str(tmp_path), "MAILBOX": "/b.md"}
    src = mirror.Source(cfg, tmp_path / "m")
    assert src.mailbox == Path("/b.md")
    assert src.mailbox_command() is None
