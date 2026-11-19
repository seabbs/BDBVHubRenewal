"""The dashboard reads a training run's log, checkpoints and process."""

import json
import os
import time

from flowlib.dashboard import logs

LINE = (
    "2026-11-16 16:50:01 [train] step {step}: loss {loss}, held-out "
    "(first 100) {held}"
)
SOURCES = ", by source 3day 0.31, daily 0.29, weekly 0.12"


def test_steps_are_parsed_with_their_losses(tmp_path):
    path = tmp_path / "train.log"
    path.write_text(
        LINE.format(step=10, loss=0.5, held=0.4) + SOURCES + "\n"
        "noise\n" + LINE.format(step=20, loss=0.45, held=0.39) + "\n"
    )
    log = logs.Log(path)
    rows = log.rows()
    assert [r["step"] for r in rows] == [10, 20]
    assert rows[0]["loss"] == 0.5
    assert rows[0]["held"] == 0.4
    assert rows[0]["src"] == {"3day": 0.31, "daily": 0.29, "weekly": 0.12}
    assert rows[1]["src"] == {}


def test_only_whole_lines_are_read_and_new_ones_appended(tmp_path):
    path = tmp_path / "train.log"
    path.write_text(LINE.format(step=10, loss=0.5, held=0.4) + "\n" + "2026")
    log = logs.Log(path)
    assert [r["step"] for r in log.rows()] == [10]
    with path.open("a") as f:
        f.write("-11-16 16:51:00 [train] step 20: loss 0.4, held-out (x) 0.3\n")
    assert [r["step"] for r in log.rows()] == [10, 20]


def test_a_rewritten_log_is_read_again(tmp_path):
    path = tmp_path / "train.log"
    path.write_text(LINE.format(step=10, loss=0.5, held=0.4) + "\n" * 5)
    log = logs.Log(path)
    log.rows()
    path.write_text(LINE.format(step=5, loss=0.6, held=0.5) + "\n")
    assert [r["step"] for r in log.rows()] == [5]


def test_a_missing_log_has_no_rows(tmp_path):
    assert logs.Log(tmp_path / "none.log").rows() == []


def test_checkpoints_are_listed_by_size(tmp_path):
    for n in (1000, 250):
        d = tmp_path / "train" / f"n{n}"
        d.mkdir(parents=True)
        (d / "settings.json").write_text(json.dumps({"hash": f"h{n}"}))
    got = logs.checkpoints(tmp_path)
    assert [(name, s["hash"]) for name, s, _ in got] == [
        ("n250", "h250"),
        ("n1000", "h1000"),
    ]


def test_prefixes_from_the_trainer_command_line_or_saved_folders(tmp_path):
    args = "python scripts/finetune_stream.py a b --prefixes 250 500 8000 --x"
    assert logs.prefixes(tmp_path, args, (250,)) == [250, 500, 8000]
    (tmp_path / "train" / "n750").mkdir(parents=True)
    got = logs.prefixes(tmp_path, "", default=(250, 1000))
    assert got == [250, 750, 1000]


def test_a_trainer_is_alive_while_its_pid_runs(tmp_path):
    (tmp_path / "trainer.pid").write_text(f"{os.getpid()}\n")
    text, alive = logs.trainer_state(tmp_path)
    assert alive is True
    assert str(os.getpid()) in text
    (tmp_path / "trainer.pid").write_text("999999999\n")
    assert logs.trainer_state(tmp_path)[1] is False


def _status(folder, stage, age_s):
    path = folder / "train" / "stream-status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"stage": stage}))
    when = time.time() - age_s
    os.utime(path, (when, when))


def test_a_fresh_stream_status_means_alive_whatever_the_pid(tmp_path):
    # Under Snakemake the pid file is an old launcher's.
    (tmp_path / "trainer.pid").write_text("999999999\n")
    _status(tmp_path, "training", 60)
    text, alive = logs.trainer_state(tmp_path)
    assert alive is True
    assert "status" in text
    # No pid file at all.
    (tmp_path / "trainer.pid").unlink()
    assert logs.trainer_state(tmp_path)[1] is True


def test_a_stale_status_and_a_dead_pid_mean_dead(tmp_path):
    (tmp_path / "trainer.pid").write_text("999999999\n")
    _status(tmp_path, "training", 3600)
    assert logs.trainer_state(tmp_path)[1] is False


def test_a_stopped_run_is_neither_alive_nor_dead(tmp_path):
    for stage in ("finished", "stopped at n500"):
        _status(tmp_path, stage, 3600)
        text, alive = logs.trainer_state(tmp_path)
        assert alive is None
        assert stage in text


def test_a_run_reads_every_segment_log_and_the_old_train_log(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "train.log").write_text(
        LINE.format(step=10, loss=0.5, held=0.4) + "\n"
    )
    (tmp_path / "train" / "n250.log").write_text(
        LINE.format(step=20, loss=0.45, held=0.39) + "\n"
    )
    log = logs.RunLog(tmp_path)
    assert [r["step"] for r in log.rows()] == [10, 20]
    # A new segment's log is found as soon as it is written.
    (tmp_path / "train" / "n500.log").write_text(
        LINE.format(step=30, loss=0.4, held=0.38) + "\n"
    )
    assert [r["step"] for r in log.rows()] == [10, 20, 30]


def test_a_train_log_linked_to_a_segment_log_is_read_once(tmp_path):
    (tmp_path / "train").mkdir()
    seg = tmp_path / "train" / "n2000.log"
    seg.write_text(LINE.format(step=20, loss=0.45, held=0.39) + "\n")
    (tmp_path / "train.log").symlink_to(seg)
    assert [r["step"] for r in logs.RunLog(tmp_path).rows()] == [20]


def test_steps_are_parsed_with_their_nowcast_and_forecast_parts(tmp_path):
    path = tmp_path / "train.log"
    path.write_text(
        LINE.format(step=10, loss=0.5, held=0.4)
        + " (nowcast 0.15, forecast 0.25)"
        + SOURCES
        + "\n"
        + LINE.format(step=20, loss=0.45, held=0.39)
        + "\n"
    )
    rows = logs.Log(path).rows()
    assert rows[0]["held"] == 0.4
    assert rows[0]["parts"] == {"nowcast": 0.15, "forecast": 0.25}
    assert rows[0]["src"] == {"3day": 0.31, "daily": 0.29, "weekly": 0.12}
    # A line from before the split has no parts.
    assert rows[1]["parts"] == {}
