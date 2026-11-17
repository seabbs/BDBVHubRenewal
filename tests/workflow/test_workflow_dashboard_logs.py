"""The dashboard reads a training run's log, checkpoints and process."""

import json
import os

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
    assert rows[1]["src"]["daily"] is None


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
    assert logs.prefixes(tmp_path, args) == [250, 500, 8000]
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
