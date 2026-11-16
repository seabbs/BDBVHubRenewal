"""`scripts/follow_checkpoint.py` picks the final run's newest checkpoint."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "follow_checkpoint.py"
spec = importlib.util.spec_from_file_location("follow_checkpoint", SCRIPT)
follow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(follow)


def test_the_newest_release_of_the_grid_wins() -> None:
    releases = [
        ("chronos-ft-e8a1f5ae338b", "2026-11-15T16:21:13Z", "r3", 4000),
        ("chronos-ft-aaaaaaaaaaaa", "2026-11-16T10:00:00Z", "r4", 1000),
        ("chronos-ft-bbbbbbbbbbbb", "2026-11-16T20:00:00Z", "r4.1", 2000),
        ("realdata-ft-cccccccccccc", "2026-11-17T00:00:00Z", "r4", 0),
    ]
    assert follow.newest(releases, "r4") == ("bbbbbbbbbbbb", 2000)
    assert follow.newest(releases[:1], "r4") is None


def test_the_pin_is_written_into_the_config(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("render: development\nfinetuned: null\nn: 1\n")
    follow.write_pin(config, "bbbbbbbbbbbb")
    assert "finetuned: bbbbbbbbbbbb\n" in config.read_text()
    assert config.read_text().startswith("render: development\n")
