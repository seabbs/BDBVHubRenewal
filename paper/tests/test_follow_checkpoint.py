"""`scripts/follow_checkpoint.py` pins the final run's newest checkpoint."""

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "follow_checkpoint.py"
spec = importlib.util.spec_from_file_location("follow_checkpoint", SCRIPT)
follow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(follow)


def _run(root: Path, sizes: list[int]) -> Path:
    """A run folder with `nK` checkpoints, each with its settings."""
    for k in sizes:
        d = root / f"n{k}"
        d.mkdir(parents=True)
        (d / "settings.json").write_text(
            json.dumps({"n_outbreaks": k, "consumed_sha256": f"c{k}"})
        )
    return root


def test_local_checkpoints_are_matched_to_their_releases(tmp_path) -> None:
    run = _run(tmp_path / "slots", [1000, 2000, 4000])
    releases = {"c1000": "aaaaaaaaaaaa", "c2000": "bbbbbbbbbbbb"}
    found = follow.published(run, releases)
    # 4,000 has no release yet, so it is not pinned.
    assert found == [("aaaaaaaaaaaa", 1000), ("bbbbbbbbbbbb", 2000)]
    assert follow.published(tmp_path / "absent", releases) == []


def test_the_pin_and_the_run_s_checkpoints_are_written(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "render: development\nfinetuned: null\nfinal_checkpoints: []\nn: 1\n"
    )
    follow.write_pins(config, [("aaaaaaaaaaaa", 1000), ("bbbbbbbbbbbb", 2000)])
    text = config.read_text()
    assert "finetuned: bbbbbbbbbbbb\n" in text
    assert "final_checkpoints: [aaaaaaaaaaaa, bbbbbbbbbbbb]\n" in text
    assert text.startswith("render: development\n")


def test_slots_releases_pin_by_their_spec() -> None:
    # A `chronos-slots-<hash>` release is pinned as `slots-<hash>`, the
    # spec the loader takes; `chronos-ft-<hash>` as its hash.
    assert follow.spec_of("chronos-slots-aaaaaaaaaaaa") == "slots-aaaaaaaaaaaa"
    assert follow.spec_of("chronos-ft-bbbbbbbbbbbb") == "bbbbbbbbbbbb"
    assert follow.spec_of("realdata-ft-cccccccccccc") is None
