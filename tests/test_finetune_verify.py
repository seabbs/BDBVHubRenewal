"""A checkpoint's hash recomputes from the commit it records."""

import inspect
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from finetune_verify import function_source, problems, settings_problems

from bdbvhubrenewal.forecasters import foundation

FOUNDATION = Path(foundation.__file__)


def test_function_source_is_what_inspect_hashes():
    text = FOUNDATION.read_text()
    for f in (foundation.hub_triangle, foundation.layout_block):
        assert function_source(text, f.__name__) == inspect.getsource(f)


@pytest.fixture
def checkpoint(tmp_path):
    import hashlib

    files = {"scripts/a.py": b"a = 1\n", "src/b.py": b"b = 2\n"}
    code = {p: hashlib.sha256(v).hexdigest() for p, v in files.items()}
    settings = {"code_sha256": code, "n_outbreaks": 10}
    text = json.dumps(settings, sort_keys=True).encode()
    settings["hash"] = hashlib.sha256(text).hexdigest()[:12]
    settings["code_commit"] = "abc"
    settings["held_out_loss"] = 0.1
    (tmp_path / "settings.json").write_text(json.dumps(settings))
    return tmp_path, files


def test_a_checkpoint_that_matches_its_commit_has_no_problems(checkpoint):
    where, files = checkpoint
    assert problems(where, lambda _, path: files[path]) == []


def test_changed_code_at_the_commit_is_a_problem(checkpoint):
    where, files = checkpoint
    files = {**files, "src/b.py": b"b = 3\n"}
    found = problems(where, lambda _, path: files[path])
    assert any("src/b.py" in p for p in found)


def test_a_changed_setting_breaks_the_hash(checkpoint):
    where, files = checkpoint
    settings = json.loads((where / "settings.json").read_text())
    settings["n_outbreaks"] = 11
    (where / "settings.json").write_text(json.dumps(settings))
    found = problems(where, lambda _, path: files[path])
    assert any("hash" in p for p in found)


def test_real_validation_results_are_outside_the_hash(checkpoint):
    where, files = checkpoint
    settings = json.loads((where / "settings.json").read_text())
    settings["real_validation_loss"] = 0.2
    settings["real_validation_sha256"] = "abc"
    (where / "settings.json").write_text(json.dumps(settings))
    assert problems(where, lambda _, path: files[path]) == []


def test_loss_parts_are_outside_the_hash(checkpoint):
    where, files = checkpoint
    settings = json.loads((where / "settings.json").read_text())
    for name in ("held_out_loss", "real_validation_loss"):
        settings[f"{name}_nowcast"] = 0.04
        settings[f"{name}_forecast"] = 0.06
    (where / "settings.json").write_text(json.dumps(settings))
    assert problems(where, lambda _, path: files[path]) == []


def test_code_paths_point_names_at_moved_files(checkpoint):
    where, files = checkpoint
    settings = json.loads((where / "settings.json").read_text())
    moved = {"scripts/a.py": "workflow/a.py", "src/b.py": "packages/x/b.py"}
    settings["code_paths"] = moved
    (where / "settings.json").write_text(json.dumps(settings))
    at = {moved[p]: v for p, v in files.items()}
    assert problems(where, lambda _, path: at[path]) == []


def test_foundation_functions_are_read_from_code_paths(tmp_path):
    import hashlib

    src = "def hub_triangle():\n    return 1\n"
    code = {"foundation.hub_triangle": hashlib.sha256(src.encode()).hexdigest()}
    settings = {"code_sha256": code}
    text = json.dumps(settings, sort_keys=True).encode()
    settings["hash"] = hashlib.sha256(text).hexdigest()[:12]
    settings["code_paths"] = {"foundation": "packages/f/foundation.py"}
    (tmp_path / "settings.json").write_text(json.dumps(settings))
    files = {"packages/f/foundation.py": src.encode()}
    assert problems(tmp_path, lambda _, path: files[path]) == []


R6 = Path(__file__).parent / "fixtures" / "checkpoints"


@pytest.mark.parametrize("name", ["r6-n250", "r6-n500"])
def test_r6_checkpoints_give_their_hash(name):
    settings = json.loads((R6 / name / "settings.json").read_text())
    assert settings_problems(settings) == []


def test_the_code_names_are_the_paths_first_hashed():
    import finetune_chronos as ft

    assert set(ft.CODE) == {
        "scripts/finetune_simulate.R",
        "scripts/finetune_chronos.py",
        "src/bdbvhubrenewal/forecasters/finetune.py",
    }
    root = Path(__file__).parents[1]
    for path in (*ft.CODE.values(), ft.FOUNDATION_PATH):
        assert (root / path).exists(), path
