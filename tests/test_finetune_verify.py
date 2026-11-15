"""A checkpoint's hash recomputes from the commit it records."""

import inspect
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from finetune_verify import function_source, problems

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
    assert problems(where, lambda commit, path: files[path]) == []


def test_changed_code_at_the_commit_is_a_problem(checkpoint):
    where, files = checkpoint
    files = {**files, "src/b.py": b"b = 3\n"}
    found = problems(where, lambda commit, path: files[path])
    assert any("src/b.py" in p for p in found)


def test_a_changed_setting_breaks_the_hash(checkpoint):
    where, files = checkpoint
    settings = json.loads((where / "settings.json").read_text())
    settings["n_outbreaks"] = 11
    (where / "settings.json").write_text(json.dumps(settings))
    found = problems(where, lambda commit, path: files[path])
    assert any("hash" in p for p in found)
