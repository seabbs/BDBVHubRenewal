"""Where shared helpers live before the packages move (WP2a, WP2b).

The generic modules go to `nowcastutils`, which may not import the hub
data code that goes to `bdbvhub`.
"""

from __future__ import annotations

import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[2] / "src/bdbvhubrenewal"

GENERIC = (
    "data/dates.py",
    "data/files.py",
    "data/releases.py",
    "data/snapshots.py",
    "forecasters/base.py",
)


def _defs(rel: str) -> set[str]:
    tree = ast.parse((PKG / rel).read_text())
    out = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            out |= {t.id for t in targets if isinstance(t, ast.Name)}
    return out


def _imported(rel: str) -> set[str]:
    tree = ast.parse((PKG / rel).read_text())
    return {
        n.module
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and n.module
    }


def test_generic_helpers_live_in_generic_modules() -> None:
    assert {
        "MAX_PAIR_AGE",
        "cutoff_lags",
        "latest_onset_days",
        "release_cutoff",
        "release_pairs",
    } <= _defs("data/releases.py")
    assert "write_csv" in _defs("data/files.py")
    assert "MissingSnapshots" in _defs("data/snapshots.py")
    assert "Forecast" in _defs("forecasters/base.py")


def test_hub_release_labels_live_with_the_hub_data() -> None:
    assert {"ORIGINAL_AS_OF", "release_id"} <= _defs("data/tidy.py")
    assert not {"ORIGINAL_AS_OF", "release_id"} & _defs("data/releases.py")


def test_generic_data_modules_import_no_hub_data_code() -> None:
    allowed = {
        f"bdbvhubrenewal.{m.removesuffix('.py').replace('/', '.')}"
        for m in GENERIC
    }
    for rel in GENERIC[:4]:
        own = {m for m in _imported(rel) if m.startswith("bdbvhubrenewal")}
        assert own <= allowed, (rel, own - allowed)
