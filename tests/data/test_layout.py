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
    "data/hierarchy.py",
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
    assert "MissingSnapshots" in _defs("forecasters/base.py")
    assert "Forecast" in _defs("forecasters/forecast.py")


def test_hub_release_labels_live_with_the_hub_data() -> None:
    assert {"ORIGINAL_AS_OF", "release_id"} <= _defs("data/tidy.py")
    assert not {"ORIGINAL_AS_OF", "release_id"} & _defs("data/releases.py")


def test_generic_data_modules_import_no_hub_data_code() -> None:
    allowed = {
        f"bdbvhubrenewal.{m.removesuffix('.py').replace('/', '.')}"
        for m in GENERIC
    }
    for rel in (m for m in GENERIC if m.startswith("data/")):
        own = {m for m in _imported(rel) if m.startswith("bdbvhubrenewal")}
        assert own <= allowed, (rel, own - allowed)


def test_snapshots_hold_no_hub_location_codes() -> None:
    assert not {"NATIONAL", "parent_code"} & _defs("data/snapshots.py")
    assert "CD" not in (PKG / "data/snapshots.py").read_text()


def _imports_from(rel: str, module: str) -> set[str]:
    tree = ast.parse((PKG / rel).read_text())
    return {
        a.name
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and n.module == module
        for a in n.names
    }


def test_zone_levels_and_weekly_counts_are_generic() -> None:
    assert {"is_zone", "province_of"} <= _defs("data/hierarchy.py")
    assert "weekly_counts" in _defs("data/snapshots.py")
    moved = {"is_zone", "province_of", "weekly_zone_counts"}
    assert not moved & _defs("data/zones.py")
    for path in PKG.rglob("*.py"):
        rel = path.relative_to(PKG).as_posix()
        names = _imports_from(rel, "bdbvhubrenewal.data.zones")
        assert not names & moved, (rel, names & moved)


def test_model_code_imports_no_hub_zone_module() -> None:
    for rel in (
        "forecasters/base.py",
        "forecasters/foundation.py",
        "forecasters/zones.py",
        "forecasters/renewal.py",
        "evaluation/nowcast.py",
    ):
        assert "bdbvhubrenewal.data.zones" not in _imported(rel), rel


def test_renewal_code_reads_no_hub_delay_or_places() -> None:
    for rel in (
        "model/inputs.py",
        "model/zones.py",
        "forecasters/renewal.py",
        "forecasters/zonerenewal.py",
    ):
        own = _imported(rel)
        assert "bdbvhubrenewal.data.delays" not in own, rel
        assert "load_locations" not in _imports_from(rel, "bdbvhubrenewal.data")
        assert "_centres.csv" not in (PKG / rel).read_text(), rel
    assert "zone_places" in _defs("data/zones.py")
