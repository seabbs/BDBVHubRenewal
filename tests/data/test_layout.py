"""Where shared helpers live across the workspace packages.

The generic modules are in `nowcastutils`, which may not import the hub
data code. Modules are named by import path, so the checks follow them
when they move between packages.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

GENERIC = (
    "nowcastutils.dates",
    "nowcastutils.files",
    "nowcastutils.hierarchy",
    "nowcastutils.releases",
    "nowcastutils.snapshots",
    "nowcastutils.base",
)

HUB_ZONES = "bdbvhub.data.zones"


def _file(module: str) -> Path:
    spec = importlib.util.find_spec(module)
    assert spec is not None, module
    assert spec.origin is not None, module
    return Path(spec.origin)


def _tree(module: str) -> ast.Module:
    return ast.parse(_file(module).read_text())


def _defs(module: str) -> set[str]:
    out = set()
    for node in _tree(module).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            out |= {t.id for t in targets if isinstance(t, ast.Name)}
    return out


def _imported(module: str) -> set[str]:
    return {
        n.module
        for n in ast.walk(_tree(module))
        if isinstance(n, ast.ImportFrom) and n.module
    }


def _imports_from(module: str, source: str) -> set[str]:
    return {
        a.name
        for n in ast.walk(_tree(module))
        if isinstance(n, ast.ImportFrom) and n.module == source
        for a in n.names
    }


def test_generic_helpers_live_in_generic_modules() -> None:
    assert {
        "MAX_PAIR_AGE",
        "cutoff_lags",
        "latest_onset_days",
        "release_cutoff",
        "release_pairs",
    } <= _defs("nowcastutils.releases")
    assert "write_csv" in _defs("nowcastutils.files")
    assert "MissingSnapshots" in _defs("nowcastutils.base")
    assert "Forecast" in _defs("nowcastutils.forecast")


def test_hub_release_labels_live_with_the_hub_data() -> None:
    hub = _defs("bdbvhub.data.tidy")
    assert {"ORIGINAL_AS_OF", "release_id"} <= hub
    labels = {"ORIGINAL_AS_OF", "release_id"}
    assert not labels & _defs("nowcastutils.releases")


def test_generic_modules_import_only_generic_modules() -> None:
    for module in GENERIC:
        own = {
            m
            for m in _imported(module)
            if m.split(".")[0] in ("bdbvhubrenewal", "bdbvhub")
        }
        assert not own, (module, own)


def test_snapshots_hold_no_hub_location_codes() -> None:
    assert not {"NATIONAL", "parent_code"} & _defs("nowcastutils.snapshots")
    assert "CD" not in _file("nowcastutils.snapshots").read_text()


def test_zone_levels_and_weekly_counts_are_generic() -> None:
    assert {"is_zone", "province_of"} <= _defs("nowcastutils.hierarchy")
    assert "weekly_counts" in _defs("nowcastutils.snapshots")
    moved = {"is_zone", "province_of", "weekly_zone_counts"}
    assert not moved & _defs(HUB_ZONES)
    root = _file("bdbvhub").parent
    for path in root.rglob("*.py"):
        module = ".".join(
            ("bdbvhub", *path.relative_to(root).with_suffix("").parts)
        ).removesuffix(".__init__")
        names = _imports_from(module, HUB_ZONES)
        assert not names & moved, (module, names & moved)


def test_model_code_imports_no_hub_zone_module() -> None:
    for module in (
        "nowcastutils.base",
        "nowcastfoundation.forecaster",
        "nowcastbaselines.zones",
        "nowcastrenewal.forecaster",
        "nowcasteval.nowcast",
    ):
        assert HUB_ZONES not in _imported(module), module


def test_renewal_code_reads_no_hub_delay_or_places() -> None:
    for module in (
        "nowcastrenewal.model.inputs",
        "nowcastrenewal.model.zones",
        "nowcastrenewal.forecaster",
        "nowcastrenewal.zones",
    ):
        assert "bdbvhub.data.delays" not in _imported(module), module
        loaders = _imports_from(module, "bdbvhub.data.tidy")
        assert "load_locations" not in loaders, module
    assert "zone_places" in _defs(HUB_ZONES)
