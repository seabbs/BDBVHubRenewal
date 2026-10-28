"""Paths and manifest for the hub validation fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BUNDLE = Path(__file__).resolve().parent
REPO = BUNDLE.parents[2]
R_VALIDATOR = REPO / "scripts" / "validate_with_hubvalidations.R"
MANIFEST: dict[str, Any] = json.loads(
    (BUNDLE / "fixtures" / "manifest.json").read_text()
)
FIXTURES: list[dict[str, Any]] = MANIFEST["fixtures"]
# Pinned copy of hub-config, so tests do not change when the hub adds
# reference dates. Set HUB_PATH to test against a live clone instead.
SNAPSHOT = BUNDLE / "fixtures" / "hub-snapshot"
METADATA_DIRS = [
    BUNDLE / "fixtures" / "constructed" / "model-metadata",
    BUNDLE / "fixtures" / "real" / "model-metadata",
]
