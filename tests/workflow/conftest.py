"""Put the workflow's helpers on the path."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "workflow"))
