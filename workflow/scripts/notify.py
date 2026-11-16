"""Post a mailbox line from a shell rule: notify.py MESSAGE..."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from flowlib import localconfig, notify

notify.post(" ".join(sys.argv[1:]), localconfig.read())
