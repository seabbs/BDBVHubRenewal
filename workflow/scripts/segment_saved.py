"""Post the mailbox line for a saved checkpoint: segment_saved.py SETTINGS N"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from flowlib import localconfig, notify

settings, n = Path(sys.argv[1]), sys.argv[2]
key = json.loads(settings.read_text())["hash"]
notify.post(f"checkpoint n{n} saved, hash {key}", localconfig.read())
