# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Recompute the trainer's real-validation loss on CPU for both models.

Ties `covariate_ablation.py`'s block rebuild and adapter loading to the
recorded `real_validation_loss` (n1000) and `ZERO_SHOT_REAL`.

    python scripts/covariate_ablation_anchor.py DELAYS ADAPTER
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import finetune_chronos as ft
import finetune_real_validation as frv
import finetune_stream as fs


def main() -> None:
    """Print each model's trainer-style loss on the BDBV blocks."""
    import torch
    from chronos import Chronos2Pipeline
    from peft import PeftModel

    from bdbvhubrenewal.forecasters.foundation import (
        CHRONOS_MODEL,
        CHRONOS_REVISION,
    )

    torch.set_num_threads(1)
    d = pd.read_csv(sys.argv[1])
    line_list = d[d["vintage"] == frv.VINTAGE]
    blocks = ft.real_blocks(line_list, frv.CUTS, frv.LAST_WEEK, frv.LAG)

    def load():
        return Chronos2Pipeline.from_pretrained(
            CHRONOS_MODEL, revision=CHRONOS_REVISION, device_map="cpu"
        ).model

    tuned = PeftModel.from_pretrained(load(), sys.argv[2]).merge_and_unload()
    for name, model in (("zero-shot", load()), ("n1000", tuned)):
        print(name, fs.held_out_loss(model, blocks, 256))


if __name__ == "__main__":
    main()
