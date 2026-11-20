# nowcastfoundation

Foundation-model nowcasts and forecasts.

- `forecaster`: Chronos-2 forecasters on reporting triangles.
- `finetune`, `series_bias`, `checkpoints`: fine-tuning blocks, a learned per-series bias, and the published checkpoints.
- `checkpoint`, `live_checkpoint`, `live_checkpoint_page`, `sidecar`: scoring a checkpoint against references, its live page, and held-out blocks.
- `simulate`: simulated outbreaks for training.
- `r/`: the R simulation engine, an R package.

```sh
uv add nowcastfoundation
```
