# Evidence for #192: Chronos-2 candidate models

- `results/fm-summary.csv`: relative WIS against `randomwalknowcast`, log WIS, coverage and bias on the complete target, by nowcast and forecast weeks, all places and national, on the cells every model forecasts (`scripts/summary.py`).
  The Chronos-2 models give no delay, so they are scored on the complete target only.
  No forecast week is complete at the latest release (2026-11-08), so the table has nowcast weeks only.
- `results/nowcast-scores.csv`, `results/nowcast-origins.csv`: the release backtest's scores and origins.
- `results/fan-national.png`: national fans at 2026-10-24 and 2026-11-01 (`scripts/fan.py`).

Fits: `scripts/run-mac.sh` on the laptop's Apple GPU (MPS), one model and origin at a time, into a scratch fit store, at fm-chronos 7e5e506.
Scoring: `scripts/score.sh RESULTS MODEL...` at fm-chronos 675055f: `fit-store load` of `stores/base`, dropping the stored delays, then `backtest --models randomwalk randomwalknowcast --no-recovery`, then `summary.py`.
The forecasts do not depend on the delay, so the stored fits are those fm-chronos 675055f gives.

Lag arms (`scripts/arms.py`, `scripts/score-arm.sh`): the ragged models with lag cells carried forward (`carry`), one reported-so-far covariate per place (`reported`), or weekly D = 6 (`d6`). Summaries in `results/arms/`.

Stores (`stores/`): the scratch fit stores the tables score, `base` for the four models and `arm-*` for the arms' ragged models. Load one with `bdbvhubrenewal fit-store load --store stores/<name> --models ...`.

Data-driven D (`scripts/dmax.py`, `scripts/arm_dmax.py`): D per origin is the age by which 99% of the country's hub target is reported under randomwalknowcast's chain ladder delay (35 to 43 days); weekly D is that in weeks, rounded up. Scores in `results/arms/fm-summary-dmax.csv`, fans in `results/arms/fan-dmax-*.png`, fits in `stores/arm-dmax`.

## Fine-tuned ragged models (#193)

`fm_chronos_weekly_ragged_ft` and `fm_chronos_daily_ragged_ft` are the ragged models with Chronos-2 fine-tuned by LoRA on 15,723 blocks from 2,000 simulated epinowcast outbreaks (fm-finetune ac74886).
The checkpoint is in `checkpoint/`: the adapter (`finetuned-ckpt`), its settings (hash 2c2b22a03107) and the training history.
1,000 steps at lr 1e-4 with 5% of blocks held out took 53 min on the laptop's Apple GPU; held-out loss fell from 0.197 to 0.157.

Fits: `scripts/ft_fits.py d4|dmax STORE`, zero-shot and fine-tuned, at D = 4 weeks / 28 days and at the data-driven D (as `arm_dmax.py`), into `stores/ft-d4` and `stores/ft-dmax`.
Scoring: `scripts/score-ft.sh STORE RESULTS` (`fit-store load`, the release backtest with the random walks, `summary.py`), then `scripts/ft_table.py randomwalknowcast RESULTS...`.
Results in `results/finetune/`: `ft-table.md` (relative WIS against randomwalknowcast, relative log WIS in brackets, all places, matched cells), `fm-summary-<arm>.csv` (against randomwalknowcast, as `summary.py`), and fans of each layout, zero-shot against fine-tuned, at D = 4 w / 28 d (`fan-ft-<source>.png`, `scripts/fan_ft.py`).
