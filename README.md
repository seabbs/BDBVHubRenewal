# Evidence for #192: Chronos-2 candidate models

- `results/fm-summary.csv`: relative WIS against `randomwalknowcast`, log WIS, coverage and bias, by target (reported, complete), nowcast and forecast weeks, all places and national, on the cells every model forecasts (`scripts/summary.py`).
- `results/nowcast-scores.csv`, `results/nowcast-origins.csv`: the release backtest's scores and origins.
- `results/fan-national.png`: national fans at 2026-10-24 and 2026-11-01 (`scripts/fan.py`).

Fits: `scripts/run-mac.sh` on the laptop's Apple GPU (MPS), one model and origin at a time, into a scratch fit store, at fm-chronos 7e5e506.
Scoring: `scripts/score.sh` on archie's CPU: `fit-store load` of that store, then `backtest --models randomwalk randomwalknowcast --no-recovery`, then `summary.py`.

Lag arms (`scripts/arms.py`, `scripts/score-arm.sh`): the ragged models with lag cells carried forward (`carry`), one reported-so-far covariate per place (`reported`), or weekly D = 6 (`d6`). Summaries in `results/arms/`.

Stores (`stores/`): the scratch fit stores the tables score, `base` for the four models and `arm-*` for the arms' ragged models. Load one with `bdbvhubrenewal fit-store load --store stores/<name> --models ...`.

Data-driven D (`scripts/dmax.py`, `scripts/arm_dmax.py`): D per origin is the age by which 99% of the country's hub target is reported under randomwalknowcast's chain ladder delay (35 to 43 days); weekly D is that in weeks, rounded up. Scores in `results/arms/fm-summary-dmax.csv`, fans in `results/arms/fan-dmax-*.png`, fits in `stores/arm-dmax`.
