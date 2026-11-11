# Evidence for #192: Chronos-2 candidate models

- `results/fm-summary.csv`: relative WIS against `randomwalk`, log WIS, coverage and bias, by target (reported, complete), nowcast and forecast weeks, all places and national, on the cells every model forecasts (`scripts/summary.py`).
- `results/nowcast-scores.csv`, `results/nowcast-origins.csv`: the release backtest's scores and origins.
- `results/fan-national.png`: national fans at 2026-10-24 and 2026-11-01 (`scripts/fan.py`).

Fits: `scripts/run-mac.sh` on the laptop's Apple GPU (MPS), one model and origin at a time, into a scratch fit store, at fm-chronos 9b76db7.
Scoring: `scripts/score.sh` on archie's CPU: `fit-store load` of that store, then `backtest --models randomwalk randomwalknowcast --no-recovery`, then `summary.py`.
