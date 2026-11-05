# Plausibility review of the 2026-11-07 round

Evidence for the plausibility issues on the 2026-11-07 round (data as of 2026-11-01).
This branch is not for merging.

- `flags.py` computes the flags in `flags.csv` from the candidate files of run 41691221605 and a local `pprandomwalknowcast` run on the same inputs.
- `quantiles.csv` holds the 5%, 50%, 95% and 97.5% quantiles of every model, place and week.
- `figs.py` draws one grid of flagged places per model.
- `diag_rw.py` prints the random walk origin (delay growth, start week, nowcast) for chosen places.
- `diag_rec.py` compares the zone walks with the province walk before reconciliation.
- `flags2.py` is the flag rule calibrated to Sam's places (collapse, blow-up, no cases ever), writing `flags2.csv`; `figs2.py` draws Sam's places per model and Mutwanga against the other hub teams.
