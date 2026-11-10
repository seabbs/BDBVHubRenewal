# Plausibility review of the 2026-11-14 round

Evidence for the plausibility issues on the 2026-11-14 round (data as of the 2026-11-08 release, latest onset 11-05).
This branch is not for merging.

Inputs:
- the random walk models run locally on main at 96ac128d with `nowcast --model <m> --vintage 2026-11-08 --reference-date 2026-11-14`;
- renewal, renewalsp and renewalbvd from the stored 11-08 fits on the `fit-store` branch (dbace285, fitted at 60c44de6), with zones made by `store_zones.py` (`zones_for_model`, so no refit);
- last week's submitted files (`data/our-submissions`) and the 11-07 candidate quantiles (`q1107.csv`, from the 2026-11-07 evidence at 971b2572).

Scripts (each takes the scratch dir `S` first):
- `make_ts.py` stacks the national, province and zone counts of every release into `S/hub/ts.csv`.
- `check1107.py` scores the 11-07 round against the 11-08 release (`check1107.txt`).
- `flags3.py` writes `flags3.csv` and `quantiles.csv.gz`; its docstring states every rule, including uplift and step-up.
- `edge.py` writes `edge.csv`: draws below reported counts, zone and national sums, week-to-week jumps, never-seeded zones, quantile order.
- `seeding.py` writes `seeding.csv`: never-seeded zones seeded per draw against the data.
- `nowcast_growth.py` writes `nowcast_growth.csv`: nowcasts of 10-24 and 10-31 against past release-to-release growth.
- `diag_start.py` writes `stale-starts.csv`: zones whose walk starts 4 or more weeks before the cut-off.
- `then_now.py` writes `then_now.csv` and `then_now.md`; `figs_then_now.py` draws `then-now.png`.
- `figs3.py` draws one grid per model (`<model>.png`); `table.py` prints the flagged places of a model.
- `*_fa04ff08.*`: the renewal models rechecked on the newer stored 11-08 fits (`fit-store` b12df214, fitted at fa04ff08).
- `audit1107.py` audits the 2026-11-07 round (origin 11-01): the files we had then against main's code now, both scored on the reported target of the 2026-11-08 release through the model's delay (`score_releases`); `audit_report.py` writes `then_now_1107.md`, `audit1107_cells.csv` and `audit1107_summary.csv`. The renewal models now are the 11-01 fits at fa04ff08 on `fit-store`.
- `falls.py` measures how often and how far reported week counts fall in later releases (`falls.csv.gz`); `below_vs_data.py` compares each model's draws below the reported count with those falls (`below_vs_data.csv`).
- `zone_falls_score.py` compares `zone-backtest` scores of main (896976d2) and #163 (cab33511), on the reconciled zones, split by whether the count fell (`zone_falls_score*.csv`); `below_vs_data_896976d2.csv` is `below_vs_data.py` on the same two commits at the 11-08 origin.
