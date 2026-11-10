# Why our 2026-11-14 forecasts sit above the hub's

Scripts and small outputs behind the comment on #164.
This branch is not for merging.
All runs are randomwalknowcast at the 2026-11-08 release, 2000 samples, on main 896976d2 unless named.

- `decomp.py` runs main and variants (`pre150`, `nonb` no NB or removal noise, `nowalk` walk spread near 0, `noplus1` no `apply_delay` +1) and writes `places.csv` (sizes, sources, start weeks) and `nowcast_point.csv`.
- `errs.py` prints the retrospective nowcast errors the NB sizes are fitted to.
- `backcheck.py` applies the 11-08 delay (in sample) and each release's own delay (out of sample) to earlier releases and compares with the 11-08 counts (`backcheck.csv`).
- `runrw.py` and `qs.py` give the bisect over #153, #167/#168, #171 and #150 (`bisect_quantiles.txt`) and main at 11-01 for the 11-07 round.

Run from a worktree root, e.g. `uv run python -I evidence/excess-2026-11-14/decomp.py OUT 2000 main,pre150,main+nonb,main+nowalk`.
