# Local-level walk for randomwalknowcast

`mainnow` is main (bc134ccb) and `ll` is the PR. `armtab.py`, `checktab.py` and `leveltab.py` make the tables in the PR; `blowdiag.py` is the AND-rule blow-up diagnosis behind it.

`final` is the PR head (16023193); `llnoise` and `llnb` are the 1/(F + y) noise and prior-median size arms on the first version (`ll`). `checks.py` is plausibility's check module from #89. `pp/` holds pprandomwalknowcast zone scores.

## Rejected options

All on main (bc134ccb); `randomwalknowcast`, WIS / log WIS.

- Share thresholds alone (2.5%, 5%, 10%, or 5% OR 5 cases): zone 1.362 / 0.187 against 1.268 / 0.174. Every full week is at least about 20% reported, so they only move the partial cut-off week (`../reliable-start/`).
- Skipping weeks below 5% and 5 cases instead of filtering them: 69 blow-up rows, zone 1.694. The walk starts from a stale week and ignores the low counts reported since (`blowdiag.py`).
- The first filter noise, 1/(1 + y) + (1 - F)/F: recovery 9.94 and zone 1.321, against 9.28 and 1.297 with 1/(F + y) (`ll`, `llnoise`).
- Main's start with only the new introductions: release 9.61, zone 1.269, ties main (`../reliable-start/cand`).
