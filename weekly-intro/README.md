# Evidence for PR #63

`final/` holds every number in the PR body, from main (a5ad8b92) and the PR head (63f0b209), made by `summarise.py`.

- `scores.csv`: recovery, release and zone backtest scores (zone as submitted).
- `silent-share.csv`: share of silent zones with a 90% upper bound above 3 (100 samples, 11-01).
- `places.csv`: named places, week 11-07 (100 samples).
- `never-case.csv` and `tails.csv`: 1000 samples, weeks 11-07 to 11-28.
- `nb-sizes.txt`: pooled NB sizes by province at 11-01.
- `rwn-places-11-01.png`: the named places, main against the PR.

`earlier-base-e334212b/` holds the comparison arms run on the previous main (e334212b): main, R rule on daily rows, R rule on weekly rows, x/F plus introductions, and ancestor estimates. `variants.py` holds the patches behind those arms.
