# Sam's rule for empty recent weeks, tested on #63

- `A`: #63 at 63f0b209 (tables also in `../final/`, as `pr`).
- `B`: each place walks from its latest full week with a report, with the weekly x/F nowcast up to it and no +1; silent places start introductions after the latest full week; pooled spread after an introduction (`arm-B.patch`).
- `B0`: B with spread 0 after an introduction (`variants.py`, `sig0`).
- `C`: B with the +1 rule kept for the nowcast weeks (`variants.py`, `plus1`).

`scores.csv`, `silent-share.csv`, `never-case.csv`, `tails.csv` and `places.csv` cover main, A, B, B0 and C, made by `summarise.py`.
