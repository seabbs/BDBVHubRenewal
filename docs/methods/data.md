# Data

Extraction of the hub data, its vintages, and the synthetic vintages used in testing.
{ .page-subtitle }

What the data show is on the [Data results page](../data.md).

## Target and notation

The target is $y_{p}(w)$, the final count of confirmed cases in province $p$ with symptom onset in epi week $w$ (Sunday to Saturday, labelled by its Saturday).
It includes onsets the hub imputes for cases reported without one.
A vintage $v$ is the data published on date $T_v$.
It gives, for each $p$ and $w$, known onsets $K_{p,v}(w)$, imputed onsets $I_{p,v}(w)$ and their total $Y_{p,v}(w) = K_{p,v}(w) + I_{p,v}(w)$.
A forecast made from vintage $v$ serves the round with reference date $r_v$, the first Saturday whose submission window (Sunday to Tuesday, Kinshasa time) closes on or after $T_v$.
Horizon $h$ is the number of weeks from $r_v$ to $w$.
Backtest forecasts cover $h = -4, \dots, 3$.
The national value is the sum over provinces, sample by sample.

## Extraction

`task data` downloads the hub files and writes the tidy tables to `data/`.
The tidy tables are small and committed.
Raw downloads go to `data/raw/` and are not.
`data/sources.json` records the hub commit and where each delay vintage came from.

| Hub file | Tidy table | What it holds | Vintages |
|---|---|---|---|
| `target-data/time-series.csv` | `target.csv` | Weekly confirmed cases by onset week for the country and provinces: observed, imputed and total. | All, in one file |
| `auxiliary-data/daily-confirmed-case.csv` | `daily.csv` | Daily cases by onset day, split into observed and imputed onsets. | All, in one file |
| `auxiliary-data/weekly-reporting-delays-by-province.csv` | `delays.csv.gz` | One row per known-onset case: province, onset week and onset-to-report delay. | Latest only; earlier ones from the archive folder and the git history |
| `target-data/time-series.csv` (health zones) | `health_zones.csv.gz` | Weekly totals by health zone. Not modelled yet. | All |
| `auxiliary-data/locations.csv` | `locations.csv` | Codes, names and parents, plus populations and centres added here. | Static |
| `hub-config/*.json` | `hub-config/` | Rounds, tasks, output types and the metadata schema. | Copied at download |

The split into observed and imputed in `target.csv` comes from the daily file summed into epi weeks.
It matches the published weekly total in every row.
The delay rows match the observed (known-onset) totals of their vintage exactly.

**Not published by the hub:** deaths, suspected cases, counts by report date, cases without an onset by report date, the imputation code and populations.
Province populations come from the Institut National de la Statistique, *Annuaire statistique RDC 2020* (2019 figures), via [Wikipedia](https://en.wikipedia.org/wiki/Provinces_of_the_Democratic_Republic_of_the_Congo), as in BVDOutbreakSize.
Centres of affected provinces are the WorldPop-weighted centres used by BVDOutbreakSize.

## Vintages

Every hub data release is kept as a vintage, labelled by its `as_of` date.
Recent weeks in a vintage are incomplete and are revised upwards in later vintages.
So every forecast is made from one vintage and scored against a later one.
Every vintage carries a `kind` (`real`, `reconstructed` or `simulated`), and every score and plot shows it.

## Delay and missingness data

The delay file gives one row per case with a known onset: province, onset week and onset-to-report delay in days.
It is published for the latest vintage only.
Earlier delay vintages come from the hub's archive folder and git history.
The 2026-08-18 vintage has none.

The target data split each week's cases into known onsets and imputed onsets.
The hub imputes an onset for each case reported without one, by subtracting a delay drawn from the pooled published delays ([hub issue 61](https://github.com/InsightNet-US/BDBV-Modeling-Hub/issues/61)).
So the share of cases without an onset is measured at every vintage.
The [Model page](model.md#missing-onsets) uses both.

## Reconstructed vintages

There are only four real vintages.
Reconstructed vintages rebuild, for any past Saturday $A$, what the hub would roughly have shown, from the latest real vintage.
Cases reach a vintage some days after report, so the cut-off is $A' = A - 3$ days.

- **Known onsets.** Each delay row is a case with onset week $W$ and delay $d$.
  Its onset day is uniform in the week, so it counts as reported by $A'$ with probability $\frac{1}{7}\sum_{u=0}^{6} \mathbb{1}\{W - u + d \le A'\}$.
- **Imputed onsets.** Their report dates are not public.
  The hub imputed each onset by subtracting a delay drawn from the pooled delays.
  So a case imputed to week $W$ counts as reported by $A'$ with probability $\frac{1}{7}\sum_{u} \hat G(A' - W + u)$, with $\hat G$ the pooled delay cdf.
- **Delay data.** Rows whose report (taking the onset mid-week) falls by $A'$.

The 3-day lag gives the smallest net difference from the real 2026-08-18, 2026-10-09 and 2026-10-17 vintages over their last four weeks (see [Data](../data.md#reconstructed-vintages)).
Reconstructed vintages miss real revisions, onsets added after first report and changes in the hub's imputation.
So they are cleaner than real vintages, and scores on them are somewhat optimistic.

## Simulated vintages

A simulator generates province outbreaks from the renewal model's own structure with known parameters:

- a renewal process with a national Rt that turns from 1.6 to 0.85 around day 120, fixed province offsets, gravity-kernel importation ($\varepsilon = 0.01$) and susceptible depletion;
- ascertainment rising from 0.5 to 0.75;
- a missing-onset share rising from 5% to 25% and falling to 12%;
- a Gamma delay whose mean falls from 12 to 6 days.

It then publishes vintages as the hub does: known onsets by onset week, missing onsets given an onset by a delay drawn from the known delays reported so far, their total, and the delay line data.
The truth is every confirmed case by its true onset week.
Recovery checks run on these vintages.
