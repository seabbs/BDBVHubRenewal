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

## Extraction

| Hub file | What it holds | Vintages |
|---|---|---|
| `target-data/time-series.csv` | Weekly confirmed cases by onset week, by province and nationally. Also health zone totals. | All, in one file |
| `auxiliary-data/daily-confirmed-case.csv` | Daily cases by onset day, observed and imputed, nationally and by province and health zone. The observed and imputed split of the weekly target comes from here. | All, in one file |
| `auxiliary-data/weekly-reporting-delays-by-province.csv` | One row per known-onset case: province, onset week and onset-to-report delay. | Latest only; earlier ones from the archive folder and git history |
| `auxiliary-data/locations.csv` | Location codes, names, types and parents. | Static |
| `auxiliary-data/health-zone-code-mapping.csv` | Health zone codes mapped to P-codes and DHIS2 codes. | Static |
| `hub-config/*.json` | Rounds, tasks, output types, hub settings and the metadata schema. | Copied at download |

The hub does not publish deaths, suspected cases, counts by report date or populations.
Province populations are the Institut National de la Statistique's 2019 figures (*Annuaire statistique RDC 2020*), and province centres the WorldPop-weighted centres of BVDOutbreakSize.

## Vintages

Each hub release is a vintage, labelled by its `as_of` date and of kind `real` or `simulated`.
From the 2026-11-01 release on, `as_of` is the extraction date, a day after the last reported case.
The hub states that earlier releases were labelled by the last reported case ([hub issue 90](https://github.com/InsightNet-US/BDBV-Modeling-Hub/issues/90)).
We keep each release's published label and map a rewritten label back to it.
A release's data cut-off is its last report day, as its data prove it.
A case with onset on day $t$ was reported on or after $t$, and a delay row with onset week ending $W$ and delay $d$ was reported on or after $W - 6 + d$.
The cut-off of a real release is the latest of these days over its daily onsets and delay rows.
This gives 2026-10-08 for the release labelled 2026-10-09 and 2026-10-16 for the release labelled 2026-10-17, so the data do not bear out the hub's description of those labels.
Between releases 66 to 83 cases with a known onset were added per day, so a day with no report is unlikely, and a cut-off on the label would add a day with no reports to each of these releases.
A simulated release states its cut-off.
The models, the scoring and the zone stage all use this cut-off.
Recent weeks are revised in later releases, so each forecast is made from one vintage and scored on the counts later releases report.

## Release snapshots

`auxiliary-data/daily-confirmed-case.csv` holds, for each release, the daily onsets known at that release.
The count of a day is its known plus imputed onsets, and the counts of a week sum to the hub's weekly target.
A release has an `as_of` date and a data cut-off ([Vintages](#vintages)).
A release observes the days up to its cut-off, and a day in range with no row counts 0.
The age of day $t$ at release $r$ is the cut-off minus $t$, the onset clock.
Counts of a day are compared across releases on this clock.
A release and the next release give one pair of releases.
Its step is the difference of the two cut-offs in days.

## Delay and missingness data

The hub imputes an onset for each case reported without one by subtracting a delay drawn from the pooled published delays ([hub issue 61](https://github.com/InsightNet-US/BDBV-Modeling-Hub/issues/61)).
So the share of cases without an onset is measured at every vintage.

## Simulated vintages

Simulated vintages come from the model's own renewal with known parameters.
The default scenario uses:

- expected onsets from the model's renewal and generation interval, with a national Rt that turns from 1.6 to 0.85 around day 120, fixed province offsets, gravity-kernel importation ($\varepsilon = 0.01$) and susceptible depletion;
- Poisson confirmed cases, with ascertainment rising from 0.5 to 0.75;
- a missing-onset share by report day rising from 5% to 25% and falling to 12%;
- a double censored Gamma delay whose mean falls from 12 to 6 days;
- a lag from report to publication, 85% uniform on 0 to 3 days and 15% uniform on 4 to 60 days, as late entries in the real delay files;
- 30% of cases reported with no onset get one 7 to 42 days after publication, so imputed counts of past weeks fall and known counts rise, as between real vintages.

Vintages are published as the hub does, with missing onsets imputed from the known delays reported so far.
Daily releases are published on Tuesdays, Wednesdays, Fridays and Saturdays.
Their weekly sums are the weekly target.
Each is cut off by the lag of the latest real release on that weekday, or the median lag of the real releases where none fell on that weekday.
The truth is every confirmed case by its true onset week.

The calibrated scenarios copy one real vintage.
Each province's expected onsets follow its real weekly totals, spread over the days and smoothed over 7 days, up to 4 weeks before the vintage.
Rt is the ratio of onsets to the force of infection, capped at 3, and onsets above what it explains are seeds.
From 4 weeks before the vintage, Rt is the mean of its last 3 weeks times 0.75, 1 or 1.25 (falling, flat or rising), for 10 weeks past the vintage.
Ascertainment is 1, there is no importation and the whole population is at risk.
The delay is the real delay by onset week, completed for right truncation from weeks with full follow-up.
The missing-onset share is the real national imputed share by onset week, $(I + 1)/(Y + 2)$, averaged over 5 weeks and moved later by the mean delay.
Zones take the real share of each province's cases.
Late entries and onset backfill are as in the default, and 2% of cases are removed 7 to 42 days after publication.
Their truth is the last simulated vintage.

## BVDOutbreakSize estimates

`renewalbvd` uses the Rt estimates in the public `results-<run>` releases of [BVDOutbreakSize](https://github.com/epiforecasts/BVDOutbreakSize) (MIT licence, [Zenodo 10.5281/zenodo.20312758](https://doi.org/10.5281/zenodo.20312758)).
These are the national $R_T$ draws, the patch summaries from 23 October with the national $R_T$ row they are relative to, and the release's data cut-off.
Early releases give only the growth rate $r$, which is turned into $R_T = (1 + r\theta)^k$ with BVDOutbreakSize's Gamma generation interval (mean 15.3 days, SD 9.3 days, shape $k$, scale $\theta$).
Each forecast uses the newest release whose files were all last updated by the forecast's cut-off, whose data end by the forecast date and whose fit converged.
The cut-off is the end of the Sunday that opens the round's submission window, Kinshasa time, or the end of the vintage date if that is later.
The joint fit's worst R-hat must be at most 1.1 (`[bvd] max_rhat` in `config/backtest.toml`), and effective sample sizes are ignored.
The R-hat is read from the fit diagnostics table in the release's rendered site (`site.zip`), and a release without a readable table is passed over for the next older one.
Each fit records the release it used, its R-hat and why newer releases were skipped.
Their use is on the [Renewal models page](model.md#bvdoutbreaksize-informed).
