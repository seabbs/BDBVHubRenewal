# Paper forecast schema

Every study writes its forecasts to one long table in this schema.
The scores and figures read only this table and the observations table below.
`paperfigs.schema.validate_forecasts` and `paperfigs.schema.validate_observations` check a table before use.

## Forecasts

One row per model, location, origin date, target date and quantile level or sample.
A forecast is either a set of quantiles or a set of samples.
Germany uses quantiles; synthetic and Ebola use samples.

| Column | Type | Meaning |
|---|---|---|
| `study` | str | One of `synthetic`, `germany` or `ebola`. |
| `model` | str | Model name, such as `randomwalknowcast` or `fm_chronos_daily_ragged_ft`. |
| `location` | str | Location code within the study. |
| `stratum` | str | Optional. Subgroup within a location, such as a German age group (`00-04`, ..., `80+`). Defaults to `all` when the column is absent. |
| `origin_date` | date | Date the forecast was made, using data as of this date. |
| `target_date` | date | Date the forecast is for (the onset day, or the start of the week). |
| `horizon` | int | `target_date - origin_date` in the study's one time step (days for `synthetic` and `germany`, weeks for `ebola`). Zero or negative is a nowcast. |
| `quantile_level` | float | In (0, 1) for a quantile forecast; empty for a sample forecast. |
| `sample_id` | int | Sample index for a sample forecast; empty for a quantile forecast. A sample is a joint draw across the targets of one model, location and origin. |
| `value` | float | Predicted quantile or sample, at least zero. |
| `truth` | float | Count as reported D days after the target date (see Truth), at least zero. Empty if not yet known. |

Rules the validator checks:

- all columns present, with no extra columns;
- `study` is one of the three studies;
- each row has exactly one of `quantile_level` or `sample_id`, and every row of a forecast uses the same form;
- `value` is at least zero;
- `truth` is the same on every row of a forecast;
- sample forecasts have no duplicate `sample_id`;
- quantile forecasts have levels in (0, 1), the level 0.5, no duplicate levels, and quantiles that do not decrease with level.

Quantile forecasts should carry the levels for the study's coverage widths: 0.025, 0.25, 0.5, 0.75 and 0.975 for germany.
The WIS uses every level present, so it is only comparable between models with the same levels.
Tables are written as CSV (gzip is fine) with ISO dates.

## Truth

Following the German nowcast hub (Wolffram et al. 2023), the truth for a target is the count in the latest release on or before `target_date + D` days.
It is missing until a release on or after that date exists.
`paperfigs.truth.rolling_truth(observations, delay_days=D)` builds it from release vintages, and `paperfigs.truth.attach_truth` sets the `truth` column.
`delay_days` is a number of days, or a series indexed by location for a D per location; the synthetic study uses each outbreak's 99% reporting delay (`delay_q99` in its factors table).

## Observations

Data as of each origin date, for the nowcast panels.
One row per study, location, target date and as-of date.

| Column | Type | Meaning |
|---|---|---|
| `study` | str | As above. |
| `location` | str | As above. |
| `stratum` | str | As above; optional, default `all`. |
| `target_date` | date | Reference date of the count. |
| `as_of_date` | date | Date the data were available. |
| `value` | float | Count as reported by `as_of_date`, at least zero. |

The latest `as_of_date` for a study is the final data.

## Scores

`paperfigs.scores.score_forecasts` gives one row per forecast with known truth (the key columns and `truth`) with:

- `metric`, `wis` for quantile forecasts and `crps` for sample forecasts;
- `score`, the WIS or the CRPS on the natural scale;
- `log_score`, the same after `log1p` of the forecast and truth (Bosse et al. 2023);
- `bias`, scoringutils' bias in [-1, 1], positive for overprediction, using the count form for sample forecasts in every study (`STUDY_COUNTS`);
- `coverage_<width>`, whether the truth is in the central interval (from linear sample quantiles for samples). Widths are set per study in `STUDY_COVERAGE`: 50% and 95% for germany, whose hub has no 0.05 or 0.95 level, and 50% and 90% otherwise.

These match scoringutils 2.2 defaults, checked by `tests/test_scoringutils.py` against `tests/scoringutils_reference.R`.

`paperfigs.scores.relative_scores` divides each model's mean score by the baseline's on the forecasts both made (matched on location, stratum, origin, target and horizon), within each group.
The baseline is set per study in `STUDY_BASELINES`: `randomwalknowcast` for synthetic, `KIT-frozen_baseline` for germany and `BDBVHub-baseline` for Ebola.
