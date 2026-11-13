# Paper forecast schema

Every study writes its forecasts to one long table in this schema.
The scores and figures read only this table and the observations table below.
`paperfigs.schema.validate_forecasts` and `paperfigs.schema.validate_observations` check a table before use.

## Forecasts

One row per model, location, origin date, target date and quantile level.

| Column | Type | Meaning |
|---|---|---|
| `study` | str | One of `synthetic`, `germany` or `ebola`. |
| `model` | str | Model name, such as `randomwalknowcast` or `fm_chronos_daily_ragged_ft`. |
| `location` | str | Location code within the study. |
| `origin_date` | date | Date the forecast was made, using data as of this date. |
| `target_date` | date | Date the forecast is for. |
| `horizon` | int | `target_date - origin_date` in the study's time step (days for `synthetic` and `germany`, weeks for `ebola`). Zero or negative is a nowcast. |
| `quantile_level` | float | In (0, 1). |
| `value` | float | Predicted quantile, at least zero. |
| `truth` | float | Final observed value for the target, at least zero. Missing if not yet known. |

Rules the validator checks:

- all columns present, with no extra columns;
- `study` is one of the three studies;
- `quantile_level` is in (0, 1) and `value` is at least zero;
- `truth` is the same on every row of a forecast;
- each forecast has the level 0.5 and no duplicate levels;
- quantiles do not decrease with level within a forecast.

Each forecast should carry the levels 0.05, 0.25, 0.5, 0.75 and 0.95, so that 50% and 90% coverage can be scored.
The scores use every level present, so the WIS is only comparable between models with the same levels.

## Observations

Data as of each origin date, for the nowcast panels.
One row per study, location, target date and as-of date.

| Column | Type | Meaning |
|---|---|---|
| `study` | str | As above. |
| `location` | str | As above. |
| `target_date` | date | Reference date of the count. |
| `as_of_date` | date | Date the data were available. |
| `value` | float | Count as reported by `as_of_date`, at least zero. |

The latest `as_of_date` for a study is the final data.

## Scores

`paperfigs.scores.score_forecasts` gives one row per forecast (all columns above but `quantile_level` and `value`) with:

- `wis`, the quantile weighted interval score on the natural scale;
- `log_wis`, the same after `log1p` of quantiles and truth (Bosse et al. 2023);
- `bias`, the scoringutils quantile bias in [-1, 1], positive for overprediction;
- `coverage_50` and `coverage_90`, whether the truth is in the central interval.

`paperfigs.scores.relative_wis` divides each model's mean WIS by the baseline's (default `randomwalknowcast`) on the forecasts both made, within each group.
