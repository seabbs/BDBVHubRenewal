# BDBV Modeling Hub submission specification

This spec covers the round `BDBV 2026 nowcasting and forecasting` of [InsightNet-US/BDBV-Modeling-Hub](https://github.com/InsightNet-US/BDBV-Modeling-Hub).
It was derived from the hub at commit [`88478a3`](https://github.com/InsightNet-US/BDBV-Modeling-Hub/tree/88478a33b6f81dc717eec16d3ec0417c8f0d5588) (2026-10-27).
Line references are to that commit and use these short names.

- `T` is `hub-config/tasks.json`.
- `A` is `hub-config/admin.json`.
- `S` is `hub-config/model-metadata-schema.json`.
- `MO` is `model-output/README.md`.
- `MM` is `model-metadata/README.md`.
- `W` is `.github/workflows/validate-submission.yaml`.

Where the config and the README differ, the config is what CI enforces.
Behaviour marked "tested" was checked by running hubValidations 3.0.0 (the version the hub's CI installed on 2026-10-27) on the fixtures in `fixtures/`.
A pinned copy of the three config files is in `fixtures/hub-snapshot/hub-config/`.

## 1. File path and name

- Path: `model-output/<model_id>/<reference_date>-<model_id>.csv` (MO L26–36, L54–66).
- `model_id` is `<team_abbr>-<model_abbr>`, joined by one hyphen.
  The metadata file is `model-metadata/<model_id>.yml` (MO L43–45).
- `reference_date` is `YYYY-MM-DD` and must equal the `reference_date` column (MO L64, L90).
- Format: `csv` only (A L14).
- The directory name must equal the `model_id` in the file name (hubValidations `file_location`, tested).
- `team_abbr` and `model_abbr` match `^[a-zA-Z0-9_+]+$` with at most 16 characters each (S L11–26).
- One file per model per round (hubValidations `file_n`).

## 2. Columns and types

Exactly these seven columns, in any order, and no others (MO L70–80, tested with `invalid_extra_column`, `invalid_missing_column` and `valid_reordered_columns`).

| column | type | rule |
|---|---|---|
| `reference_date` | date `YYYY-MM-DD` | a single value per file, equal to the file name date |
| `target` | string | `wk inc confirmed cases` |
| `target_end_date` | date `YYYY-MM-DD` | one of the listed Saturdays |
| `location` | string | one of 546 listed codes |
| `output_type` | string | `sample` |
| `output_type_id` | integer | sample index |
| `value` | double | finite, `>= 0`, not missing |

The task id columns are `reference_date`, `location`, `target` and `target_end_date` (T L147–205).
There is no `horizon` column.

## 3. Task ids

### `reference_date` (round id)

- The round id comes from the `reference_date` column (`round_id_from_variable: true`, `round_id: "reference_date"`, T L142–143).
- Accepted values: `optional: ["2026-10-31"]` at this commit (T L148–151).
- It is the Saturday ending the epi week in which the file is submitted (MO L84–90).
- A workflow proposes the next Saturday every Saturday at 08:00 Kinshasa time and opens a PR for human review (`.github/workflows/advance-weekly-dates.yaml`).
  A file for a date not yet merged into `T` fails `round_id_valid` (tested, `invalid_unknown_round`).

### `location`

- `required: null`, `optional:` 546 codes (T L152–155).
  These are exactly the codes in `auxiliary-data/locations.csv`: `CD`, 26 provinces `CD-XX` and 519 health zones `CD-XX-NN`.
- No location is required, so any subset may be submitted (MO L128).

### `target`

- `optional: ["wk inc confirmed cases"]` (T L156–159).
- Definition: confirmed cases with symptom onset in the Sunday to Saturday epi week ending on `target_end_date` (T L224–233).
  The final revised value in `target-data/time-series.csv` is the target.
  `target_type: continuous`, `is_step_ahead: true`, `time_unit: week`.

### `target_end_date`

- `optional:` the Saturdays `2026-02-14` to `2026-11-21` (T L160–205).
- The list is extended by one week each time a reference date is added (MO L102).
- The README rule is any epi week from `2026-02-14` up to `reference_date + 21 days` (MO L102).
  The config enforces this only through the list.
  For `2026-10-31` the last listed date is exactly `reference_date + 3 weeks`.
- Not every week is required (MO L102).
- Horizon in weeks is `(target_end_date - reference_date) / 7`.
  It ranges from `-37` (2026-02-14) to `+3` for the 2026-10-31 round.

## 4. Output type `sample`

Config (T L207–221):

```json
"sample": {
  "output_type_id_params": {
    "type": "integer",
    "min_samples_per_task": 100,
    "max_samples_per_task": 100,
    "compound_taskid_set": ["reference_date", "location", "target"]
  },
  "value": {"type": "double", "minimum": 0},
  "is_required": true
}
```

- `output_type` must be `sample` (MO L130–132).
- `output_type_id` is an integer sample index (T L210, MO L136).
- Because `target_end_date` is not in `compound_taskid_set`, one sample index is one joint trajectory over all `target_end_date`s for its location (MO L138).
- Each `(reference_date, location, target)` has exactly 100 distinct sample indices (T L211–212, MO L136).
- Each sample index has exactly one row per `target_end_date` submitted for its location (MO L138).
- `value` is a non-negative double (T L215–218).
  Non-integer values pass (tested, `valid_noninteger_values`).
- No duplicate `(task ids, output_type, output_type_id)` rows (`rows_unique`).

### Sample indices in hubValidations (tested)

The README says ids may be reused across locations (`1` to `100` in each) or unique across the file (MO L138).
hubValidations 3.0.0 treats all rows sharing an `output_type_id` as one sample and infers the compound task id set from the data.
The two schemes then behave differently.

- **Reused ids** (as in the only accepted round-2 file, `OHTJHU-BM`).
  A configured compound task id that varies within a sample is dropped, so the inferred set is `[reference_date, target]`.
  hubValidations accepts this as "coarser" than the configured set.
  Each id is then a draw that is joint across locations.
  Every id must cover the identical set of `(location, target_end_date)` pairs (`spl_non_compound_tid`).
  So every location must use the same 100 ids, and every id must have the full week set for each location.
  Different locations may cover different weeks (tested, `location_week_sets_differ` is valid).
- **Unique ids per location.**
  The inferred set is the configured one.
  Every sample must cover the identical set of `target_end_date`s, so **all locations must cover the same weeks** (tested, `location_week_sets_differ_unique_ids` fails `spl_non_compound_tid`).
  This is stricter than the README.
- Mixing the two fails `spl_compound_taskid_set` (tested, `invalid_samples_101`).
- 99 samples in one location with reused ids fails `spl_non_compound_tid`, not `spl_n` (tested, `invalid_samples_99`).
- A missing week in one trajectory fails `spl_non_compound_tid` (tested, `invalid_ragged_trajectory`).

**Recommended layout:** reuse ids `1..100` in every location, use the same set of weeks for every location, and make sample `k` of `CD` the sum of sample `k` of the provinces.
This passes under either reading and matches the joint meaning hubValidations gives reused ids.

## 5. Submission window

- `submissions_due: {relative_to: reference_date, start: -6, end: -4}` (T L238–242).
- Time zone `Africa/Kinshasa` (A L15), which is WAT, UTC+1, with no daylight saving.
- hubValidations opens the window at 00:00:00 on the start day and closes it at 23:59:59 on the end day, in hub time.
  For `2026-10-31` that is Sunday 2026-10-25 00:00:00 WAT (2026-10-24 23:00 UTC) to Tuesday 2026-10-27 23:59:59 WAT (22:59:59 UTC).
- The check compares the clock at the time the CI job runs (`validate_pr(..., skip_submit_window_check = FALSE)`, W).
  A re-run or a new push after the window closes fails.
- `validate_pr` defaults to `file_modification_check = "error"` with `allow_submit_window_mods = TRUE`.
  So a submitted file can be changed or deleted only inside its window.

## 6. Model metadata

Schema (S), with `additionalProperties: true` at the top level (S L120).

Required (S L121–129):

| field | rule |
|---|---|
| `team_name` | string |
| `team_abbr` | string, `^[a-zA-Z0-9_+]+$`, at most 16 characters |
| `model_name` | string |
| `model_abbr` | string, `^[a-zA-Z0-9_+]+$`, at most 16 characters |
| `model_contributors` | array of objects with `name`, `affiliation`, `orcid` (`^\d{4}-\d{4}-\d{4}-[\dX]{4}$`), `email` (format email), `twitter`; no sub-field is required |
| `license` | one of `CC0-1.0`, `CC-BY-4.0`, `CC-BY_SA-4.0`, `PPDL`, `ODC-by`, `ODbL`, `OGL-3.0` |
| `model_details` | object with required `data_inputs` (string) and `methods` (string, at most 200 characters); optional `methods_long`; no other keys |

Optional: `model_version`, `website_url` (uri), `repo_url` (uri), `citation`, `team_funding`, `ensemble_of_hub_models` (boolean).

Checks run by `validate_model_metadata` (tested): the extension is `yml` or `yaml`, the contents match the schema, and the file name equals `<team_abbr>-<model_abbr>`.
hubValidations enforces `format: email` (tested, `invalid_bad_email`).

## 7. Round rules

- Two rounds share `round_id: reference_date` (T L4–140 and L141–243).
  Round 1, `BVBD 2026 cumulative outbreak size estimate`, is closed in practice but its reference dates stay in `T` (MO L234–250).
  Its window is −7 to +28 days (T L135–139), so round-1 dates up to 2026-10-12 still pass the window check until 2026-11-09.
- A file belongs to the round whose reference dates include its date.
  The two lists do not overlap.
- One model task per round, so `spl_mt_unique` is trivial here.

## 8. Ambiguities and inconsistencies

1. **Reused versus unique sample ids.**
   The README allows both, but with unique ids hubValidations also requires every location to cover the same weeks.
   See section 4.
2. **Joint meaning of reused ids.**
   `compound_taskid_set` says samples are independent across locations, but reusing ids makes hubValidations treat them as joint across locations.
   Nothing says whether the hub will use cross-location joint structure (for example to aggregate health zones).
3. **Sample id type is not enforced.**
   The config says `integer`, but `100.5` passes hubValidations 3.0.0 because the hub-wide `output_type_id` column is typed `double` (round 1 has quantile levels).
   `hub_validate.py` fails it as `spl_id_type`.
4. **Horizon rule is README-only.**
   "Up to three weeks after the reference date" is enforced only by the length of the `target_end_date` list.
   If the list ever runs ahead of the reference dates, later weeks would pass.
   `hub_validate.py` warns (`readme_horizon`) but does not fail.
5. **Values need not be integers.**
   The target is a count, but `value` is `double`.
6. **Target wording.**
   The target says "laboratory-confirmed cases", but the observed counts include onsets imputed by back-sampling a delay (target-data README).
7. **Timing of new reference dates.**
   The next `reference_date` is proposed on Saturday at 08:00 WAT by a bot PR that needs human review.
   If it is not merged before a submission, `round_id_valid` fails.
8. **Window check time.**
   The check uses the time the CI job runs, not the PR opening time.
   A late re-run fails, and whether maintainers merge such PRs is not stated.
9. **Name lengths.**
    MO L35 says team and model "less than 15 characters", MO L66 says "less than 16" and S allows up to 16.
    MM says team and model names under 50 characters, which S does not enforce.
    Use at most 14 characters for each abbreviation to satisfy all three.
10. **Required contributor fields.**
    MM says name, affiliation and email are required for each contributor.
    S requires none of them (tested, `valid_no_email` passes).
11. **Schema quirk.**
    In S L53 `"additionalProperties": false` sits inside the contributor `properties`.
    It is read as a property named `additionalProperties`, so contributor objects may carry any extra keys.
12. **License wording.**
    MM suggests `cc-by-4.0` in lower case, which fails the enum (tested, `invalid_license_lowercase`).
    The enum entry `CC-BY_SA-4.0` looks like a typo for `CC-BY-SA-4.0`.
13. **Metadata README leftovers.**
    MM mentions influenza hospital admissions and links to FluSight's licence list.
    These are template text.

## 9. Fixtures

`fixtures/manifest.json` lists every fixture, its expected failing checks for Python and R and, for real files, the source path and commit.

Real submissions copied from the hub:

| file | source commit | note |
|---|---|---|
| `model-output/OHTJHU-BM/2026-10-31-OHTJHU-BM.csv` | `d1ddf9d` | the only round-2 file; passed hub CI on PR #64; 62 locations, 9 weeks, reused ids |
| `model-output/epiforecasts-renewal/2026-10-12-epiforecasts-renewal.csv` | `6b9254a` | round 1, quantile, mean and median |
| `model-output/MOBS-GLEAM_BDBV/2026-06-28-MOBS-GLEAM_BDBV.csv` | `01f8a71` | round 1, columns out of order |
| `model-metadata/OHTJHU-BM.yml` | `d1ddf9d` | |
| `model-metadata/epiforecasts-renewal.yml` | `4174b51` | contributors without email |
| `model-metadata/MOBS-GLEAM_BDBV.yml` | `ae5ffd8` | block-style list |

Constructed fixtures are written by `make_fixtures.py`.
Each invalid file changes one thing in `valid_basic`.
