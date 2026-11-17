# Reporting fixtures

Small score tables the report tests read.

`nowcast-scores.csv` and `zone-scores.csv` keep no samples.
Their `log_crps` column is a stand-in set equal to `log_wis` when the pages moved to CRPS.
It is not a computed score, so read it only as test input.
