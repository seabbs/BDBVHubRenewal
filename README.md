# Stored German forecasts of the fine-tuned Chronos-2 model

Written by `scripts/germany_ft_store.py` on main from the checkpoint
scorer's cache, and read by the Pages build (`bdbvhubrenewal
germany-ft-load`). One directory per checkpoint release tag, with
`forecasts.csv.gz` (`germany-backtest`'s forecast columns,
`fm_chronos_daily_ragged_ft` and `fm_chronos_weekly_ragged_ft` only)
and `manifest.json`. Development nowcast dates only: nothing on or after
the split date in
config/germany.toml.
