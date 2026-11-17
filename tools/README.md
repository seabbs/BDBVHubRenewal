# Optional tools

Helpers for running long jobs detached on a shared host. Nothing in the
workflow (`workflow/Snakefile`) or the paper (`task paper`) needs them.

- `run_target.sh NAME TARGET...`: run paper workflow targets in the
  background, single-threaded and niced, logging to `PAPER_LOGS/NAME.log`.
- `paper_local.sh [TARGET]`: wait for the held-out inputs, then build the
  paper in the background.

Each prints its progress. Setting `NOTIFY` and `MAILBOX` in
`config/local.env` also appends those lines to a local notes file
(`workflow/scripts/notify.sh`); by default nothing is written there.
