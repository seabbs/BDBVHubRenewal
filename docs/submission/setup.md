# Setup

Steps before the first live run.
{ .page-subtitle }

This is the checklist from `ci/SETUP.md`.
The [Workflow](workflow.md) page explains each part.

- [ ] As `seabbs-bot`, fork the hub once: `gh repo fork InsightNet-US/BDBV-Modeling-Hub --clone=false`.
- [ ] In the fork's Settings → Actions → General, disable Actions.
- [ ] Create a **classic** personal access token on `seabbs-bot` with the scopes `public_repo` and `workflow`, with an expiry (for example 90 days), and put a renewal reminder in the calendar.
- [ ] In this repository, Settings → Environments → New environment `hub-submission`: deployment branches **Selected branches**, `main` only; no required reviewers; environment secret `HUB_SUBMIT_TOKEN` set to the token.
- [ ] Settings → Actions → General → Workflow permissions: **Read repository contents**. If a ruleset covers all branches, exclude `submission-ledger`.
- [ ] Optionally set repository variables: `HUB_MODEL` (default: the latest entry of `results/model-choice.yml`), `HUB_MODEL_ID` (default: `config/hub.toml`), `HUB_FORK`, `EXPECTED_LOCATIONS` (for example `CD,CD-IT,CD-NK,CD-HU,CD-TO`), `HUB_VALIDATIONS`, `FIT_RUNNER`, and the `BDBV_CONVERGENCE_*` and `BDBV_PPC_*` overrides.
- [ ] Make the last edit to the `cron` lines in `submit.yml` yourself (`/commit --as-me`), so GitHub sends scheduled-run failures to you, not the bot. Set Settings → Notifications → Actions to "Notify me on failed workflows only".
- [ ] Ask the hub organisers (Justin Lessler or Claire Smith) to approve the first run of the hub's validation workflow on the bot's pull request, and say weekly bot pull requests will follow.
- [ ] Run a dry run from the Actions tab and read the body in the `submit` job summary.
- [ ] Time a real fit with a dry run on a past vintage (`force`, `ignore_window`, `reference_date`).
- [ ] Inside a window, with you watching, run once with `dry_run: false` and check the pull request on the hub.
- [ ] Set the repository variable `DRY_RUN` to `false`. Set it back to `true`, or disable the workflow, to pause.

Scheduled workflows in a public repository stop after 60 days without activity, so commit to `main` now and then.
GitHub-hosted runners cap a job at 6 hours.
The fit jobs time out at 350 minutes.
`ci/SETUP.md` covers runtimes, larger runners and where each part of the approach comes from (BVDOutbreakSize, cityforecasts and ecdc-weekly-growth-forecasts).
