# Setup

Steps before the first live submission.
{ .page-subtitle }

Dry runs need none of this.
Follow the steps in order.
`task submission-preflight` checks every step it can read and prints what is still missing.
Run it after each part, logged in to `gh` as `seabbs`, since some settings are admin-only reads.
The [Workflow](workflow.md) page explains the pipeline these steps unlock.

## Why each piece is needed

- **A fork.** Neither `seabbs` nor `seabbs-bot` can push to the hub, so each round's branch goes to a fork. The branch is cut from the hub's `main`, so the fork is never synced.
- **A classic token on `seabbs-bot`.** The workflow's own `GITHUB_TOKEN` only works on this repository. It cannot push to the fork or open a pull request on the hub. Fine-grained tokens cannot contribute to public repositories the account is not a member of.
- **`public_repo` scope.** It pushes to the fork and opens the pull request on the hub.
- **`workflow` scope.** The pushed branch carries the hub's latest commits. When those change `.github/workflows/` (the hub changes them often), GitHub refuses the push without this scope.
- **The bot account.** A leaked token reaches only the bot, and the pull requests are marked as automated.
- **The `hub-submission` environment.** Only jobs on `main` can read the token, so a pull request to this repository never sees it.

## Checklist

### On `seabbs-bot`

1. Fork the hub once.
   `gh repo fork InsightNet-US/BDBV-Modeling-Hub --clone=false`
2. Turn off Actions in the fork, so the hub's workflows never run there.
   `gh api -X PUT repos/seabbs-bot/BDBV-Modeling-Hub/actions/permissions -F enabled=false`
3. Create the token at Settings → Developer settings → Personal access tokens → **Tokens (classic)** → Generate new token (classic).
   Note `bdbv-hub-submission`.
   Expiry 90 days.
   Scopes `public_repo` and `workflow` only.
   Put a renewal reminder in the calendar for day 80.
   The preflight warns from day 80, and an expired token fails the `submit` job.

### On `seabbs/BDBVHubRenewal` (as `seabbs`)

4. Restrict the `hub-submission` environment to `main`.
   It already exists but any branch can deploy to it.
   Settings → Environments → `hub-submission` → Deployment branches → **Selected branches and tags** → add `main`.
   Leave required reviewers and the wait timer off, since the gate is the approval and scheduled runs would otherwise stall.
   From the command line:
   `gh api -X PUT repos/seabbs/BDBVHubRenewal/environments/hub-submission -F 'deployment_branch_policy[protected_branches]=false' -F 'deployment_branch_policy[custom_branch_policies]=true'`
   then
   `gh api -X POST repos/seabbs/BDBVHubRenewal/environments/hub-submission/deployment-branch-policies -f name=main -f type=branch`
5. Add the token as an environment secret, not a repository secret.
   `gh secret set HUB_SUBMIT_TOKEN --env hub-submission -R seabbs/BDBVHubRenewal`
   (it prompts for the value).
6. Make the default workflow token read-only.
   Only the `record` job asks for more (`contents: write` to push the `submission-ledger` branch, `actions: write` to rebuild the docs).
   `gh api -X PUT repos/seabbs/BDBVHubRenewal/actions/permissions/workflow -f default_workflow_permissions=read -F can_approve_pull_request_reviews=false`
   There are no rulesets now. If one is added that covers all branches, exclude `submission-ledger`.
7. Set the locations every submission must hold, so a file missing a province fails the checks.
   Use the locations in the latest dry run's file.
   `gh variable set EXPECTED_LOCATIONS -R seabbs/BDBVHubRenewal -b "CD,CD-IT,CD-NK,CD-HU,CD-TO"`
8. Own the schedule.
   GitHub emails scheduled-run failures only to whoever last changed the `cron` lines in `.github/workflows/submit.yml`, and that is `seabbs-bot` now.
   Make a small edit to those lines (a comment is enough), commit it with `/commit --as-me` and push to `main`.
   Then set Settings → Notifications → System → Actions to "Notify me on failed workflows only".
9. Leave `HUB_MODEL`, `HUB_MODEL_ID` and `HUB_FORK` unset.
   The model is the latest entry of `results/model-choice.yml` and the hub id comes from `config/hub.toml` (`epiforecasts-seabbsrenewal`).
   `HUB_VALIDATIONS`, `FIT_RUNNER` and the `BDBV_CONVERGENCE_*` and `BDBV_PPC_*` overrides are optional ([`ci/SETUP.md`](https://github.com/seabbs/BDBVHubRenewal/blob/main/ci/SETUP.md)).

### Before going live

10. Run `task submission-preflight` and clear every `MISSING` line.
    The expected warnings are `mode` (dry run) and, until step 12, `model choice`.
11. Review the model metadata in `hub/model-metadata/epiforecasts-seabbsrenewal.yml` (team, model name, contributors, licence and methods text).
    The first pull request adds it to the hub, written for whichever model is chosen.
12. Review the held-out results and commit the model to submit as a dated entry in `results/model-choice.yml`.
    Until then the workflow submits persistence under the hub id, and the preflight warns.
13. Tell the hub organisers (Justin Lessler and Claire Smith) that weekly pull requests will come from `seabbs-bot` as `epiforecasts-seabbsrenewal`.
    The first one is a first-time contribution from a fork, so a maintainer must click **Approve and run** on the hub's validation workflow.
14. Dry run with the current data.
    `gh workflow run submit.yml -R seabbs/BDBVHubRenewal`
    Read the pull request body in the `submit` job summary and the `gate` artefact.
15. Time a real fit on a past round.
    `gh workflow run submit.yml -R seabbs/BDBVHubRenewal -f force=true -f ignore_window=true -f reference_date=2026-10-31`
    Read the `Fit took N s` line in the fit job summary.
    Each fit takes about 2.7 to 3.4 hours on a hosted runner.
    With `ignore_window` the fit jobs stop at 350 minutes and the retry always runs if needed.
    If the measured times differ, update `retry_minutes` in the `[timing]` table of `ci/policy.toml`.
16. Do one supervised live run inside the window (Sunday 00:00 to Tuesday 23:59 Kinshasa time, UTC+1).
    Start it on Sunday or Monday, so the whole chain has time to run.
    Each fit job stops 45 minutes before the close, and the retry runs only if at least 285 minutes are left after the main fit (`[timing]` in `ci/policy.toml`).
    `gh workflow run submit.yml -R seabbs/BDBVHubRenewal -f dry_run=false`
    Check the pull request on the hub, its validation run, and the round's page linked from it.
17. Go live.
    `gh variable set DRY_RUN -R seabbs/BDBVHubRenewal -b false`
    Scheduled runs then open or update the round's pull request.
    To pause, set `DRY_RUN` back to `true` or run `gh workflow disable submit.yml -R seabbs/BDBVHubRenewal`.

## Keeping it running

- Renew the token before it expires and update the secret (step 5).
- Scheduled workflows in a public repository stop after 60 days without activity, so commit to `main` now and then. The preflight reports a disabled workflow.
- GitHub-hosted runners cap a job at 6 hours. Each fit job times out at 350 minutes, or 45 minutes before the window closes if that is sooner.

`ci/SETUP.md` covers runtimes, larger runners and where each part of the approach comes from (BVDOutbreakSize, cityforecasts and ecdc-weekly-growth-forecasts).
