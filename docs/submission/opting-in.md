# Opting in

The steps to start submitting a hub model.
{ .page-subtitle }

There are two hub models.
Each is opted in on its own in `config/hub.toml`.

| hub model | model | falls back to | needs |
| --- | --- | --- | --- |
| `seabbs-baseline` | persistence on a nowcast (`persistencenow`) | persistence | nothing more |
| `seabbs-renewal` | the renewal chain | its refit, `renewalsp`, `persistencenow`, persistence | a model choice in `results/model-choice.yml` |

`DRY_RUN` is the global switch.
Until it is `false`, an opted-in model is built and checked but nothing is pushed.

`task submission-preflight` checks every step it can and lists what is missing.
Run it after each part, logged in to `gh` as `seabbs`.

## Once

1. As `seabbs-bot`, fork the hub.
   `gh repo fork InsightNet-US/BDBV-Modeling-Hub --clone=false`
2. As `seabbs-bot`, turn off Actions in the fork.
   `gh api -X PUT repos/seabbs-bot/BDBV-Modeling-Hub/actions/permissions -F enabled=false`
3. As `seabbs-bot`, create a **classic** token: Settings → Developer settings → Personal access tokens → Tokens (classic).
   Scopes `public_repo` and `workflow` only, 90-day expiry.
   Fine-grained tokens cannot open pull requests on a repository the bot does not belong to.
   `workflow` is needed because each branch carries the hub's own workflow changes.
4. Restrict the `hub-submission` environment to `main`.
   `gh api -X PUT repos/seabbs/BDBVHubRenewal/environments/hub-submission -F 'deployment_branch_policy[protected_branches]=false' -F 'deployment_branch_policy[custom_branch_policies]=true'`
   `gh api -X POST repos/seabbs/BDBVHubRenewal/environments/hub-submission/deployment-branch-policies -f name=main -f type=branch`
5. Store the token in that environment.
   `gh secret set HUB_SUBMIT_TOKEN --env hub-submission -R seabbs/BDBVHubRenewal`
6. Make the default workflow token read-only.
   `gh api -X PUT repos/seabbs/BDBVHubRenewal/actions/permissions/workflow -f default_workflow_permissions=read -F can_approve_pull_request_reviews=false`
7. Set the locations each file must hold.
   `gh variable set EXPECTED_LOCATIONS -R seabbs/BDBVHubRenewal -b "CD,CD-IT,CD-NK,CD-HU,CD-TO"`
8. Commit an edit to the `cron` lines of `.github/workflows/submit.yml` as yourself (`/commit --as-me`), so failure emails reach you.
9. Tell Justin Lessler and Claire Smith that weekly pull requests will come from `seabbs-bot`.
   A maintainer must approve the hub's checks on the first one.

## For each hub model

1. Review its metadata in `hub/model-metadata/<hub model>.yml`.
2. Set `submit = true` for it in `config/hub.toml` and push to `main`.
   For `renewal`, also commit the chosen model to `results/model-choice.yml`.
3. Dry run, and read the pull request body in the job summary.
   `gh workflow run submit.yml -R seabbs/BDBVHubRenewal`
4. Inside the window (Sunday to Tuesday, Kinshasa time), do one supervised live run.
   `gh workflow run submit.yml -R seabbs/BDBVHubRenewal -f dry_run=false`
5. Go live.
   `gh variable set DRY_RUN -R seabbs/BDBVHubRenewal -b false`

To pause, set `DRY_RUN` back to `true`, set `submit = false`, or run `gh workflow disable submit.yml -R seabbs/BDBVHubRenewal`.
Renew the token before it expires (step 5 of Once).
