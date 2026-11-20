# Packages

The code behind the three projects, as seven packages.
{ .page-subtitle }

Each package will have its own docs site, with install, getting started, how-to pages and an API reference.
The packages move out of `bdbvhubrenewal` one at a time.
nowcastutils, nowcasteval and nowcastbaselines are their own packages now.
The others are still modules of `bdbvhubrenewal`, and their API is in the [API reference](../reference/index.md).

| | Package | What it holds | Code and API |
|---|---|---|---|
| ![](../assets/logos/nowcastutils/logo-mark.svg){ width="40" height="40" } | [**nowcastutils**](nowcastutils.md){ #nowcastutils } | Releases, snapshots, location hierarchies, the forecast task and sample trajectories | [nowcastutils](../reference/nowcastutils.md) |
| ![](../assets/logos/nowcasteval/logo-mark.svg){ width="40" height="40" } | [**nowcasteval**](nowcasteval.md){ #nowcasteval } | Scores (WIS and its parts, CRPS, energy), backtests and trajectory pairing | [nowcasteval](../reference/nowcasteval.md) |
| ![](../assets/logos/nowcastworkflow/logo-mark.svg){ width="40" height="40" } | **nowcastworkflow**{ #nowcastworkflow } | The Snakemake workflow, its modules and the command line | [CLI](../development/cli.md) |
| ![](../assets/logos/nowcastbaselines/logo-mark.svg){ width="40" height="40" } | [**nowcastbaselines**](nowcastbaselines.md){ #nowcastbaselines } | The random walk and the random walk nowcast | [nowcastbaselines](../reference/nowcastbaselines.md) |
| ![](../assets/logos/nowcastfoundation/logo-mark.svg){ width="40" height="40" } | **nowcastfoundation**{ #nowcastfoundation } | Foundation model forecasts and fine-tuning on simulated outbreaks | [Using the fine-tuned model](../methods/finetuned-chronos.md) |
| ![](../assets/logos/nowcastrenewal/logo-mark.svg){ width="40" height="40" } | **nowcastrenewal**{ #nowcastrenewal } | The renewal models and their BVDOutbreakSize inputs | [Renewal model](../reference/model.md), [Methods](../renewal/methods.md) |
| ![](../assets/logos/bdbvhub/logo-mark.svg){ width="40" height="40" } | **bdbvhub**{ #bdbvhub } | Hub formatting, checks, the fit store and the submission | [Submission](../reference/submission.md) |

- [Setup](../setup.md): install the package and the hub data.
- [Getting started](../getting-started.ipynb): one forecast end to end.
- [Development](../development/contributing.md): contributing, adding a model and the command line.
- [Citation and licence](../citation.md).

Next: [Setup](../setup.md).
