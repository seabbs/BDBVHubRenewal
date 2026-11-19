# Packages

The code behind the three projects, as seven packages.
{ .page-subtitle }

Each package will have its own docs site, with install, getting started, how-to pages and an API reference.
Until the packages are split out, they are modules of `bdbvhubrenewal`, and their API is in the [API reference](../reference/index.md).

| | Package | What it holds | Code and API |
|---|---|---|---|
| ![](../assets/logos/nowcastutils/logo-mark.svg){ width="40" height="40" } | **nowcastutils**{ #nowcastutils } | Releases, snapshots, location hierarchies and file formats | [Data](../reference/data.md) |
| ![](../assets/logos/nowcasteval/logo-mark.svg){ width="40" height="40" } | **nowcasteval**{ #nowcasteval } | Scores (WIS and its parts, CRPS, energy), backtests and trajectory pairing | [Evaluation](../reference/evaluation.md) |
| ![](../assets/logos/nowcastworkflow/logo-mark.svg){ width="40" height="40" } | **nowcastworkflow**{ #nowcastworkflow } | The Snakemake workflow, its modules and the command line | [CLI](../development/cli.md) |
| ![](../assets/logos/nowcastbaselines/logo-mark.svg){ width="40" height="40" } | **nowcastbaselines**{ #nowcastbaselines } | The random walk and the random walk nowcast | [Forecasters](../reference/forecasters.md), [Baselinenowcast](../reference/baselinenowcast.md) |
| ![](../assets/logos/nowcastfoundation/logo-mark.svg){ width="40" height="40" } | **nowcastfoundation**{ #nowcastfoundation } | Foundation model forecasts and fine-tuning on simulated outbreaks | [Using the fine-tuned model](../methods/finetuned-chronos.md) |
| ![](../assets/logos/nowcastrenewal/logo-mark.svg){ width="40" height="40" } | **nowcastrenewal**{ #nowcastrenewal } | The renewal models and their BVDOutbreakSize inputs | [Renewal model](../reference/model.md), [Methods](../renewal/methods.md) |
| ![](../assets/logos/bdbvhub/logo-mark.svg){ width="40" height="40" } | **bdbvhub**{ #bdbvhub } | Hub formatting, checks, the fit store and the submission | [Submission](../reference/submission.md) |

- [Setup](../setup.md): install the package and the hub data.
- [Getting started](../getting-started.ipynb): one forecast end to end.
- [Development](../development/contributing.md): contributing, adding a model and the command line.
- [Citation and licence](../citation.md).

Next: [Setup](../setup.md).
