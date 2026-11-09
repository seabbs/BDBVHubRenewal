<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img src="assets/logo.svg" alt="BDBVHubRenewal" width="275" height="44">
  </picture>
</h1>

**Forecasts for the BDBV Modeling Hub.**

[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://samabbott.co.uk/BDBVHubRenewal/)
[![Pages](https://github.com/seabbs/BDBVHubRenewal/actions/workflows/pages.yml/badge.svg)](https://github.com/seabbs/BDBVHubRenewal/actions/workflows/pages.yml)
[![codecov](https://codecov.io/gh/seabbs/BDBVHubRenewal/branch/main/graph/badge.svg)](https://codecov.io/gh/seabbs/BDBVHubRenewal)
[![licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/1435982169.svg)](https://doi.org/10.5281/zenodo.23942616)

| Component | Tests | Lint | Coverage |
| --- | --- | --- | --- |
| Package (`src/`) | [![Tests](https://github.com/seabbs/BDBVHubRenewal/actions/workflows/test.yml/badge.svg)](https://github.com/seabbs/BDBVHubRenewal/actions/workflows/test.yml) | [![Lint](https://github.com/seabbs/BDBVHubRenewal/actions/workflows/lint.yml/badge.svg)](https://github.com/seabbs/BDBVHubRenewal/actions/workflows/lint.yml) | [![package coverage](https://codecov.io/gh/seabbs/BDBVHubRenewal/branch/main/graph/badge.svg?flag=package)](https://codecov.io/gh/seabbs/BDBVHubRenewal?flags%5B0%5D=package) |
| Submission CI (`ci/`) | [![Submission CI tests](https://github.com/seabbs/BDBVHubRenewal/actions/workflows/submission-ci-tests.yml/badge.svg)](https://github.com/seabbs/BDBVHubRenewal/actions/workflows/submission-ci-tests.yml) | [![Lint](https://github.com/seabbs/BDBVHubRenewal/actions/workflows/lint.yml/badge.svg)](https://github.com/seabbs/BDBVHubRenewal/actions/workflows/lint.yml) | [![ci coverage](https://codecov.io/gh/seabbs/BDBVHubRenewal/branch/main/graph/badge.svg?flag=ci)](https://codecov.io/gh/seabbs/BDBVHubRenewal?flags%5B0%5D=ci) |

The [BDBV Modeling Hub](https://github.com/InsightNet-US/BDBV-Modeling-Hub) collects weekly nowcasts and short-term forecasts of confirmed Bundibugyo virus disease cases in the 2026 outbreak in the Democratic Republic of the Congo.
Participating teams submit sample trajectories of cases by week of symptom onset, nationally, by province and by health zone.
The hub is run by Insight Net centres (ACCIDDA and epiEngage), the Kinshasa School of Public Health and the Institut National de Santé Publique.

We submit two models to the hub.
`seabbs-rwnowcast` is a baseline forecast adjusted for reporting delays.
`seabbs-renewal` is a simple renewal model that extends our earlier modelling of this outbreak in [BVDOutbreakSize](https://github.com/epiforecasts/BVDOutbreakSize).
We compare both with models that lack that input, a random walk with no delay adjustment and renewal models without the BVDOutbreakSize Rt.
The package, `bdbvhubrenewal`, fits the models, scores them and writes the hub submission files.

`seabbs-rwnowcast` is `randomwalknowcast`, baselinenowcast ([Johnson et al. 2025](https://doi.org/10.12688/wellcomeopenres.25027.2)) ported to JAX, with a random walk on top.
It runs by province and health zone, and the national value is the sum of the provinces.
`randomwalk` is the same walk with no nowcast.
The renewal models are built on [PyRenew](https://github.com/CDCgov/PyRenew) and model daily symptom onsets, with a learnt onset-to-report delay and the hub's imputation of missing onsets.
`renewal` fits the provinces together, with a pooled Rt and importation between provinces through a gravity kernel.
`renewalsp` fits the same model to each province on its own, with no pooling or importation.
`renewalbvd` is `renewal` with the BVDOutbreakSize Rt estimates added to the fit.
`seabbs-renewal` submits the first model in the chain in `ci/policy.toml` that passes its checks, starting with `renewalbvd`.
See the [documentation](https://samabbott.co.uk/BDBVHubRenewal/) for the methods and results.

## Installation

Install the package and its dependencies with [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/seabbs/BDBVHubRenewal
cd BDBVHubRenewal
uv sync
```

## Usage

The [Getting started](https://samabbott.co.uk/BDBVHubRenewal/getting-started/) page forecasts, plots and scores both baselines, `randomwalknowcast` and `randomwalk`, in Python.

## Data

The surveillance data are from the Institut National de Santé Publique (INSP) and the Incident Management Support Team (IMST), Democratic Republic of the Congo.
They are processed by the Atlantic Coast Center for Infectious Disease Dynamics and Analytics (ACCIDDA) and the Kinshasa School of Public Health, University of Kinshasa, and distributed through the [BDBV Modeling Hub](https://github.com/InsightNet-US/BDBV-Modeling-Hub).

## Contributing

Contributions are welcome.
See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citing

[![DOI](https://zenodo.org/badge/1435982169.svg)](https://doi.org/10.5281/zenodo.23942616)

Abbott S (2026). BDBVHubRenewal: renewal-model nowcasts and forecasts for the BDBV Modeling Hub. Zenodo. <https://doi.org/10.5281/zenodo.23942616>.

```bibtex
@software{abbott_bdbvhubrenewal,
  author = {Abbott, Sam},
  title = {{BDBVHubRenewal}: renewal-model nowcasts and forecasts for the {BDBV} Modeling Hub},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.23942616},
  url = {https://doi.org/10.5281/zenodo.23942616}
}
```

The DOI covers all versions, and each release has its own DOI on Zenodo.
The metadata are in [CITATION.cff](CITATION.cff), which GitHub shows as the "Cite this repository" button.

## Funding

This study is funded by the National Institute for Health and Care Research (NIHR) Health Protection Research Unit in Health Analytics & Modelling, a partnership between the UK Health Security Agency, Imperial College London and the London School of Hygiene & Tropical Medicine (grant code NIHR207404).
The views expressed are those of the author(s) and not necessarily those of the NIHR, UK Health Security Agency or the Department of Health and Social Care.

## Licence

The code is under the MIT licence, see [LICENSE](LICENSE).
