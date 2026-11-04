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

| Component | Tests | Lint | Coverage |
| --- | --- | --- | --- |
| Package (`src/`) | [![Tests](https://github.com/seabbs/BDBVHubRenewal/actions/workflows/test.yml/badge.svg)](https://github.com/seabbs/BDBVHubRenewal/actions/workflows/test.yml) | [![Lint](https://github.com/seabbs/BDBVHubRenewal/actions/workflows/lint.yml/badge.svg)](https://github.com/seabbs/BDBVHubRenewal/actions/workflows/lint.yml) | [![package coverage](https://codecov.io/gh/seabbs/BDBVHubRenewal/branch/main/graph/badge.svg?flag=package)](https://codecov.io/gh/seabbs/BDBVHubRenewal?flags%5B0%5D=package) |
| Submission CI (`ci/`) | [![Submission CI tests](https://github.com/seabbs/BDBVHubRenewal/actions/workflows/submission-ci-tests.yml/badge.svg)](https://github.com/seabbs/BDBVHubRenewal/actions/workflows/submission-ci-tests.yml) | [![Lint](https://github.com/seabbs/BDBVHubRenewal/actions/workflows/lint.yml/badge.svg)](https://github.com/seabbs/BDBVHubRenewal/actions/workflows/lint.yml) | [![ci coverage](https://codecov.io/gh/seabbs/BDBVHubRenewal/branch/main/graph/badge.svg?flag=ci)](https://codecov.io/gh/seabbs/BDBVHubRenewal?flags%5B0%5D=ci) |

Forecasts of weekly confirmed Bundibugyo virus disease cases in the Democratic Republic of the Congo for the [BDBV Modeling Hub](https://github.com/InsightNet-US/BDBV-Modeling-Hub).
Forecasts are made nationally, by province and by health zone.
The package, `bdbvhubrenewal`, fits the models, scores them and writes the hub submission files.
Two hub models are submitted, `seabbs-rwnowcast` and `seabbs-renewal`.
`seabbs-rwnowcast` is `randomwalknowcast`, baselinenowcast ([Johnson et al. 2025](https://doi.org/10.12688/wellcomeopenres.25027.2), Wellcome Open Research), ported to JAX, with a random walk on top.
It runs at province and zone level, and the national value is the sum of the provinces.
`randomwalk` is the same walk with no nowcast, kept for comparison.
`seabbs-renewal` carries one of the renewal models, `renewalbvd`, `renewalsp` or `renewal`, through the chain in `ci/policy.toml`.
The renewal models are renewal models of daily symptom onsets, built on [PyRenew](https://github.com/CDCgov/PyRenew), with a learnt onset-to-report delay and the hub's imputation of missing onsets in the model.
`renewalsp` fits each province on its own.
`renewal` fits the provinces together, with a pooled Rt and importation between provinces through a gravity kernel.
`renewalbvd` is `renewal` with the Rt estimates of [BVDOutbreakSize](https://github.com/epiforecasts/BVDOutbreakSize) added to the fit.
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

<!-- Zenodo DOI badge: replace XXXXXXX with the record number after the first release. -->
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

Abbott S (2026). BDBVHubRenewal: renewal-model nowcasts and forecasts for the BDBV Modeling Hub. Version 0.1.0. <https://github.com/seabbs/BDBVHubRenewal>.

```bibtex
@software{abbott_bdbvhubrenewal,
  author = {Abbott, Sam},
  title = {{BDBVHubRenewal}: renewal-model nowcasts and forecasts for the {BDBV} Modeling Hub},
  year = {2026},
  version = {0.1.0},
  url = {https://github.com/seabbs/BDBVHubRenewal}
}
```

The metadata are in [CITATION.cff](CITATION.cff), which GitHub shows as the "Cite this repository" button.

## Funding

This study is funded by the National Institute for Health and Care Research (NIHR) Health Protection Research Unit in Health Analytics & Modelling, a partnership between the UK Health Security Agency, Imperial College London and the London School of Hygiene & Tropical Medicine (grant code NIHR207404).
The views expressed are those of the author(s) and not necessarily those of the NIHR, UK Health Security Agency or the Department of Health and Social Care.

## Licence

The code is under the MIT licence, see [LICENSE](LICENSE).
