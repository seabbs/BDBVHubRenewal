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
See the [documentation](https://samabbott.co.uk/BDBVHubRenewal/) for the methods and results.

## Installation

Install the package and its dependencies with [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/seabbs/BDBVHubRenewal
cd BDBVHubRenewal
uv sync
```

## Usage

The [Getting started](https://samabbott.co.uk/BDBVHubRenewal/getting-started/) page walks through a small fit.

## Contributing

Contributions are welcome.
See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citing

To cite this software, use [CITATION.cff](CITATION.cff), or the "Cite this repository" button on GitHub.

## Licence

The code is under the MIT licence, see [LICENSE](LICENSE).
