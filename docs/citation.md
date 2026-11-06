# Citation and licence

How to cite the software and the data, the funding and the licence.
{ .page-subtitle }

## Citing this software

The citation metadata are in [CITATION.cff](https://github.com/seabbs/BDBVHubRenewal/blob/main/CITATION.cff).
GitHub turns it into the "Cite this repository" button on the repository page.

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

## Archived results

Each hub release we fit gets a [GitHub release](https://github.com/seabbs/BDBVHubRenewal/releases) named `results-<hub release>`, made once every stored fit at that release is of the current code.
It holds every model's forecast at the release in the hub's format, for the country, provinces and health zones, with the scores, the fit summaries, the data and the rendered site.
Each code release `vX.Y.Z` gets `results-vX.Y.Z` in the same form.
Each submitted round's code is released as `round-<reference date>`, which Zenodo archives with a DOI.

## Citing the data

The data come from the [BDBV Modeling Hub](https://github.com/InsightNet-US/BDBV-Modeling-Hub).
Credit them as the hub asks.
The surveillance data are from the Institut National de Santé Publique (INSP) and the Incident Management Support Team (IMST), Democratic Republic of the Congo.
They are processed by the Atlantic Coast Center for Infectious Disease Dynamics and Analytics (ACCIDDA) and the Kinshasa School of Public Health, University of Kinshasa.

## Funding

This study is funded by the National Institute for Health and Care Research (NIHR) Health Protection Research Unit in Health Analytics & Modelling, a partnership between the UK Health Security Agency, Imperial College London and the London School of Hygiene & Tropical Medicine (grant code NIHR207404).
The views expressed are those of the author(s) and not necessarily those of the NIHR, UK Health Security Agency or the Department of Health and Social Care.

## Licence

The code is under the MIT licence.
See [LICENSE](https://github.com/seabbs/BDBVHubRenewal/blob/main/LICENSE).
