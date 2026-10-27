# Nowcast

The current nowcast and forecast from the latest vintage.
{ .page-subtitle }

--8<-- "nowcast-meta.md"

--8<-- "latest-submission.md"

Bars are the published counts by onset week: known onsets below and imputed onsets above.
Faint lines are each model's 100 sample trajectories.
Bands are its 50% and 90% intervals and the solid line its median.
The dotted line is the reference date.
Weeks before the band are treated as complete by persistence.

!!! warning "Not a submission"
    These forecasts are built with the docs.
    Nothing has been submitted to the hub.

--8<-- "nowcast-tabs.md"

The figure comes from [`plot_nowcast`][bdbv_hub_renewal.plots.plot_nowcast], which the submission pull request will also embed:

```python
from bdbv_hub_renewal.plots import plot_nowcast

fig = plot_nowcast(location="CD-IT", vintage="2026-10-24")
fig.savefig("nowcast-cd-it.png")
```
