# Nowcast

The current nowcast and forecast from the latest vintage.
{ .page-subtitle }

--8<-- "nowcast-meta.md"

Bars are the published counts by onset week: known onsets below and imputed onsets above.
Lines and bands are each model's median and 50% and 90% intervals.
The dotted line is the reference date.
Weeks before the band are treated as complete by persistence.

!!! warning "Not a submission"
    These forecasts are built with the docs.
    Nothing has been submitted to the hub.

--8<-- "nowcast-tabs.md"

The figure comes from [`plot_nowcast`][bdbv_forecasts.plots.plot_nowcast], which the submission pull request will also embed:

```python
from bdbv_forecasts.plots import plot_nowcast

fig = plot_nowcast(location="CD-IT", vintage="2026-10-24")
fig.savefig("nowcast-cd-it.png")
```
