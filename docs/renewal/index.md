# Joint-model-informed renewal

We compare renewal models fitted to the hub data on their own with renewal models informed by a larger joint model, [BVDOutbreakSize](https://github.com/epiforecasts/BVDOutbreakSize), through approximate Markov melding.
All are built on [PyRenew](https://github.com/CDCgov/PyRenew) and fit the hub's known and imputed onsets.

![Share of the last onset week reported by days since onset for each fitted renewal model, beside the random walk nowcast's.](../figures/renewal-delay.svg#only-light)
![Share of the last onset week reported by days since onset for each fitted renewal model, beside the random walk nowcast's.](../figures/renewal-delay-dark.svg#only-dark)

- [Renewal models](../methods/model.md): the zone model, its single-patch province form, and the BVDOutbreakSize-informed model.
- [Methods](methods.md): how the models work, from data to submission, with their limitations.

Next: [Methods](methods.md).
