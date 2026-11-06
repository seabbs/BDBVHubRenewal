# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Hub metadata text for each model that may be submitted.

`bdbvhubrenewal metadata --model <model>` renders it with
`bdbvhubrenewal.submission.hub.render_model_metadata`.
"""

from __future__ import annotations

__all__ = ["MODEL_METADATA"]

_RENEWAL_OBSERVATION = (
    "Rt is dated by infection, and onsets are infections convolved with "
    "a log-normal incubation period with uncertain mean and SD "
    "(BVDOutbreakSize's priors). "
    "Confirmed onsets follow a time-varying ascertainment. The "
    "onset-to-report delay is nonparametric, a discrete hazard that is "
    "smooth over delay and shifts weekly with onset day, and known onsets "
    "are right truncated by it and by the report-to-publication lag. The "
    "share of reports with no onset date is modelled by province. The "
    "hub's imputed onsets are modelled as the hub makes them, from cases "
    "missing an onset date by report day, back-sampled with the hub's "
    "delay. Weekly known and imputed counts are negative binomial and the "
    "delays of known onsets are Dirichlet-multinomial. Over the forecast "
    "weeks the log Rt walk reverts towards Rt of 1 with a half-life of 2 "
    "weeks. Fitted by NUTS (NumPyro) to each data vintage. Each sample is "
    "one joint posterior "
    "trajectory. The national value is the sum of province samples plus "
    "the national cases in no province. Health zones are not fitted: "
    "each province sample is split across its zones with shares drawn "
    "once per sample from a Dirichlet built on recent and longer-run "
    "published zone counts, keeping each zone's published count, so zone "
    "samples sum to the province sample. See "
    "https://samabbott.co.uk/BDBVHubRenewal/methods/model/."
)
_RENEWAL_INPUTS = (
    "Hub target data (weekly confirmed cases by onset week, observed plus "
    "imputed, for the country, provinces and health zones), the hub's "
    "published onset-to-report delays by vintage, province populations and "
    "population-weighted province centres. The generation interval is "
    "BVDOutbreakSize's."
)

MODEL_METADATA = {
    "randomwalk": {
        "model_name": "RandomWalk baseline",
        "methods": (
            "Latest reported onset week carried forward per place, with "
            "log-scale random walk noise and Poisson draws and no nowcast. "
            "Health zones and provinces, with national the sum of provinces."
        ),
        "methods_long": (
            "For each place, the count reported for the latest onset week, "
            "the one holding the data cut-off, is carried forward as "
            "reported, with no adjustment for reporting delays. Later weeks "
            "are drawn on the log scale as a random walk whose weekly SD is "
            "the SD of the last 8 weekly changes of log(1 + cases), with "
            "Poisson noise. Each sample is a joint trajectory and the "
            "national value is the sum of province samples. See "
            "https://samabbott.co.uk/BDBVHubRenewal/methods/baselines/."
        ),
        "data_inputs": (
            "The hub's weekly target data for provinces and health zones."
        ),
    },
    # The PyRenew models, submitted as seabbs-renewal. The gate writes the
    # metadata of the model the chain submits (`metadata --model <variant>`).
    # The committed hub/model-metadata/seabbs-renewal.yml is that of
    # `[submission] model` in config/hub.toml: regenerate it with
    #   bdbvhubrenewal metadata --model <renewal|renewalsp|renewalbvd>
    # when that changes (tests/submission/test_fitting.py checks it).
    "renewal": {
        "model_name": "Province renewal model (PyRenew)",
        "methods": (
            "Province renewal model in PyRenew with pooled Rt, gravity "
            "importation and depletion, fitted to known and imputed onsets "
            "with modelled delays; national is the sum of provinces."
        ),
        "methods_long": (
            "A renewal process for each province with cases, with a national "
            "weekly random walk on log Rt plus partially pooled province "
            "deviations, importation between provinces through a gravity "
            "kernel, and susceptible depletion. Provinces with no cases get "
            "importations that can seed local spread. " + _RENEWAL_OBSERVATION
        ),
        "data_inputs": _RENEWAL_INPUTS,
    },
    "renewalsp": {
        "model_name": "Single-patch renewal model (PyRenew)",
        "methods": (
            "Independent renewal model in PyRenew for each province with "
            "cases, with no pooling or coupling, fitted to known and "
            "imputed onsets with modelled delays; national is the sum."
        ),
        "methods_long": (
            "A renewal process fitted to each province alone, with a weekly "
            "random walk on log Rt and no pooling, importation between "
            "provinces or depletion. Provinces with at least 20 cases in "
            "their last 6 data weeks are fitted this way. A province with "
            "fewer cases has a constant Poisson rate, with a Gamma prior "
            "updated by its weekly counts, drawn once per trajectory. "
            + _RENEWAL_OBSERVATION
        ),
        "data_inputs": _RENEWAL_INPUTS,
    },
    "renewalbvd": {
        "model_name": "Province renewal model with BVD Rt (PyRenew)",
        "methods": (
            "Province renewal model in PyRenew with pooled Rt, gravity "
            "importation and depletion, with a tempered likelihood on "
            "BVDOutbreakSize Rt, fitted to known and imputed onsets."
        ),
        "methods_long": (
            "A renewal process for each province with cases, with a national "
            "weekly random walk on log Rt plus partially pooled province "
            "deviations, importation between provinces through a gravity "
            "kernel, and susceptible depletion. Provinces with no cases get "
            "importations that can seed local spread. The model's national "
            "and patch Rt in the week of the newest BVDOutbreakSize release "
            "available at the cut-off are tied to that release's Rt "
            "estimates, as a likelihood term tempered by 0.5. "
            + _RENEWAL_OBSERVATION
        ),
        "data_inputs": (
            _RENEWAL_INPUTS + " The newest BVDOutbreakSize release published "
            "by the forecast's cut-off, for its Rt estimates."
        ),
    },
    # A hub model of its own (seabbs-rwnowcast), submitted every round.
    "randomwalknowcast": {
        "model_name": "RandomWalk on a nowcast",
        "methods": (
            "Baselinenowcast chain ladder on daily release snapshots, "
            "negative binomial new reports minus Poisson removals, then a "
            "log-scale random walk. Health zones and provinces, national "
            "the sum of provinces."
        ),
        "methods_long": (
            "The baselinenowcast method (Johnson et al. 2025, Wellcome Open "
            "Research, doi:10.12688/wellcomeopenres.25027.2; "
            "https://github.com/epinowcast/baselinenowcast/), from a JAX "
            "port of baselinenowcast 0.2.2, with a random walk on top. "
            "Daily onsets in the hub's weekly release snapshots are nowcast "
            "with a chain ladder. Unobserved report days are skipped, and "
            "every release pair in the 42-day window informs one daily delay, "
            "fitted by non-negative least squares on the log scale with "
            "releases as batched reports. A place uses its own delay at an "
            "age if it rests on at least 10 cases, "
            "and otherwise its parent's (zone, then province, then "
            "national). Daily nowcasts are summed to hub weeks. "
            # The uncertainty. Edit here if the observation model changes.
            "The observation model is baselinenowcast's negative binomial on "
            "the remaining count minus Poisson removals of the count "
            "reported, both fitted per horizon from retrospective nowcasts, "
            "so counts can be revised down. "
            "Each nowcast draw is carried forward from its latest week as a "
            "random walk on the log scale with Poisson noise. A place under "
            "10 cases a week, or with fewer than 3 errors, uses the spreads "
            "of its parent, and each place its own removal rate. The "
            "national value is the sum of provinces. See "
            "https://samabbott.co.uk/BDBVHubRenewal/methods/baselines/."
        ),
        "data_inputs": (
            "Hub daily onsets by release (observed plus imputed) for "
            "provinces and health zones, and the hub's weekly target data."
        ),
    },
}
"""Metadata text for each model that may be submitted."""
