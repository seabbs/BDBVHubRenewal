# The epinowcast model used to simulate outbreaks

We simulated each synthetic outbreak from the generative model of epinowcast @placeholder.
We used `enw_simulate()`, which runs the epinowcast Stan model with fixed parameters and no likelihood.
It then draws reported counts from the model's generated quantities, so the simulated data come from the same model epinowcast fits.
`enw_simulate()` is on the `feature/829-simulate-forecast` branch of epinowcast, at commit `e084b08089796ca7def8dac43fd62d29ae34258b`.
The simulation script is `scripts/finetune_simulate.R`, and the settings drawn for each outbreak are in the synthetic scenario grid table.
Each outbreak is simulated on a daily time step, by place, with a longest reporting delay of 56 days.
A parameter we did not set is drawn from its epinowcast prior.

## Expectation model

Expected cases follow a renewal process, set up with `enw_expectation(r = ~1)` and a generation time.
The expected latent cases on day $t$ are $\lambda_t = R_t \sum_{s=1}^{45} g_s \lambda_{t-s}$, where $g_s$ is the generation time.
The generation time is a gamma distribution discretised to 1 to 45 days, with its mean and coefficient of variation drawn per outbreak.
The log of $R_t$ is supplied to `enw_simulate()` as `growth_rate`, one value per place and day.
Each place's log $R_t$ path is a weighted sum of a path shared by all places and a path of its own.
A path starts at a drawn initial value, has a Poisson number of step changes at random days and a daily random walk, and is held between -1.2 and 1.2.
The initial latent cases (`expr_lelatent_int`) are a drawn total split over places by log-normal weights.
Expected onsets are the latent cases with no latent reporting delay, scaled by the latent-to-observation proportion.

## Reference date model

The reporting delay of an onset is set by `enw_reference()`.
The parametric part is a lognormal delay, with a log mean (`refp_mean_int`) and log standard deviation (`refp_sd_int`) drawn per outbreak.
The log mean changes linearly by week of onset, through a `week` effect.
In outbreaks with several places, each place has its own random effect on the log mean, with a drawn standard deviation.
The non-parametric part adds a random effect for each delay, `~ 0 + (1 | delay)`, on the logit hazard of being reported.
The standard deviation of these delay effects is drawn per outbreak.

## Report date model

Effects of the report date are set by `enw_report()`.
A random effect for each day of the week of report, `~ 0 + (1 | day_of_week)`, acts on the logit hazard of being reported.
Their standard deviation is drawn per outbreak.
In half the outbreaks reports are made every day.
In the rest, `enw_dayofweek_structural_reporting()` limits reports to one to three weekdays, drawn from Monday to Friday.
Cases are then reported only on those days.

## Observation model

The count of cases with onset on day $t$ reported with delay $d$ is negative binomial, set by `enw_obs(family = "negbin")`.
Its mean is the expected onsets on day $t$ times the probability of a delay of $d$ given the reference and report date models.
Its overdispersion is set by `sqrt_phi`, drawn per outbreak, with $\phi = 1 / \texttt{sqrt\_phi}^2$ so that larger values give more overdispersion.
An outbreak is kept if it has between 50 and 30,000 reported cases, and otherwise its settings are drawn again.
