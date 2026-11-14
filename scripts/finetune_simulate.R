# Simulate outbreaks with epinowcast to fine-tune Chronos-2
# (src/bdbvhubrenewal/forecasters/finetune.py).
#
# Each outbreak is one draw of epinowcast's generative model with fixed
# parameters (`enw_simulate`). An outbreak's design, its categorical
# choices, is drawn first (`draw_design`), then its continuous settings
# (`draw_values`):
#
# - strata (30%, 40%, 20%, 10%): one, 2 to 26 places, 2 to 8 age groups
#   in one place, or 6 to 16 places by 3 to 7 age groups. Strata share part of their growth
#   (a shared path, and for places by ages a path by place) and their
#   delay (random effects by place and by age), and some places are
#   quiet (seeded with almost no cases);
# - generative process, from `enw_expectation`: a renewal process with a
#   drawn generation time driven by a log Rt path with steps and drift,
#   the same with latent infections and an incubation delay
#   (`latent_reporting_delay`), or with no generation time a weekly random
#   walk on the growth rate, a random walk on log incidence, growth with
#   step changes, or smoothly varying growth. On top, each outbreak has a
#   trajectory, a quarter each: turning over, flat or slowly varying,
#   rising then falling, or a second wave (`trajectory`). The growth rate
#   is given to `enw_simulate`, so the `r` formula is not used;
# - reference date (`enw_reference`): a parametric delay alone, a
#   non-parametric hazard alone, or both. The parametric delay is a
#   lognormal, gamma or exponential, and the hazard has effects by delay,
#   by reference week, or both;
# - delay changes: one outbreak in five has a stable delay. The others
#   draw a combination of a linear drift in the delay's location, a drift
#   in its spread, up to three step changes in both at random weeks (step
#   covariates in the parametric formula) and a weekly random walk
#   (`rw(week)`);
# - report date (`enw_report`): any of day-of-week effects, holidays (a
#   `holiday` covariate on a few report dates) and report-week effects,
#   `(1 | week)` or `rw(week)`; reports every day or on one to three
#   weekdays (a `structural` table), which may change once on a report
#   date;
# - observations: "negbin" (NB2), "negbin1d" (NB1) or "poisson";
# - longest delay: the shortest of 3, 4, 6, 8 and 12 weeks that is at
#   least five times the delay's mean, or one step longer, up to ten times
#   the mean, in one outbreak in four.
#
# The NB1 dispersion matches the NB2 scale (epinowcast issue #937). With
# expected onsets lambda and delay probabilities p_d, NB2 has
# Var(N) = lambda + phi * lambda^2 * sum(p_d^2) and NB1 has
# Var(N) = lambda * (1 + xi), so xi = phi * lambda_hat * sum(p_d^2). In
# epinowcast phi = sqrt_phi^2 (NB2) and xi = sqrt_phi^2 (NB1), so the NB1
# draw is sqrt_phi * sqrt(lambda_hat * sum(p_d^2)), with sum(p_d^2) from
# the baseline delay and lambda_hat the median of the stratum-days with
# at least one expected onset, at least 1.
#
# A draw that fails (most often from overflowing growth) or whose size is
# out of range is drawn again with the same design, so the designs keep
# their shares. A design that fails five times is logged and skipped.
#
# The output is each outbreak's daily reporting triangle, a count by
# stratum (`location`), onset day and report date. The weekly training
# blocks sum it to onset weeks with the hub's weekly releases
# (`forecasters.finetune.releases`), so they share the daily outbreaks'
# generation times, and the total over strata is its own series.
#
# `enw_simulate` is on the feature/829-simulate-forecast branch of
# epinowcast, pinned below. Install it into a local library first, from
# a clone of epinowcast:
#
#   git -C <epinowcast clone> archive e084b080 | tar -x -C src
#   R CMD INSTALL --library=output/finetune/rlib src
#
# Run from the repo root:
#
#   Rscript scripts/finetune_simulate.R <n> <seed> <out_dir>

epinowcast_sha <- "e084b080"
grid_name <- "r3"
args <- commandArgs(trailingOnly = TRUE)
n_outbreaks <- as.integer(args[[1]])
seed <- as.integer(args[[2]])
out_dir <- args[[3]]
lib <- Sys.getenv("ENW_LIB", "output/finetune/rlib")
.libPaths(c(lib, .libPaths()))

suppressPackageStartupMessages({
  library(epinowcast)
  library(data.table)
})
options(cmdstanr_warn_inits = FALSE, mc.cores = 1)
enw_set_cache(
  Sys.getenv("ENW_CACHE", "output/finetune/enwcache"), type = "session"
)
model <- suppressMessages(enw_model(threads = FALSE))

# Observation draws per simulation: each is its own realisation of the
# same epidemic and reporting, sharing the costly setup.
draws <- as.integer(Sys.getenv("ENW_DRAWS", "10"))
# Set ENW_DEBUG to see Stan's messages.
debug <- nzchar(Sys.getenv("ENW_DEBUG"))

processes <- c(
  "renewal", "renewal_incubation", "growth_walk", "log_incidence_walk",
  "growth_steps", "growth_smooth"
)

# Discretised gamma on 1, ..., n steps.
gamma_pmf <- function(mean, cv, n) {
  shape <- 1 / cv^2
  cdf <- pgamma(0:n, shape = shape, rate = shape / mean)
  pmf <- diff(cdf)
  pmf / sum(pmf)
}

# A trajectory's shift of log Rt (`scale` 1) or of the daily growth rate
# (`scale` 0.1) over `n` days: a linear decline (turning over), none
# (flat), a rise then a fall, or a decline then a step up (second wave).
trajectory <- function(n, kind, v, scale) {
  t <- seq_len(n) / n
  scale * switch(kind,
    turnover = -(0.1 + 0.3 * v$shape_size) * t,
    flat = numeric(n),
    rise_fall = (0.1 + 0.3 * v$shape_size) * (1 - 2 * t),
    second_wave = -0.2 * v$shape_size * t +
      (0.15 + 0.25 * v$wave_size) * (t >= v$wave_at)
  )
}

# The log reproduction number of a daily growth rate `r` under a gamma
# generation time (the inverse of its Laplace transform).
log_r_from_growth <- function(r, gt_mean, gt_cv) {
  x <- 1 + r * gt_mean * gt_cv^2
  ifelse(x > 0, log(pmax(x, 1e-12)) / gt_cv^2, -2)
}

# One log Rt path: the initial growth's Rt, steps at random times and a
# random walk drift.
log_rt_path <- function(n, v) {
  at <- sort(sample.int(n, v$rt_steps, replace = TRUE))
  jumps <- numeric(n)
  jumps[at] <- rnorm(v$rt_steps, -0.1, v$rt_step_sd)
  log_r_from_growth(v$r0, v$gt_mean, v$gt_cv) + cumsum(jumps) +
    cumsum(rnorm(n, 0, v$rt_drift_sd))
}

# One daily growth rate path of a process with no generation time.
growth_path <- function(n, process, v) {
  switch(process,
    # A weekly AR(1) on growth, which reverts to the initial growth.
    growth_walk = {
      weeks <- ceiling(n / 7)
      g <- numeric(weeks)
      for (w in seq_len(weeks)[-1]) {
        g[w] <- v$walk_pacf * g[w - 1] + rnorm(1, 0, v$walk_sd)
      }
      v$r0 + rep(g, each = 7)[seq_len(n)]
    },
    # A random walk on log incidence: white noise on the growth rate.
    log_incidence_walk = v$r0 + rnorm(n, 0, v$incidence_sd),
    growth_steps = {
      at <- sort(sample.int(n, v$growth_steps, replace = TRUE))
      jumps <- numeric(n)
      jumps[at] <- rnorm(v$growth_steps, -0.01, v$growth_step_sd)
      v$r0 + cumsum(jumps)
    },
    growth_smooth = v$r0 + v$smooth_amp *
      sin(2 * pi * seq_len(n) / v$smooth_period + v$smooth_phase)
  )
}

# The delay's epinowcast parameters (`refp_mean_int`, `refp_sd_int`), its
# 99th centile and its probabilities on 0, ..., max_delay - 1 days.
# Lognormal is meanlog and sdlog; gamma is shape exp(mean_int) and rate
# sd_int; the exponential rate is exp(-mean_int). With no parametric
# delay, the non-parametric hazard's intercept is the constant hazard of
# a geometric delay with that mean.
delay_parameters <- function(family, v, max_delay = NULL) {
  out <- switch(family,
    lognormal = list(
      mean_int = log(v$delay_median), sd_int = v$delay_sdlog,
      q99 = qlnorm(0.99, log(v$delay_median), v$delay_sdlog),
      cdf = function(d) plnorm(d, log(v$delay_median), v$delay_sdlog)
    ),
    gamma = {
      shape <- 1 / v$delay_cv^2
      rate <- shape / v$delay_mean
      list(
        mean_int = log(shape), sd_int = rate,
        q99 = qgamma(0.99, shape, rate),
        cdf = function(d) pgamma(d, shape, rate)
      )
    },
    exponential = list(
      mean_int = log(v$delay_mean),
      q99 = qexp(0.99, 1 / v$delay_mean),
      cdf = function(d) pexp(d, 1 / v$delay_mean)
    ),
    none = {
      hazard <- 1 / (1 + v$delay_mean)
      list(
        hazard = hazard, q99 = qgeom(0.99, hazard),
        cdf = function(d) pgeom(d - 1, hazard)
      )
    }
  )
  if (!is.null(max_delay)) {
    pmf <- diff(out$cdf(0:max_delay))
    out$pmf <- pmf / sum(pmf)
  }
  out
}

weekdays_drawn <- function() {
  # Onsets are reported every day or, in one outbreak in three, on one to
  # three weekdays.
  if (runif(1) < 2 / 3) {
    return("all")
  }
  days <- c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
  paste(sort(sample(days, sample(1:3, 1))), collapse = ";")
}

# The categorical choices of an outbreak, kept when a draw fails.
draw_design <- function() {
  strata <- sample(
    c("single", "places", "ages", "places_ages"), 1,
    prob = c(0.3, 0.4, 0.2, 0.1)
  )
  reference <- sample(
    c("parametric", "non_parametric", "both"), 1
  )
  stable <- runif(1) < 0.2
  change <- function(p) !stable && runif(1) < p
  parametric <- reference != "non_parametric"
  hazard <- if (reference == "parametric") {
    "none"
  } else if (stable) {
    "delay"
  } else {
    sample(c("delay", "week", "delay_week"), 1)
  }
  # Places by ages have at most 60 strata.
  n_places <- switch(strata,
    single = 1L, ages = 1L,
    places = sample(2:26, 1),
    places_ages = sample(6:16, 1)
  )
  n_ages <- switch(strata,
    single = 1L, places = 1L,
    ages = sample(2:8, 1),
    places_ages = sample(3:min(7L, 60L %/% n_places), 1)
  )
  list(
    grid = grid_name,
    strata = strata,
    n_places = n_places,
    n_ages = n_ages,
    process = sample(processes, 1),
    trajectory = sample(
      c("turnover", "flat", "rise_fall", "second_wave"), 1
    ),
    reference = reference,
    delay_family = if (parametric) {
      sample(
        c("lognormal", "gamma", "exponential"), 1, prob = c(0.4, 0.4, 0.2)
      )
    } else {
      "none"
    },
    hazard = hazard,
    stable = stable,
    delay_drift = parametric && change(0.6),
    spread_drift = parametric && change(0.5),
    delay_steps = if (parametric && !stable) {
      sample(0:3, 1, prob = c(3, 3, 2, 2))
    } else {
      0L
    },
    delay_walk = parametric && change(0.3),
    reference_dow = runif(1) < 0.5,
    report_dow = runif(1) < 0.5,
    report_holiday = runif(1) < 0.25,
    report_week = sample(c("none", "week", "rw"), 1, prob = c(3, 1, 1)),
    report_days = weekdays_drawn(),
    report_change = change(0.3),
    report_days_after = weekdays_drawn(),
    family = sample(c("negbin", "negbin1d", "poisson"), 1)
  )
}

# The continuous settings of an outbreak, drawn again when a draw fails.
draw_values <- function(design) {
  places_ages <- design$strata == "places_ages"
  log_uniform <- function(lo, hi) exp(runif(1, log(lo), log(hi)))
  v <- list(
    weeks = if (places_ages) sample(16:24, 1) else sample(16:32, 1),
    offset = sample(0:6, 1),
    quiet = runif(1, 0, 0.9),
    coupling = runif(1),
    gt_mean = log_uniform(2.5, 20),
    gt_cv = runif(1, 0.4, 0.8),
    incubation_mean = runif(1, 3, 12),
    incubation_cv = runif(1, 0.3, 0.7),
    # Initial daily growth for every process; a renewal process takes the
    # Rt that gives it under its generation time.
    r0 = max(min(rnorm(1, 0, 0.03), 0.08), -0.08),
    shape_size = runif(1),
    wave_size = runif(1),
    wave_at = runif(1, 0.4, 0.8),
    rt_steps = rpois(1, 1),
    rt_step_sd = runif(1, 0.05, 0.25),
    rt_drift_sd = runif(1, 0, 0.03),
    walk_pacf = runif(1, 0.7, 0.95),
    walk_sd = runif(1, 0.01, 0.03),
    incidence_sd = runif(1, 0.05, 0.3),
    growth_steps = sample(1:3, 1),
    growth_step_sd = runif(1, 0.02, 0.06),
    smooth_amp = runif(1, 0.01, 0.06),
    smooth_period = runif(1, 40, 200),
    smooth_phase = runif(1, 0, 2 * pi),
    # The seed is scaled so the expected size is the target.
    target_size = log_uniform(30, 1e5) *
      max(1, design$n_places * design$n_ages / 26),
    stratum_sd = runif(1, 0.5, 2),
    delay_median = log_uniform(1, 10),
    delay_sdlog = runif(1, 0.5, 2.5),
    delay_mean = log_uniform(2, 21),
    # Gamma delays with shape above 2 fail in the simulate branch.
    delay_cv = runif(1, 0.8, 1.6),
    # Log changes in the delay's location and spread from the first week
    # to the last, as a linear drift.
    delay_change = rnorm(1, 0, 0.3),
    spread_change = rnorm(1, 0, 0.3),
    step_at_1 = runif(1, 0.1, 0.9),
    step_at_2 = runif(1, 0.1, 0.9),
    step_at_3 = runif(1, 0.1, 0.9),
    step_mean_1 = rnorm(1, 0, 0.3),
    step_mean_2 = rnorm(1, 0, 0.3),
    step_mean_3 = rnorm(1, 0, 0.3),
    step_spread_1 = rnorm(1, 0, 0.3),
    step_spread_2 = rnorm(1, 0, 0.3),
    step_spread_3 = rnorm(1, 0, 0.3),
    walk_mean_sd = runif(1, 0, 0.1),
    walk_spread_sd = runif(1, 0, 0.1),
    delay_place_sd = runif(1, 0.1, 0.5),
    delay_age_sd = runif(1, 0.05, 0.25),
    hazard_walk_sd = runif(1, 0.05, 0.2),
    hazard_sd = runif(1, 0, 0.15),
    hazard_week_sd = runif(1, 0, 0.4),
    reference_dow_sd = runif(1, 0.1, 0.5),
    dow_sd = runif(1, 0.1, 0.6),
    holidays = sample(2:6, 1),
    holiday_effect = runif(1, -2.5, -0.5),
    report_week_sd = runif(1, 0, 0.2),
    report_change_at = runif(1, 0.2, 0.8),
    sqrt_phi = log_uniform(0.1, 1)
  )
  v$max_delay <- max_delay_for(delay_parameters(design$delay_family, v)$q99)
  v
}

# The longest delay in days: the shortest of 3, 4, 6, 8 and 12 weeks at
# or above the delay's 99th centile, at most 12 weeks.
max_delay_for <- function(q99) {
  options <- c(21L, 28L, 42L, 56L, 84L)
  options[min(which(c(options >= q99, TRUE))[1], length(options))]
}

# Expected onsets of each stratum by day, as epinowcast's renewal or
# growth model gives them from the latent seeding and the growth rate
# (ignoring an incubation delay, which keeps their scale).
expected_onsets <- function(size, gt, log_r) {
  n_gt <- length(gt)
  n <- nrow(log_r)
  out <- matrix(0, n_gt + n, length(size))
  out[seq_len(n_gt), ] <- rep(size, each = n_gt)
  rev_gt <- rev(gt)
  for (t in seq_len(n)) {
    past <- out[t:(t + n_gt - 1), , drop = FALSE]
    out[n_gt + t, ] <- exp(log_r[t, ]) * colSums(past * rev_gt)
  }
  out
}

# A formula from its terms, `~ 0` if there are none.
as_formula <- function(intercept, terms) {
  terms <- c(if (intercept) "1" else "0", terms)
  as.formula(paste("~", paste(terms, collapse = " + ")))
}

# Standard deviations of a module's random effects, named by effect.
effect_sds <- function(rdesign, sds) {
  effects <- colnames(rdesign)[-1]
  array(as.numeric(unlist(sds[effects])))
}

simulate_outbreak <- function(design, v) {
  # Four weeks plus the longest delay past the last cut-off, so a block's
  # horizon (four weeks) is complete.
  max_delay <- v$max_delay
  n_time <- 7L * (v$weeks + max_delay %/% 7L + 4L)
  place_ids <- sprintf("P%02d", seq_len(design$n_places))
  age_ids <- sprintf("A%d", seq_len(design$n_ages))
  dates <- as.IDate("2026-01-03") + v$offset + seq_len(n_time) - 1L
  grid <- CJ(
    place = place_ids, age = age_ids, reference_date = dates,
    delay = 0:(max_delay - 1)
  )
  grid[, report_date := reference_date + delay][, delay := NULL]
  grid <- grid[report_date <= max(dates)][, confirm := 0L]
  pobs <- suppressMessages(suppressWarnings(enw_preprocess_data(
    grid, by = c("place", "age"), max_delay = max_delay
  )))
  strata <- unique(pobs$metareference[[1]][, .(.group, place, age)])
  setorder(strata, .group)
  multi_place <- design$n_places > 1
  multi_age <- design$n_ages > 1
  at_week <- function(share) dates[1] + round(share * 7 * v$weeks)

  # Expectation.
  renewal <- design$process %in% c("renewal", "renewal_incubation")
  gt <- if (renewal) gamma_pmf(v$gt_mean, v$gt_cv, 45L) else 1
  lrd <- if (design$process == "renewal_incubation") {
    gamma_pmf(v$incubation_mean, v$incubation_cv, 30L)
  } else {
    1
  }
  # A reference-date day-of-week effect acts on the expected counts.
  expectation <- enw_expectation(
    r = ~1, generation_time = gt, latent_reporting_delay = lrd,
    observation = if (design$reference_dow) ~ (1 | day_of_week) else ~1,
    data = pobs
  )
  n_r <- expectation$data$expr_t
  path <- function() {
    if (renewal) {
      shifted <- log_rt_path(n_r, v) + trajectory(n_r, design$trajectory, v, 1)
      # Rt within what daily growth of -0.1 to 0.1 gives.
      limits <- log_r_from_growth(c(-0.1, 0.1), v$gt_mean, v$gt_cv)
      pmin(pmax(shifted, limits[1]), limits[2])
    } else {
      shifted <- growth_path(n_r, design$process, v) +
        trajectory(n_r, design$trajectory, v, 0.1)
      pmin(pmax(shifted, -0.1), 0.1)
    }
  }
  # Strata share a path, and places by ages also share their place's.
  shared <- path()
  by_place <- lapply(place_ids, function(p) path())
  names(by_place) <- place_ids
  log_r <- sapply(seq_len(nrow(strata)), function(i) {
    own <- if (design$strata == "places_ages") {
      by_place[[strata$place[i]]]
    } else {
      path()
    }
    v$coupling * shared + (1 - v$coupling) * own
  })
  log_r <- matrix(log_r, nrow = n_r)
  place_weight <- exp(rnorm(design$n_places, 0, v$stratum_sd))
  place_weight[runif(design$n_places) < v$quiet] <- 0
  if (all(place_weight == 0)) place_weight[1] <- 1
  age_weight <- exp(rnorm(design$n_ages, 0, v$stratum_sd / 2))
  weight <- place_weight[match(strata$place, place_ids)] *
    age_weight[match(strata$age, age_ids)]
  if (design$strata == "ages") {
    weight[runif(length(weight)) < v$quiet / 2] <- 0
    if (all(weight == 0)) weight[1] <- 1
  }
  # The expected outbreak is linear in the seed, so the seed is scaled to
  # give the target size rather than redrawn until the size fits.
  share <- weight / sum(weight)
  scale <- v$target_size / sum(expected_onsets(share, gt, log_r))
  v$seed_size <- scale
  size <- pmax(scale * share, exp(-8))
  lelatent <- matrix(
    log(size), nrow = expectation$data$expr_gt_n, ncol = nrow(strata),
    byrow = TRUE
  )

  # Reference date.
  steps <- sprintf("step%d", seq_len(design$delay_steps))
  for (k in seq_along(steps)) {
    at <- at_week(v[[sprintf("step_at_%d", k)]])
    set(pobs$metareference[[1]], j = steps[k],
        value = as.numeric(pobs$metareference[[1]]$date >= at))
  }
  stratum_terms <- c(
    if (multi_place) "(1 | place)", if (multi_age) "(1 | age)"
  )
  parametric <- design$delay_family != "none"
  refp <- if (parametric) {
    as_formula(TRUE, c(
      if (design$delay_drift || design$spread_drift) "week",
      steps, stratum_terms, if (design$delay_walk) "rw(week)"
    ))
  } else {
    ~0
  }
  hazard_terms <- switch(design$hazard,
    none = NULL,
    delay = c("rw(delay)", "(1 | delay)"),
    week = "(1 | week)",
    delay_week = c("rw(delay)", "(1 | delay)", "(1 | week)")
  )
  refnp <- if (design$hazard == "none") {
    ~0
  } else if (parametric) {
    as_formula(FALSE, hazard_terms)
  } else {
    as_formula(TRUE, c(hazard_terms, stratum_terms))
  }
  reference <- enw_reference(
    parametric = refp,
    distribution = if (parametric) design$delay_family else "none",
    non_parametric = refnp, data = pobs
  )
  rd <- reference$data
  delay <- delay_parameters(design$delay_family, v, max_delay)
  stratum_sd <- list(place = v$delay_place_sd, age = v$delay_age_sd)
  parameters <- list()
  if (parametric) {
    fcols <- colnames(rd$refp_fdesign)
    n_weeks <- max(pobs$metareference[[1]]$week)
    mean_beta <- rnorm(length(fcols))
    sd_beta <- rnorm(length(fcols))
    mean_beta[fcols == "week"] <- if (design$delay_drift) {
      v$delay_change / n_weeks
    } else {
      0
    }
    sd_beta[fcols == "week"] <- if (design$spread_drift) {
      v$spread_change / n_weeks
    } else {
      0
    }
    for (k in seq_along(steps)) {
      mean_beta[fcols == steps[k]] <- v[[sprintf("step_mean_%d", k)]]
      sd_beta[fcols == steps[k]] <- v[[sprintf("step_spread_%d", k)]]
    }
    parameters$refp_mean_int <- delay$mean_int
    parameters$refp_mean_beta <- array(mean_beta)
    sds <- effect_sds(rd$refp_rdesign, stratum_sd)
    if (length(sds)) {
      parameters$refp_mean_beta_sd <- sds
      parameters$refp_sd_beta_sd <- array(sds / 2)
    }
    if (design$delay_family != "exponential") {
      parameters$refp_sd_int <- delay$sd_int
      parameters$refp_sd_beta <- array(sd_beta)
    }
    if (design$delay_walk) {
      parameters$refp_arima_z <- matrix(
        rnorm(rd$refp_arima_T * rd$refp_arima_G),
        rd$refp_arima_T, rd$refp_arima_G
      )
      parameters$refp_arima_sigma <- array(v$walk_mean_sd)
      if (design$delay_family != "exponential") {
        parameters$refp_arima_sd_sigma <- array(v$walk_spread_sd)
      }
    }
  }
  if (design$hazard != "none") {
    parameters$refnp_beta <- array(rnorm(rd$refnp_fncol))
    parameters$refnp_beta_sd <- effect_sds(rd$refnp_rdesign, c(
      list(delay = v$hazard_sd, week = v$hazard_week_sd), stratum_sd
    ))
    if (!parametric) {
      parameters$refnp_int <- qlogis(delay$hazard)
    }
    # A smooth hazard by delay: a random walk over delays.
    if (rd$refnp_arima_present == 1) {
      parameters$refnp_arima_z <- matrix(
        rnorm(rd$refnp_arima_T * rd$refnp_arima_G),
        rd$refnp_arima_T, rd$refnp_arima_G
      )
      parameters$refnp_arima_sigma <- array(v$hazard_walk_sd)
    }
  }
  if (design$reference_dow) {
    ed <- expectation$data
    parameters$expl_beta <- array(rnorm(ed$expl_fncol))
    parameters$expl_beta_sd <- array(v$reference_dow_sd)
  }

  # Report date.
  if (design$report_holiday) {
    holidays <- sample(dates, v$holidays)
    set(pobs$metareport[[1]], j = "holiday",
        value = as.numeric(pobs$metareport[[1]]$date %in% holidays))
  }
  report_terms <- c(
    if (design$report_dow) "(1 | day_of_week)",
    if (design$report_holiday) "holiday",
    switch(design$report_week, none = NULL, week = "(1 | week)",
           rw = "rw(week)")
  )
  before <- strsplit(design$report_days, ";")[[1]]
  after <- if (design$report_change) {
    strsplit(design$report_days_after, ";")[[1]]
  } else {
    before
  }
  structural <- NULL
  if (!identical(c(before, after), c("all", "all"))) {
    switch_at <- at_week(v$report_change_at)
    structural <- enw_structural_reporting_metadata(pobs)
    structural[, report := as.integer(fifelse(
      report_date < switch_at,
      "all" %in% before | weekdays(report_date) %in% before,
      "all" %in% after | weekdays(report_date) %in% after
    ))]
    structural <- structural[, .(.group, date, report_date, report)]
  }
  # epinowcast fails on a report formula with only a random walk and no
  # intercept, so that one has an intercept, which the report model
  # always fixes at zero.
  rw_only <- identical(report_terms, "rw(week)")
  report <- enw_report(
    non_parametric = as_formula(rw_only, report_terms),
    structural = structural, data = pobs
  )
  rep <- report$data
  if (length(report_terms)) {
    rcols <- colnames(rep$rep_fdesign)
    rep_beta <- rnorm(length(rcols))
    rep_beta[rcols == "holiday"] <- v$holiday_effect
    parameters$rep_beta <- array(rep_beta)
    parameters$rep_beta_sd <- effect_sds(
      rep$rep_rdesign, list(day_of_week = v$dow_sd, week = v$report_week_sd)
    )
    if (design$report_week == "rw") {
      parameters$rep_arima_z <- matrix(
        rnorm(rep$rep_arima_T * rep$rep_arima_G),
        rep$rep_arima_T, rep$rep_arima_G
      )
      parameters$rep_arima_sigma <- array(v$report_week_sd)
    }
  }
  parameters$expr_lelatent_int <- lelatent

  # Observations: the NB1 dispersion on the NB2 scale (epinowcast #937).
  onsets <- expected_onsets(size, gt, log_r)
  v$lambda_hat <- max(1, median(onsets[onsets >= 1]), na.rm = TRUE)
  v$sum_p2 <- sum(delay$pmf^2)
  v$sqrt_phi_obs <- switch(design$family,
    negbin = v$sqrt_phi,
    negbin1d = v$sqrt_phi * sqrt(v$lambda_hat * v$sum_p2),
    poisson = NA_real_
  )
  if (design$family != "poisson") {
    parameters$sqrt_phi <- array(v$sqrt_phi_obs)
  }
  sims <- suppressMessages(suppressWarnings(enw_simulate(
    pobs, growth_rate = as.numeric(log_r), parameters = parameters,
    reference = reference, report = report, expectation = expectation,
    obs = enw_obs(family = design$family, data = pobs), model = model,
    seed = sample.int(.Machine$integer.max, 1), draws = draws,
    show_messages = debug, show_exceptions = debug, refresh = 0
  )))
  # The draws in the order `enw_pp_summary` binds them to the triangle,
  # read directly as summarising them is slow.
  cells <- data.table::copy(sims$new_confirm[[1]])
  data.table::setorderv(cells, c(".group", "reference_date"))
  obs <- sims$fit[[1]]$draws("pp_obs", format = "matrix")
  pp <- rbindlist(lapply(seq_len(nrow(obs)), function(k) {
    cells[, .(.group, place, age, reference_date, report_date,
      draw = k, mean = as.numeric(obs[k, ])
    )]
  }))
  pp[, location := if (multi_place && multi_age) {
    paste(place, age, sep = "-")
  } else if (multi_age) {
    age
  } else {
    place
  }]
  list(
    values = v,
    counts = pp[mean > 0, .(
      outbreak = NA_character_, draw, timestep = "day", location,
      reference_date, report_date, count = as.integer(mean)
    )]
  )
}

# Outbreaks outside `min_size` to `max_size` cases are drawn again; many
# strata allow larger outbreaks.
min_size <- 10
max_size <- function(design) {
  300000 * max(1, design$n_places * design$n_ages / 26)
}

if (sys.nframe() == 0L) {
  set.seed(seed)
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  t0 <- Sys.time()
  settings <- list()
  counts <- list()
  failures <- list()
  for (i in seq_len(n_outbreaks)) {
    id <- sprintf("%d-%04d", seed, i)
    design <- draw_design()
    t1 <- Sys.time()
    out <- NULL
    for (try in 1:5) {
      v <- draw_values(design)
      res <- tryCatch(simulate_outbreak(design, v), error = function(e) e)
      reason <- if (inherits(res, "error")) {
        conditionMessage(res)
      } else {
        # The draws' mean size is checked against the limits.
        size <- sum(res$counts$count) / draws
        if (size < min_size) {
          "too small"
        } else if (size > max_size(design)) {
          "too large"
        } else {
          NA_character_
        }
      }
      if (is.na(reason)) {
        out <- res
        break
      }
      failures[[length(failures) + 1L]] <- c(
        list(outbreak = id, try = try, reason = reason), design, v
      )
    }
    if (is.null(out)) next
    # Each draw is an outbreak `<seed>-<index>-<draw>`.
    out$counts[, outbreak := if (draws > 1) paste(id, draw, sep = "-") else id]
    out$counts[, draw := NULL]
    settings[[length(settings) + 1L]] <- c(
      list(outbreak = id), design, out$values,
      list(
        draws = draws, tries = try, cases = sum(out$counts$count) / draws,
        seconds = as.numeric(Sys.time() - t1, units = "secs")
      )
    )
    counts[[length(counts) + 1L]] <- out$counts
  }
  message(sprintf(
    "%d of %d outbreaks in %.1f s (epinowcast %s, grid %s)",
    length(settings), n_outbreaks,
    as.numeric(Sys.time() - t0, units = "secs"), epinowcast_sha, grid_name
  ))
  fwrite(
    rbindlist(counts), file.path(out_dir, sprintf("counts-%d.csv.gz", seed))
  )
  fwrite(
    rbindlist(settings), file.path(out_dir, sprintf("settings-%d.csv", seed))
  )
  if (length(failures)) {
    fwrite(
      rbindlist(failures, fill = TRUE),
      file.path(out_dir, sprintf("failures-%d.csv", seed))
    )
  }
}
