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
#   step changes, or smoothly varying growth. Each declines linearly by a
#   drawn amount, as epidemics turn over. The growth rate is given to
#   `enw_simulate`, so the `r` formula is not used;
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
# - longest delay: the shortest of 4, 8 and 12 weeks that is at least ten
#   times the delay's mean, or one step longer in one outbreak in four.
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
grid_name <- "r2-cal"
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

# One log Rt path: steps at random times, a random walk drift, and a
# linear decline (`turn`) as an epidemic turns over.
log_rt_path <- function(n, v) {
  at <- sort(sample.int(n, v$rt_steps, replace = TRUE))
  jumps <- numeric(n)
  jumps[at] <- rnorm(v$rt_steps, -0.2, v$rt_step_sd)
  path <- v$log_r0 + cumsum(jumps) + cumsum(rnorm(n, 0, v$rt_drift_sd)) -
    v$rt_turn * seq_len(n) / n
  pmin(pmax(path, -1.2), 1.2)
}

# One daily growth rate path of a process with no generation time.
growth_path <- function(n, process, v) {
  path <- switch(process,
    growth_walk = {
      steps <- rnorm(ceiling(n / 7), 0, v$walk_sd)
      v$r0 + rep(cumsum(steps), each = 7)[seq_len(n)]
    },
    # A random walk on log incidence: white noise on the growth rate.
    log_incidence_walk = v$r0 / 2 + rnorm(n, 0, v$incidence_sd),
    growth_steps = {
      at <- sort(sample.int(n, v$growth_steps, replace = TRUE))
      jumps <- numeric(n)
      jumps[at] <- rnorm(v$growth_steps, -0.01, v$growth_step_sd)
      v$r0 + cumsum(jumps)
    },
    growth_smooth = v$r0 + v$smooth_amp *
      sin(2 * pi * seq_len(n) / v$smooth_period + v$smooth_phase)
  )
  # Growth that never turns over is the main cause of outbreaks too large
  # to keep, so growth declines linearly to `turn` lower by the end.
  path <- path - v$growth_turn * seq_len(n) / n
  pmin(pmax(path, -0.1), 0.1)
}

# The delay's epinowcast parameters (`refp_mean_int`, `refp_sd_int`) and
# its probabilities on 0, ..., max_delay - 1 days, from a mean and a
# coefficient of variation. Gamma is shape exp(mean_int) and rate
# sd_int, the exponential rate exp(-mean_int). With no parametric delay,
# the non-parametric hazard's intercept is the constant hazard of a
# geometric delay with that mean.
delay_parameters <- function(family, mean, cv, max_delay) {
  d <- 0:max_delay
  out <- list()
  cdf <- switch(family,
    lognormal = {
      sdlog <- sqrt(log(1 + cv^2))
      meanlog <- log(mean) - sdlog^2 / 2
      out <- list(mean_int = meanlog, sd_int = sdlog)
      plnorm(d, meanlog, sdlog)
    },
    gamma = {
      shape <- 1 / cv^2
      out <- list(mean_int = log(shape), sd_int = shape / mean)
      pgamma(d, shape = shape, rate = shape / mean)
    },
    exponential = {
      out <- list(mean_int = log(mean))
      pexp(d, rate = 1 / mean)
    },
    none = {
      out <- list(hazard = 1 / (1 + mean))
      pgeom(d - 1, out$hazard)
    }
  )
  pmf <- diff(cdf)
  out$pmf <- pmf / sum(pmf)
  out
}

weekdays_drawn <- function() {
  # Onsets are reported every day or on one to three weekdays.
  n <- if (runif(1) < 0.5) 7L else sample(1:3, 1)
  if (n == 7L) {
    return("all")
  }
  days <- c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
  paste(sort(sample(days, n)), collapse = ";")
}

# The categorical choices of an outbreak, kept when a draw fails.
draw_design <- function() {
  strata <- sample(
    c("single", "places", "ages", "places_ages"), 1,
    prob = c(0.3, 0.4, 0.2, 0.1)
  )
  reference <- sample(
    c("parametric", "non_parametric", "both"), 1, prob = c(0.3, 0.2, 0.5)
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
  list(
    grid = grid_name,
    strata = strata,
    n_places = switch(strata,
      single = 1L, ages = 1L,
      places = sample(2:26, 1),
      places_ages = sample(6:16, 1)
    ),
    n_ages = switch(strata,
      single = 1L, places = 1L,
      ages = sample(2:8, 1),
      places_ages = sample(3:7, 1)
    ),
    process = sample(processes, 1),
    reference = reference,
    delay_family = if (parametric) {
      sample(c("lognormal", "gamma", "exponential"), 1)
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
    report_dow = runif(1) < 0.7,
    report_holiday = runif(1) < 0.3,
    report_week = sample(c("none", "week", "rw"), 1, prob = c(3, 1, 1)),
    report_days = weekdays_drawn(),
    report_change = change(0.3),
    report_days_after = weekdays_drawn(),
    family = sample(
      c("negbin", "negbin1d", "poisson"), 1, prob = c(0.4, 0.4, 0.2)
    ),
    # A long tail: the longest delay one step above what the delay's mean
    # needs (`max_delay_for`).
    long_tail = runif(1) < 0.25
  )
}

# The continuous settings of an outbreak, drawn again when a draw fails.
draw_values <- function(design) {
  places_ages <- design$strata == "places_ages"
  delay_mean <- exp(runif(1, log(2), log(21)))
  list(
    weeks = if (places_ages) sample(16:24, 1) else sample(16:40, 1),
    offset = sample(0:6, 1),
    quiet = runif(1, 0, 0.9),
    coupling = runif(1),
    gt_mean = runif(1, 2, 20),
    gt_cv = runif(1, 0.3, 0.7),
    incubation_mean = runif(1, 3, 12),
    incubation_cv = runif(1, 0.3, 0.7),
    log_r0 = rnorm(1, 0.1, 0.2),
    rt_turn = runif(1, 0, 0.6),
    rt_steps = rpois(1, 1.2),
    rt_step_sd = runif(1, 0.1, 0.5),
    rt_drift_sd = runif(1, 0, 0.03),
    # Most failed draws were growth processes overflowing at the start
    # growth rate's upper range, so it is narrower than for Rt.
    r0 = rnorm(1, 0.01, 0.02),
    growth_turn = runif(1, 0, 0.05),
    walk_sd = runif(1, 0.005, 0.015),
    incidence_sd = runif(1, 0.05, 0.3),
    growth_steps = sample(1:3, 1),
    growth_step_sd = runif(1, 0.02, 0.06),
    smooth_amp = runif(1, 0.01, 0.06),
    smooth_period = runif(1, 40, 200),
    smooth_phase = runif(1, 0, 2 * pi),
    seed_size = exp(runif(1, log(0.2), log(200))),
    stratum_sd = runif(1, 0.5, 2),
    delay_mean = delay_mean,
    max_delay = max_delay_for(delay_mean, design$long_tail),
    # Gamma delays with shape above 2 fail or report nothing in the
    # simulate branch, so the gamma's coefficient of variation is at
    # least 0.7.
    delay_cv = switch(design$delay_family,
      lognormal = runif(1, 0.3, 1.2),
      gamma = runif(1, 0.7, 1.2),
      1
    ),
    # Log changes in the delay's location and spread from the first week
    # to the last, as a linear drift.
    delay_change = rnorm(1, 0, 0.5),
    spread_change = rnorm(1, 0, 0.3),
    step_at_1 = runif(1, 0.1, 0.9),
    step_at_2 = runif(1, 0.1, 0.9),
    step_at_3 = runif(1, 0.1, 0.9),
    step_mean_1 = rnorm(1, 0, 0.4),
    step_mean_2 = rnorm(1, 0, 0.4),
    step_mean_3 = rnorm(1, 0, 0.4),
    step_spread_1 = rnorm(1, 0, 0.3),
    step_spread_2 = rnorm(1, 0, 0.3),
    step_spread_3 = rnorm(1, 0, 0.3),
    walk_mean_sd = runif(1, 0, 0.1),
    walk_spread_sd = runif(1, 0, 0.1),
    delay_stratum_sd = runif(1, 0, 0.4),
    hazard_sd = runif(1, 0, 0.4),
    hazard_week_sd = runif(1, 0, 0.4),
    dow_sd = runif(1, 0, 0.8),
    holidays = sample(2:6, 1),
    holiday_effect = runif(1, -2.5, -0.5),
    report_week_sd = runif(1, 0, 0.3),
    report_change_at = runif(1, 0.2, 0.8),
    sqrt_phi = runif(1, 0.05, 1.5)
  )
}

# The longest delay in days: the shortest of 4, 8 and 12 weeks that is at
# least ten times the delay's mean, or the next one up for a long tail.
# A longest delay many times the mean gives the discretised delay
# non-finite tail hazards in epinowcast (#939), so Stan cannot start.
max_delay_for <- function(mean, long_tail) {
  options <- c(28L, 56L, 84L)
  k <- min(which(c(options >= 10 * mean, TRUE))[1], length(options))
  options[min(k + long_tail, length(options))]
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
  expectation <- enw_expectation(
    r = ~1, generation_time = gt, latent_reporting_delay = lrd, data = pobs
  )
  n_r <- expectation$data$expr_t
  path <- function() {
    if (renewal) log_rt_path(n_r, v) else growth_path(n_r, design$process, v)
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
  size <- pmax(v$seed_size * weight / sum(weight), exp(-8))
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
    delay = "(1 | delay)",
    week = "(1 | week)",
    delay_week = c("(1 | delay)", "(1 | week)")
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
  delay <- delay_parameters(
    design$delay_family, v$delay_mean, v$delay_cv, max_delay
  )
  stratum_sd <- list(place = v$delay_stratum_sd, age = v$delay_stratum_sd)
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
  # A draw expected to be far beyond the largest size kept cannot pass,
  # so it fails before Stan runs.
  if (sum(onsets) > 50 * max_size(design)) {
    stop("too large (expected)")
  }
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
    seed = sample.int(.Machine$integer.max, 1), show_messages = debug,
    show_exceptions = debug, refresh = 0
  )))
  # The draws in the order `enw_pp_summary` binds them to the triangle,
  # read directly as summarising them is slow.
  pp <- data.table::copy(sims$new_confirm[[1]])
  data.table::setorderv(pp, c(".group", "reference_date"))
  pp[, mean := as.numeric(sims$fit[[1]]$draws("pp_obs", format = "matrix"))]
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
      outbreak = NA_character_, timestep = "day", location,
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
        size <- sum(res$counts$count)
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
    out$counts[, outbreak := id]
    settings[[length(settings) + 1L]] <- c(
      list(outbreak = id), design, out$values,
      list(
        tries = try, cases = sum(out$counts$count),
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
