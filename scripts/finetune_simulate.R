# Simulate outbreaks with epinowcast to fine-tune Chronos-2
# (src/bdbvhubrenewal/forecasters/finetune.py).
#
# Each outbreak is one draw of epinowcast's generative model with fixed
# parameters (`enw_simulate`): a renewal process driven by a given log Rt
# path, a reporting delay built from a lognormal with drift and a hazard
# on each delay, day-of-week reporting effects, and a negative binomial
# observation model. Settings are drawn from a grid of plausible values
# (`draw_settings`). The output is each outbreak's daily reporting
# triangle, a count by place, onset day and report date. The weekly
# training blocks sum it to onset weeks with the hub's weekly releases
# (`forecasters.finetune.releases`), so they share the daily outbreaks'
# generation times.
#
# `enw_simulate` is on the feature/829-simulate-forecast branch of
# epinowcast, pinned below. Install it into a local library first:
#
#   git -C ~/code/epinowcast/epinowcast archive e084b080 | tar -x -C src
#   R CMD INSTALL --library=output/finetune/rlib src
#
# Run from the repo root:
#
#   Rscript scripts/finetune_simulate.R <n> <seed> <out_dir>

epinowcast_sha <- "e084b080"
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
enw_set_cache(Sys.getenv("ENW_CACHE", "output/finetune/enwcache"), type = "session")
model <- suppressMessages(enw_model(threads = FALSE))

# Discretised gamma on 1, ..., n steps.
gamma_pmf <- function(mean, cv, n) {
  shape <- 1 / cv^2
  cdf <- pgamma(0:n, shape = shape, rate = shape / mean)
  pmf <- diff(cdf)
  pmf / sum(pmf)
}

# One log Rt path: steps at random times plus a random walk drift.
log_rt_path <- function(n, start, n_steps, step_sd, drift_sd) {
  at <- sort(sample.int(n, n_steps, replace = TRUE))
  jumps <- numeric(n)
  jumps[at] <- rnorm(n_steps, -0.2, step_sd)
  pmin(pmax(start + cumsum(jumps) + cumsum(rnorm(n, 0, drift_sd)), -1.2), 1.2)
}

draw_settings <- function(id) {
  multi <- runif(1) < 0.6
  list(
    outbreak = id,
    timestep = "day",
    # Onsets are reported every day or on one to three weekdays.
    report_days = if (runif(1) < 0.5) 7L else sample(1:3, 1),
    n_places = if (multi) sample(2:26, 1) else 1L,
    coupling = runif(1),
    weeks = sample(16:40, 1),
    offset = sample(0:6, 1),
    gt_mean = runif(1, 6, 20),
    gt_cv = runif(1, 0.3, 0.7),
    log_r0 = rnorm(1, 0.1, 0.2),
    n_steps = rpois(1, 1.2),
    step_sd = runif(1, 0.1, 0.5),
    drift_sd = runif(1, 0, 0.03),
    seed_size = exp(runif(1, log(0.5), log(20))),
    place_sd = runif(1, 0.5, 1.5),
    delay_mean = exp(runif(1, log(2), log(21))),
    delay_sd = runif(1, 0.3, 1.1),
    delay_drift = rnorm(1, 0, 0.02),
    delay_place_sd = runif(1, 0, 0.4),
    hazard_sd = runif(1, 0, 0.4),
    dow_sd = runif(1, 0, 0.8),
    sqrt_phi = runif(1, 0.05, 1)
  )
}

simulate_outbreak <- function(s) {
  max_delay <- 56L
  # Twelve weeks past the last cut-off, so a block's horizon (four weeks)
  # is complete to the longest delay (eight weeks).
  n_time <- 7L * (s$weeks + 12L)
  places <- sprintf("P%02d", seq_len(s$n_places))
  dates <- as.IDate("2026-01-03") + s$offset + seq_len(n_time) - 1L
  grid <- CJ(location = places, reference_date = dates, delay = 0:(max_delay - 1))
  grid[, report_date := reference_date + delay][, delay := NULL]
  grid <- grid[report_date <= max(dates)][, confirm := 0L]
  pobs <- suppressMessages(suppressWarnings(enw_preprocess_data(
    grid, by = "location", max_delay = max_delay,
    timestep = s$timestep
  )))

  gt <- gamma_pmf(s$gt_mean, s$gt_cv, 45L)
  expectation <- enw_expectation(r = ~1, generation_time = gt, data = pobs)
  n_r <- expectation$data$expr_t
  path <- function() {
    log_rt_path(n_r, s$log_r0, s$n_steps, s$step_sd, s$drift_sd)
  }
  shared <- path()
  log_rt <- unlist(lapply(places, function(p) {
    s$coupling * shared + (1 - s$coupling) * path()
  }))
  weight <- exp(rnorm(s$n_places, 0, s$place_sd))
  size <- s$seed_size * weight / sum(weight)
  lelatent <- matrix(
    log(size), nrow = expectation$data$expr_gt_n, ncol = s$n_places,
    byrow = TRUE
  )

  multi <- s$n_places > 1
  reference <- enw_reference(
    parametric = if (multi) ~ 1 + week + (1 | location) else ~ 1 + week,
    non_parametric = ~ 0 + (1 | delay), data = pobs
  )
  fcols <- colnames(reference$data$refp_fdesign)
  refp_beta <- rnorm(length(fcols))
  refp_beta[fcols == "week"] <- 7 * s$delay_drift
  structural <- if (s$report_days < 7L) {
    enw_dayofweek_structural_reporting(
      pobs, day_of_week = sample(
        c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday"),
        s$report_days
      )
    )
  }
  report <- enw_report(
    non_parametric = ~ 0 + (1 | day_of_week), structural = structural,
    data = pobs
  )
  parameters <- list(
    refp_mean_int = log(s$delay_mean),
    refp_sd_int = s$delay_sd,
    refp_mean_beta = array(refp_beta),
    refp_sd_beta = array(numeric(length(fcols))),
    refnp_beta = array(rnorm(reference$data$refnp_fncol)),
    refnp_beta_sd = array(s$hazard_sd),
    sqrt_phi = s$sqrt_phi,
    expr_lelatent_int = lelatent,
    rep_beta = array(rnorm(report$data$rep_fncol)),
    rep_beta_sd = array(s$dow_sd)
  )
  if (multi) {
    parameters$refp_mean_beta_sd <- array(s$delay_place_sd)
    parameters$refp_sd_beta_sd <- array(0.01)
  }
  sims <- suppressMessages(suppressWarnings(enw_simulate(
    pobs, growth_rate = log_rt, parameters = parameters,
    reference = reference, report = report, expectation = expectation,
    obs = enw_obs(family = "negbin", data = pobs), model = model,
    seed = sample.int(.Machine$integer.max, 1), show_messages = FALSE,
    show_exceptions = FALSE, refresh = 0
  )))
  # The draws in the order `enw_pp_summary` binds them to the triangle,
  # read directly as summarising them is slow.
  pp <- data.table::copy(sims$new_confirm[[1]])
  data.table::setorderv(pp, c(".group", "reference_date"))
  pp[, mean := as.numeric(sims$fit[[1]]$draws("pp_obs", format = "matrix"))]
  pp[mean > 0, .(
    outbreak = s$outbreak, timestep = s$timestep, location,
    reference_date, report_date, count = as.integer(mean)
  )]
}

set.seed(seed)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
# Outbreaks outside `min_size` to `max_size` cases are drawn again.
min_size <- 50
max_size <- 30000
t0 <- Sys.time()
settings <- list()
counts <- list()
for (i in seq_len(n_outbreaks)) {
  tries <- 0L
  repeat {
    tries <- tries + 1L
    s <- draw_settings(sprintf("%d-%04d", seed, i))
    # A draw that overflows the observation model fails, and is redrawn.
    out <- tryCatch(simulate_outbreak(s), error = function(e) NULL)
    size <- if (is.null(out)) Inf else sum(out$count)
    if (size >= min_size && size <= max_size) break
  }
  s$tries <- tries
  s$cases <- size
  settings[[i]] <- s
  counts[[i]] <- out
}
counts <- rbindlist(counts)
message(sprintf(
  "%d outbreaks in %.1f s (epinowcast %s)", n_outbreaks,
  as.numeric(Sys.time() - t0, units = "secs"), epinowcast_sha
))
fwrite(counts, file.path(out_dir, sprintf("counts-%d.csv.gz", seed)))
fwrite(rbindlist(settings), file.path(out_dir, sprintf("settings-%d.csv", seed)))
