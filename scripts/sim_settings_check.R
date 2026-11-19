# Summaries of simulated outbreaks drawn from the sampling distributions
# of the simulation settings (scripts/finetune_simulate.R), to compare
# with real data (scripts/sim_settings_real.py). No model is fitted: each
# draw is a grid cell and its settings, with the trajectory, delay and
# noise they imply.
#
# Run from the repo root:
#
#   Rscript scripts/sim_settings_check.R <n draws> <out.csv>
#
# Each row is one draw. Summaries per stratum (a place, or a place's age
# group) assume its size follows the trajectory's shape.

args <- commandArgs(trailingOnly = TRUE)
n_draws <- as.integer(args[[1]])
out_file <- args[[2]]

suppressPackageStartupMessages(library(data.table))

# The simulator's functions and settings, without epinowcast.
for (e in parse("scripts/finetune_simulate.R")) {
  if (!is.call(e) || !identical(e[[1]], as.name("<-"))) next
  rhs <- e[[3]]
  is_function <- is.call(rhs) && identical(rhs[[1]], as.name("function"))
  if (!is_function &&
    grepl("enw_|commandArgs|args\\[\\[", paste(deparse(rhs), collapse = ""))) {
    next
  }
  eval(e, globalenv())
}

set.seed(20261119)
cells <- sample.int(block, n_draws, replace = n_draws > block) - 1L
rows <- lapply(cells, function(g) {
  design <- draw_design(g)
  v <- draw_values(design)
  n <- design$weeks * 7L
  r <- trajectory_path(n, design$trajectory, v)
  shape <- exp(cumsum(r))
  shape <- shape / sum(shape)
  # Strata with cases share the outbreak's size.
  strata <- max(1, (design$n_places - length(v$quiet_places)) *
    design$n_ages)
  per_stratum <- v$size_per_stratum
  peak <- per_stratum * max(shape)
  lambda_hat <- max(1, median(per_stratum * shape[per_stratum * shape >= 1]),
    na.rm = TRUE
  )
  family <- if (design$delay_form == "non_parametric") {
    "none"
  } else {
    design$delay_family
  }
  delay <- delay_parameters(family, v, v$max_delay)
  pmf <- delay$pmf
  cdf <- cumsum(pmf)
  d <- seq_along(pmf) - 1L
  k <- if (design$observation == "negbin1d") {
    v$sqrt_phi^2 * lambda_hat * sum(pmf^2)
  } else {
    0
  }
  dow <- exp(rnorm(7, 0, v$dow_sd))
  ref_dow <- exp(rnorm(7, 0, v$reference_dow_sd))
  grow <- max(r)
  data.table(
    cell = g,
    trajectory = design$trajectory,
    strata = strata,
    weeks = design$weeks,
    max_growth = grow,
    doubling_days = if (grow > 0) log(2) / grow else Inf,
    total_per_stratum = per_stratum,
    peak_per_stratum = peak,
    total_outbreak = per_stratum * strata,
    peak_outbreak = peak * strata,
    delay_mean = sum(d * pmf),
    delay_q95 = d[which(cdf >= 0.95)[1]],
    reported_7 = cdf[min(8L, length(cdf))],
    reported_14 = cdf[min(15L, length(cdf))],
    reported_28 = cdf[min(29L, length(cdf))],
    # Coefficient of variation of a day's reports at the peak, around its
    # expectation: Poisson plus the NB1 excess on the NB2 scale (#937).
    cv_at_peak = sqrt((1 + k) / peak),
    nb1_k = k,
    report_dow_ratio = if (design$reporting == "daily") {
      max(dow) / min(dow)
    } else {
      NA_real_
    },
    onset_dow_ratio = if (design$reference_dow == "yes") {
      max(ref_dow) / min(ref_dow)
    } else {
      1
    },
    shocks = design$shocks,
    n_shocks = length(v$shock_places),
    shock_mult = if (length(v$shock_mult)) max(v$shock_mult) else NA_real_,
    shock_weeks = if (length(v$shock_weeks)) max(v$shock_weeks) else NA,
    delay_change = design$delay_change,
    sqrt_phi = v$sqrt_phi,
    observation = design$observation
  )
})
out <- rbindlist(rows)
fwrite(out, out_file)
cat("wrote", nrow(out), "draws to", out_file, "\n")
