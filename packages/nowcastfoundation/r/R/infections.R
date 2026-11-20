# Mixing between places on `n` days: the share `eps[t]` of each place's
# infection pressure that comes from other places, by row-normalised
# kernel `K` between strata (each age mixes with the same age elsewhere).
# Kinds: none; symmetric (every other place alike); asymmetric (only from
# places earlier in `mixing_order`); distance (decaying with distance
# between places at `mixing_x`, `mixing_y` over `mixing_range`); and
# time-varying (the distance kernel switching on or off over
# `mixing_switch_days` from a share `mixing_switch_at` of the way
# through). Places without cases neither give nor take infections.
place_mixing <- function(n, strata, design, v) {
  places <- sort(unique(strata$place))
  n_p <- length(places)
  kernel <- matrix(0, n_p, n_p)
  if (n_p > 1 && design$mixing != "none") {
    kernel <- switch(design$mixing,
      symmetric = 1 - diag(n_p),
      asymmetric = {
        at <- match(seq_len(n_p), v$mixing_order[v$mixing_order <= n_p])
        1 * outer(at, at, ">")
      },
      {
        xy <- cbind(v$mixing_x[seq_len(n_p)], v$mixing_y[seq_len(n_p)])
        exp(-as.matrix(dist(xy)) / v$mixing_range) * (1 - diag(n_p))
      }
    )
    quiet <- v$quiet_places[v$quiet_places <= n_p]
    kernel[quiet, ] <- 0
    kernel[, quiet] <- 0
    from <- rowSums(kernel) > 0
    kernel[from, ] <- kernel[from, ] / rowSums(kernel)[from]
  }
  idx <- match(strata$place, places)
  K <- kernel[idx, idx, drop = FALSE] * outer(strata$age, strata$age, "==")
  eps <- if (design$mixing == "none") {
    numeric(n)
  } else if (design$mixing == "time_varying") {
    start <- round(v$mixing_switch_at * n)
    ramp <- pmin(pmax((seq_len(n) - start) / v$mixing_switch_days, 0), 1)
    v$mixing_eps * if (v$mixing_switch == "on") ramp else 1 - ramp
  } else {
    rep(v$mixing_eps, n)
  }
  list(K = K, eps = eps)
}

# Strata of places seeded mid-outbreak, for `design$seeding` "mid".
late_strata <- function(strata, design, v) {
  design$seeding == "mid" &
    strata$place %in% sprintf("P%02d", v$late_places)
}

# Imports into each stratum on `n` days, in units of the seed `share`.
# A late place is imported its share over `late_days` from a share
# `late_at` of the way through. Other strata with cases are imported a
# constant `import_rate` of their share a day, or pulses of
# `import_size` times it at shares `import_at` of the way through.
stratum_imports <- function(n, share, strata, design, v) {
  out <- matrix(0, n, nrow(strata))
  late <- late_strata(strata, design, v)
  if (any(late)) {
    from <- max(1L, round(v$late_at * n))
    days <- from:min(n, from + v$late_days - 1L)
    out[days, late] <- rep(share[late] / length(days), each = length(days))
  }
  on <- !late & share > 0
  if (design$imports == "constant") {
    out[, on] <- rep(v$import_rate * share[on], each = n)
  } else if (design$imports == "pulses") {
    days <- unique(pmax(1L, round(v$import_at * n)))
    out[days, on] <- out[days, on] +
      rep(v$import_size * share[on], each = length(days))
  }
  out
}

# A generation time on 1, ..., n days, n its 99.9% quantile, from its
# family (gamma or lognormal), mean and CV.
gt_pmf <- function(mean, family, cv) {
  if (family == "gamma") {
    k <- 1 / cv^2
    cdf <- function(d) pgamma(d, k, k / mean)
    q <- qgamma(0.999, k, k / mean)
  } else {
    sdlog <- sqrt(log(1 + cv^2))
    meanlog <- log(mean) - sdlog^2 / 2
    cdf <- function(d) plnorm(d, meanlog, sdlog)
    q <- qlnorm(0.999, meanlog, sdlog)
  }
  pmf <- diff(cdf(0:max(2L, as.integer(ceiling(q)))))
  pmf / sum(pmf)
}

# `n` weeks of an AR(1) walk on log R around 0 (R = 1).
log_r_walk <- function(grid, n) {
  e <- draw_setting(grid, "walk_r_step", n)
  x <- e
  for (w in seq_len(n)[-1]) x[w] <- grid$constants$walk_ar * x[w - 1] + e[w]
  x
}

# A weekly walk read on `days` (0 the first), linear between weeks.
daily_walk <- function(days, walk) {
  x <- days / 7
  w <- floor(x)
  f <- x - w
  n <- length(walk)
  walk[pmin(w + 1, n)] * (1 - f) + walk[pmin(w + 2, n)] * f
}

# Per-stratum values of setting `name` for `n` strata, as `structure`
# says: shared (one value), pooled (the shared value times
# LogNormal(0, tau), tau half-normal; on the logit scale for a
# `probability`) or independent (each its own draw).
stratum_values <- function(grid, structure, n, name, probability = FALSE) {
  switch(structure,
    shared = rep(draw_setting(grid, name), n),
    pooled = {
      shared <- draw_setting(grid, name)
      dev <- draw_setting(grid, "pool_tau") * draw_setting(grid, "pool_z", n)
      if (probability) plogis(qlogis(shared) + dev) else shared * exp(dev)
    },
    independent = draw_setting(grid, name, n),
    stop("unknown strata structure ", structure)
  )
}

# Log R of a path on `n` days: the shapes on log R, the weekly walk
# around R = 1 (endemic), or a start at the amplitude decaying onto that
# walk (an early outbreak brought under control).
path_log_r <- function(n, kind, v) {
  t <- seq_len(n) - 1
  u <- t / n
  a <- v$amplitude
  switch(kind,
    endemic = daily_walk(t, v$walk_r),
    controlled = a * exp(-t / v$control_days) + daily_walk(t, v$walk_r),
    growth = rep(a, n),
    decline = rep(-a, n),
    flat = rep(v$flat, n),
    rise_fall_gradual = a * (v$peak - u) / max(v$peak, 1 - v$peak),
    fall_rise = -a * (v$peak - u) / max(v$peak, 1 - v$peak),
    rise_fall_sharp = a *
      (1 - 2 * plogis((t - v$peak * n) / (v$flip_days / 4.4))),
    oscillating = a * sin(2 * pi * t / v$period + v$phase),
    multi_wave = {
      k <- pmin(
        length(v$wave_scale), floor(v$waves * u + v$phase / (2 * pi)) + 1
      )
      2 * a * v$wave_scale[k] * sin(2 * pi * v$waves * u + v$phase)
    }
  )
}

# Each stratum's log R on `n` days, with places related as
# `design$place_rt` says (as `place_growth`), and a slow drift. Noise on
# daily growth becomes noise on log R through the generation time's mean.
strata_log_r <- function(grid, n, strata, design, v) {
  places <- sort(unique(strata$place))
  scale <- v$gt_mean
  shared <- path_log_r(n, v$path, v)
  own_path <- function() {
    if (v$path %in% c("endemic", "controlled")) {
      v$path
    } else {
      sample(names(grid$factors$trajectory), 1)
    }
  }
  paths <- switch(design$place_rt,
    shared = rep(list(shared), length(places)),
    independent = c(list(shared), lapply(places[-1], function(p) {
      path_log_r(n, own_path(), v)
    })),
    correlated = {
      common <- weekly_noise(n, scale * v$place_sd)
      lapply(places, function(p) {
        shared + sqrt(v$place_rho) * common +
          sqrt(1 - v$place_rho) * weekly_noise(n, scale * v$place_sd)
      })
    },
    lagged = lapply(seq_along(places), function(i) {
      k <- v$place_lags[i]
      c(rep(shared[1], k), shared)[seq_len(n)]
    })
  )
  path <- paths[match(strata$place, places)]
  drift <- log(v$r_drift) * (seq_len(n) - 1) / n
  matrix(vapply(seq_len(nrow(strata)), function(i) {
    path[[i]] + drift + (1 - v$coupling) * weekly_noise(n, 0.01 * scale)
  }, numeric(n)), nrow = n)
}

# Expected infections of each stratum: a renewal with depletion,
# lambda[t] = R[t] S[t] / N M[t] sum_s gt[s] lambda[t - s] + imports[t],
# at most S[t], from `start` (the generation time's span before the
# window), with mixing as `expected_onsets`.
# - Shift j moves R by exp(+-size[j]) along a logistic over `days[j]`
#   centred on `at[j]`. Its direction is set when it starts: a reactive
#   shift lowers R where effective R is above 1 and raises it where
#   below; the others go `up` or down.
# - Variant j, on day `at[j]`, returns a share `escape[j]` of the immune
#   (N - S) to susceptible and raises R by `mult[j]` along a logistic over
#   `days[j]`.
# Returns the infections (with `start`), R, effective R and the
# susceptible after each day.
renewal_infections <- function(start, gt, log_r, population, mixing = NULL,
                          imports = NULL, shifts = NULL, variants = NULL) {
  n_gt <- length(gt)
  n <- nrow(log_r)
  k <- ncol(log_r)
  out <- matrix(0, n_gt + n, k)
  out[seq_len(n_gt), ] <- start
  susceptible <- pmax(population - colSums(out), 0)
  rev_gt <- rev(gt)
  if (!is.null(mixing)) from <- rowSums(mixing$K)
  n_sh <- length(shifts$at)
  mult <- matrix(NA_real_, n_sh, k)
  escaped <- rep(FALSE, length(variants$at))
  r <- reff <- left <- matrix(0, n, k)
  prev <- rep(1, k)
  for (t in seq_len(n)) {
    R <- exp(log_r[t, ])
    for (j in seq_len(n_sh)) {
      if (is.na(mult[j, 1]) && t >= shifts$at[j] - shifts$days[j] / 2) {
        up <- if (shifts$reactive[j]) prev < 1 else rep(shifts$up[j], k)
        mult[j, ] <- exp(ifelse(up, 1, -1) * shifts$size[j])
      }
      if (!is.na(mult[j, 1])) {
        R <- R * (1 + (mult[j, ] - 1) *
          plogis((t - shifts$at[j]) / (shifts$days[j] / 4.4)))
      }
    }
    for (j in seq_along(variants$at)) {
      if (!escaped[j] && t >= variants$at[j]) {
        susceptible <- susceptible +
          variants$escape[j] * (population - susceptible)
        escaped[j] <- TRUE
      }
      R <- R * (1 + (variants$mult[j] - 1) *
        plogis((t - variants$at[j]) / (variants$days[j] / 4.4)))
    }
    past <- out[t:(t + n_gt - 1), , drop = FALSE]
    force <- colSums(past * rev_gt)
    if (!is.null(mixing) && mixing$eps[t] > 0) {
      force <- force +
        mixing$eps[t] * (drop(mixing$K %*% force) - from * force)
    }
    new <- R * susceptible / population * force
    if (!is.null(imports)) new <- new + imports[t, ]
    new <- pmin(new, susceptible)
    susceptible <- susceptible - new
    out[n_gt + t, ] <- new
    r[t, ] <- R
    reff[t, ] <- R * susceptible / population
    left[t, ] <- susceptible
    prev <- reff[t, ]
  }
  list(infections = out, r = r, reff = reff, susceptible = left)
}

# Expected onsets (ascertained infections) on the generation time's
# span plus `n` days, for `simulate_outbreak`. Strata weights are
# drawn (`stratum_sd`, quiet places none). Each place's population is split over
# its ages by their weights. An early window starts each stratum with
# cases from a seed on the last day before it (at least 1); an endemic or
# established one runs at the drawn level over the span. Late places,
# the import kinds (in units of the seed or level) and a trickle of
# imports into every stratum with cases come in as imports. Returns the
# onsets, the infections and the values with each stratum's population
# and the attack rate added.
latent_onsets <- function(grid, n, gt, strata, design, v) {
  if (n < 1) stop("the generation time is longer than the series")
  place_ids <- sprintf("P%02d", seq_len(design$n_places))
  age_ids <- sprintf("A%d", seq_len(design$n_ages))
  place_weight <- exp(rnorm(design$n_places, 0, v$stratum_sd))
  place_weight[v$quiet_places] <- 0
  age_weight <- exp(rnorm(design$n_ages, 0, v$stratum_sd / 2))
  at_place <- match(strata$place, place_ids)
  at_age <- match(strata$age, age_ids)
  weight <- place_weight[at_place] * age_weight[at_age]
  active <- weight > 0
  rel <- weight / mean(weight[active])
  population <- v$population[at_place] *
    (age_weight / sum(age_weight))[at_age]
  late <- late_strata(strata, design, v)
  on <- active & !late
  n_gt <- length(gt)
  start <- matrix(0, n_gt, nrow(strata))
  early <- design$window == "early"
  if (early) {
    start[n_gt, on] <- pmax(1, v$seed * rel[on])
  } else {
    start[, on] <- rep(v$level * rel[on], each = n_gt)
  }
  if (any(colSums(start) > population)) {
    stop("an established epidemic larger than its population")
  }
  unit <- (if (early) v$seed else v$level) * rel
  imports <- stratum_imports(n, unit, strata, design, v)
  imports[, on] <- imports[, on] + v$trickle
  shifts <- list(
    at = v$shift_at * n, size = v$shift_size, days = v$shift_days,
    reactive = v$shift_reactive, up = v$shift_up
  )
  variants <- list(
    at = v$variant_at * n, escape = v$variant_escape, mult = v$variant_r,
    days = v$variant_days
  )
  run <- renewal_infections(
    start, gt, strata_log_r(grid, n, strata, design, v), population,
    mixing = place_mixing(n, strata, design, v), imports = imports,
    shifts = shifts, variants = variants
  )
  v$stratum_population <- population
  v$attack <- sum(run$infections[, active]) / sum(population[active])
  # Layer 2: infections to reference events, then ascertained.
  # The outbreak's latent path: R and effective R weighted by each day's
  # infections (by population on days with none).
  inf <- run$infections[-seq_len(n_gt), , drop = FALSE]
  w <- inf / rowSums(inf)
  none <- !is.finite(rowSums(w))
  w[none, ] <- rep(population / sum(population), each = sum(none))
  pad <- rep(NA_real_, n_gt)
  asc <- v$ascertainment[strata$.group]
  pmfs <- lapply(v$incubation_mean[strata$.group], incubation_pmf,
    cv = grid$constants$incubation_cv
  )
  onsets <- sweep(convolve_rows(run$infections, pmfs), 2, asc, `*`)
  list(
    onsets = onsets,
    infections = run$infections,
    latent = data.table::data.table(
      day = seq_len(n_gt + n) - 1L,
      infections = rowSums(run$infections),
      onsets = rowSums(onsets),
      r = c(pad, rowSums(run$r * w)),
      reff = c(pad, rowSums(run$reff * w))
    ),
    values = v
  )
}
