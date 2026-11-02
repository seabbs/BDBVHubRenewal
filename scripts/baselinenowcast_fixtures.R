# Generate test fixtures from the R package baselinenowcast 0.2.2.
#
# The JAX port (src/bdbvhubrenewal/baselinenowcast/) is checked against
# these fixtures. Run from the repo root:
#
#   Rscript scripts/baselinenowcast_fixtures.R
#
# Outputs go to tests/baselinenowcast/fixtures/<case>/ as plain CSV
# matrices without a header. NA marks an unobserved cell.
# Numbers are written with 17 significant digits.
#
# Cases are toy, example (the package example triangle), neg (negative
# cells and zeros), struct2 (reporting structure 2) and hub (a daily
# triangle with a maximum delay of 84). The rel case checks the release
# extension against an independent R implementation of the E-step and
# the EM loop, with estimate_delay as the M-step.

pinned_version <- "0.2.2"

suppressPackageStartupMessages({
  library(baselinenowcast)
  library(zoo)
})

if (as.character(packageVersion("baselinenowcast")) != pinned_version) {
  stop(
    "baselinenowcast ", pinned_version, " is required, found ",
    packageVersion("baselinenowcast")
  )
}

root <- file.path("tests", "baselinenowcast", "fixtures")
dir.create(root, recursive = TRUE, showWarnings = FALSE)
writeLines(
  paste("baselinenowcast", pinned_version),
  file.path(root, "VERSION")
)

fmt <- function(v) ifelse(is.na(v), "NA", sprintf("%.17g", v))

wm <- function(m, dir, name) {
  m <- unclass(m)
  if (is.null(dim(m))) m <- matrix(m, ncol = 1)
  out <- matrix(fmt(m), nrow(m), ncol(m))
  write.table(
    out, file.path(dir, paste0(name, ".csv")),
    sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE
  )
}

as_rt <- function(m) {
  m <- unclass(m)
  dimnames(m) <- NULL
  as_reporting_triangle(
    m,
    delays_unit = "days",
    reference_dates = as.Date("2026-01-01") + seq_len(nrow(m)) - 1
  )
}

roll <- function(k) {
  function(x) zoo::rollsum(x, k = k, align = "right")
}

quiet <- function(expr) {
  suppressWarnings(suppressMessages(expr))
}

# ---- triangles -------------------------------------------------------

make_triangles <- function() {
  set.seed(20260101)
  toy <- rbind(
    c(30, 20, 10, 5),
    c(28, 22, 9, 6),
    c(35, 18, 12, 4),
    c(40, 25, 11, 7),
    c(33, 21, 10, NA),
    c(29, 19, NA, NA),
    c(38, NA, NA, NA),
    c(36, NA, NA, NA)
  )
  toy[7, 2:4] <- NA
  toy[8, 2:4] <- NA
  toy[6, 3:4] <- NA
  toy[5, 4] <- NA
  toy[4, 4] <- 7

  ex <- unclass(baselinenowcast::example_reporting_triangle)
  dimnames(ex) <- NULL

  neg <- rbind(
    c(20, 12, -2, 4, 1),
    c(25, 10, 6, -3, 0),
    c(18, 14, 5, 2, -1),
    c(30, 0, 8, 3, 2),
    c(22, 16, -4, 5, NA),
    c(27, 11, 7, NA, NA),
    c(24, 13, NA, NA, NA),
    c(31, NA, NA, NA, NA),
    c(26, NA, NA, NA, NA)
  )
  neg[5, 5] <- NA

  full <- matrix(rpois(12 * 6, lambda = rep(c(40, 20, 10, 6, 3, 1), each = 12)),
    12, 6)
  s2 <- unclass(apply_reporting_structure(as_rt(full), structure = 2))
  dimnames(s2) <- NULL

  n_ref <- 262
  max_delay <- 84
  pmf <- diff(pgamma(0:(max_delay + 1), shape = 2.2, rate = 0.12))
  pmf <- pmf / sum(pmf)
  level <- 80 * exp(0.012 * seq_len(n_ref)) *
    (1 + 0.3 * sin(seq_len(n_ref) / 9))
  hub <- t(sapply(seq_len(n_ref), function(i) {
    rpois(max_delay + 1, level[i] * pmf)
  }))
  hub <- matrix(as.numeric(hub), n_ref, max_delay + 1)
  hub[3, 5] <- -2
  hub[40, 30] <- -3
  hub[40, 31] <- 1
  hub_obs <- unclass(
    apply_reporting_structure(as_rt(hub), structure = 1)
  )
  dimnames(hub_obs) <- NULL

  list(
    toy = list(tri = toy, n_hist = 5, n_retro = 3, k = 2),
    example = list(tri = ex, n_hist = 5, n_retro = 2, k = 2),
    neg = list(tri = neg, n_hist = 6, n_retro = 3, k = 2),
    struct2 = list(tri = s2, n_hist = 7, n_retro = 4, k = 2),
    hub = list(tri = hub_obs, n_hist = 126, n_retro = 126, k = 7)
  )
}

draw_n <- c(toy = 4000, example = 4000, neg = 4000, struct2 = 4000, hub = 2000)

run_case <- function(case, spec) {
  dir <- file.path(root, case)
  dir.create(dir, showWarnings = FALSE)
  tri <- quiet(as_rt(spec$tri))
  wm(spec$tri, dir, "tri")
  n_hist <- spec$n_hist
  n_retro <- spec$n_retro
  writeLines(
    as.character(c(n_hist, n_retro, spec$k)),
    file.path(dir, "settings.txt")
  )

  pre <- quiet(preprocess_negative_values(tri))
  wm(pre, dir, "preprocess")

  tr <- quiet(truncate_to_rows(tri, n = 3))
  for (i in seq_along(tr)) wm(tr[[i]], dir, paste0("trunc_", i))
  st <- quiet(get_reporting_structure(tri))
  wm(matrix(st, ncol = 1), dir, "structure")
  ap <- quiet(apply_reporting_structure(tr[[1]], structure = st))
  wm(ap, dir, "applied_1")
  ap2 <- quiet(apply_reporting_structure(tr[[2]], structure = 1))
  wm(ap2, dir, "applied_struct1")

  grid <- expand.grid(
    scale_factor = c(3, 1, 2, 0.5),
    prop_delay = c(0.5, 0.3, 0.8),
    n_min = c(2, 0)
  )
  res <- t(vapply(seq_len(nrow(grid)), function(i) {
    out <- tryCatch(
      quiet(allocate_reference_times(
        tri,
        scale_factor = grid$scale_factor[i],
        prop_delay = grid$prop_delay[i],
        n_min_retro_nowcasts = grid$n_min[i]
      )),
      error = function(e) list(n_history_delay = NA, n_retrospective_nowcasts = NA)
    )
    c(grid$scale_factor[i], grid$prop_delay[i], grid$n_min[i],
      out$n_history_delay, out$n_retrospective_nowcasts)
  }, numeric(5)))
  wm(res, dir, "alloc")

  pmf_raw <- quiet(estimate_delay(tri, n = n_hist))
  wm(pmf_raw, dir, "pmf_raw")
  pmf <- quiet(estimate_delay(pre, n = n_hist))
  wm(pmf, dir, "pmf")
  pmf_all <- quiet(estimate_delay(pre, n = nrow(pre)))
  wm(pmf_all, dir, "pmf_all")
  pt <- quiet(apply_delay(pre, pmf))
  wm(pt, dir, "apply")
  eaa <- quiet(estimate_and_apply_delay(pre, n = n_hist))
  wm(eaa, dir, "eaa")

  cap <- NULL
  capture <- function(obs, pred) {
    cap <<- list(obs = obs, pred = pred)
    fit_by_horizon(obs, pred)
  }
  size_id <- quiet(estimate_uncertainty_retro(
    pre,
    n_history_delay = n_hist, n_retrospective_nowcasts = n_retro,
    uncertainty_model = capture
  ))
  wm(cap$obs, dir, "retro_obs_id")
  wm(cap$pred, dir, "retro_pred_id")
  wm(quiet(estimate_uncertainty_retro(
    pre,
    n_history_delay = n_hist, n_retrospective_nowcasts = n_retro
  )), dir, "size_id")

  k <- spec$k
  size_agg <- quiet(estimate_uncertainty_retro(
    pre,
    n_history_delay = n_hist, n_retrospective_nowcasts = n_retro,
    ref_time_aggregator = roll(k),
    uncertainty_model = capture
  ))
  wm(cap$obs, dir, "retro_obs_agg")
  wm(cap$pred, dir, "retro_pred_agg")
  wm(quiet(estimate_uncertainty_retro(
    pre,
    n_history_delay = n_hist, n_retrospective_nowcasts = n_retro,
    ref_time_aggregator = roll(k)
  )), dir, "size_agg")

  point <- quiet(baselinenowcast(
    tri, output_type = "point",
    scale_factor = 3, prop_delay = 0.5
  ))
  wm(point$pred_count, dir, "point")

  set.seed(1)
  n_draws <- draw_n[[case]]
  smp <- quiet(baselinenowcast(
    tri, output_type = "samples", draws = n_draws,
    scale_factor = 3, prop_delay = 0.5
  ))
  stats <- function(df) {
    by_ref <- split(df$pred_count, df$reference_date)
    cbind(
      mean = vapply(by_ref, mean, numeric(1)),
      var = vapply(by_ref, var, numeric(1))
    )
  }
  wm(stats(smp), dir, "samples_id")
  smp_agg <- quiet(baselinenowcast(
    tri, output_type = "samples", draws = n_draws,
    scale_factor = 3, prop_delay = 0.5,
    ref_time_aggregator = roll(k)
  ))
  wm(stats(smp_agg), dir, "samples_agg")
  writeLines(as.character(n_draws), file.path(dir, "draws.txt"))

  # The sampler and horizon sizes used by the baselinenowcast call.
  invisible(size_id)
}

triangles <- make_triangles()
for (case in names(triangles)) {
  message("case ", case)
  run_case(case, triangles[[case]])
}

# ---- fit_nb and sample_nb ---------------------------------------------

dir <- file.path(root, "nb")
dir.create(dir, showWarnings = FALSE)
set.seed(7)
fits <- list(
  list(x = c(5, 9, 3, 12, 7, 6), mu = c(6, 8, 4, 10, 7, 7)),
  list(x = c(0, 0, 1, 0, 2), mu = c(0.5, 1, 1, 0.2, 2)),
  list(x = rnbinom(30, mu = 40, size = 5), mu = rep(40, 30)),
  list(x = rnbinom(30, mu = 400, size = 20), mu = rep(400, 30)),
  list(x = c(0, 0, 0), mu = c(0, 0, 0)),
  list(x = rpois(40, 100), mu = rep(100, 40)),
  list(x = c(3, 80, 1, 40), mu = c(30, 30, 30, 30))
)
for (i in seq_along(fits)) {
  f <- fits[[i]]
  wm(cbind(f$x, f$mu), dir, paste0("fit_in_", i))
  wm(fit_nb(f$x, f$mu), dir, paste0("fit_out_", i))
}
# Moments of rnbinom for the sampler check.
set.seed(2)
mu <- c(0, 0.5, 5, 50, 500)
size <- c(1, 0.3, 2, 10, 100)
wm(cbind(mu, size), dir, "sample_par")
draws <- vapply(seq_along(mu), function(i) {
  x <- rnbinom(20000, size = size[i], mu = mu[i])
  c(mean(x), var(x))
}, numeric(2))
wm(t(draws), dir, "sample_mom")

# ---- aborts ----------------------------------------------------------

dir <- file.path(root, "abort")
dir.create(dir, showWarnings = FALSE)
tri <- quiet(as_rt(triangles$toy$tri))
pre <- quiet(preprocess_negative_values(tri))
pmf <- quiet(estimate_delay(pre, n = 5))
aborts <- list(
  estimate_n_too_large = function() estimate_delay(pre, n = 99),
  estimate_n_zero = function() estimate_delay(pre, n = 0),
  estimate_no_complete_row = function() estimate_delay(pre, n = 2),
  apply_wrong_length = function() apply_delay(pre, pmf[-1]),
  apply_complete = function() {
    apply_delay(as_rt(triangles$toy$tri[1:4, ]), pmf)
  },
  apply_negative_first = function() apply_delay(pre, c(-0.1, 0.5, 0.4, 0.2)),
  alloc_scale_negative = function() {
    allocate_reference_times(pre, scale_factor = -1)
  },
  alloc_prop_large = function() {
    allocate_reference_times(pre, prop_delay = 1.5)
  },
  alloc_not_enough = function() {
    allocate_reference_times(pre, scale_factor = 0.5)
  },
  structure_zero = function() apply_reporting_structure(pre, structure = 0),
  structure_big = function() apply_reporting_structure(pre, structure = 9),
  trunc_negative = function() truncate_to_rows(pre, n = -1),
  fit_nb_negative = function() fit_nb(c(-1, 2), c(1, 1)),
  fit_nb_fraction = function() fit_nb(c(1.5, 2), c(1, 1)),
  retro_too_few = function() {
    estimate_uncertainty_retro(pre, n_history_delay = 4, n_retrospective_nowcasts = 1)
  },
  retro_too_many = function() {
    estimate_uncertainty_retro(pre, n_history_delay = 6, n_retrospective_nowcasts = 6)
  },
  sample_params_short = function() {
    sample_prediction(
      apply_delay(pre, pmf), pre, c(1, 2)
    )
  }
)
status <- vapply(names(aborts), function(nm) {
  inherits(
    tryCatch(quiet(aborts[[nm]]()), error = function(e) e),
    "error"
  )
}, logical(1))
if (!all(status)) {
  stop("expected R aborts did not occur: ", toString(names(status)[!status]))
}
writeLines(names(aborts), file.path(dir, "names.txt"))

# ---- release extension -----------------------------------------------

dir <- file.path(root, "rel")
dir.create(dir, showWarnings = FALSE)
set.seed(11)
D <- 6
n_ref <- 14
cutoffs <- c(8, 11, 13)
rel_pmf <- c(0.3, 0.25, 0.15, 0.1, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01)
level <- 60 + 5 * seq_len(n_ref)
full <- t(sapply(seq_len(n_ref), function(i) {
  rpois(length(rel_pmf), level[i] * rel_pmf)
}))
# Cumulative count in release k for onset t (0-based row index t - 1).
cum <- matrix(NA_real_, n_ref, length(cutoffs))
for (k in seq_along(cutoffs)) {
  for (t in 0:(n_ref - 1)) {
    if (t <= cutoffs[k]) {
      upto <- min(cutoffs[k] - t, length(rel_pmf) - 1)
      cum[t + 1, k] <- sum(full[t + 1, 1:(upto + 1)])
    }
  }
}
# A negative revision in release 2 for onset day 9.
cum[10, 2] <- cum[10, 1] - 3
cum[10, 3] <- cum[10, 2] + 2
wm(cum, dir, "cum")
writeLines(as.character(c(D, n_ref, cutoffs)), file.path(dir, "settings.txt"))

inc <- matrix(NA_real_, n_ref, length(cutoffs))
lo <- hi <- matrix(NA_real_, n_ref, length(cutoffs))
for (k in seq_along(cutoffs)) {
  prev <- if (k == 1) -Inf else cutoffs[k - 1]
  for (t in 0:(n_ref - 1)) {
    if (t <= cutoffs[k]) {
      before <- if (k == 1) 0 else if (is.na(cum[t + 1, k - 1])) 0 else
        cum[t + 1, k - 1]
      lo_t <- max(prev - t + 1, 0)
      hi_t <- min(cutoffs[k] - t, D)
      if (lo_t <= hi_t) {
        inc[t + 1, k] <- cum[t + 1, k] - before
        lo[t + 1, k] <- lo_t
        hi[t + 1, k] <- hi_t
      }
    }
  }
}
wm(inc, dir, "increments")
wm(lo, dir, "lo")
wm(hi, dir, "hi")

# Releases play the part of report columns. The NA pattern is not
# bottom right, so the triangle is built without validation.
inc_rt <- baselinenowcast:::new_reporting_triangle(
  inc,
  as.Date("2026-01-01") + seq_len(n_ref) - 1,
  "days"
)
inc_pre <- unclass(quiet(preprocess_negative_values(
  inc_rt,
  validate = FALSE
)))
dimnames(inc_pre) <- NULL
wm(inc_pre, dir, "increments_pre")

expand <- function(p) {
  x <- matrix(0, n_ref, D + 1)
  for (t in 0:(n_ref - 1)) {
    for (k in seq_along(cutoffs)) {
      if (is.na(inc_pre[t + 1, k])) next
      js <- lo[t + 1, k]:hi[t + 1, k]
      w <- p[js + 1]
      w <- if (sum(w) > 0) w / sum(w) else rep(1 / length(js), length(js))
      x[t + 1, js + 1] <- x[t + 1, js + 1] + inc_pre[t + 1, k] * w
    }
  }
  for (t in 0:(n_ref - 1)) {
    for (d in 0:D) if (t + d > cutoffs[length(cutoffs)]) x[t + 1, d + 1] <- NA
  }
  x
}
n_used <- quiet(allocate_reference_times(
  as_rt(expand(rep(1 / (D + 1), D + 1))),
  scale_factor = 3, prop_delay = 0.5
))$n_history_delay
writeLines(as.character(n_used), file.path(dir, "n.txt"))
p <- rep(1 / (D + 1), D + 1)
wm(expand(p), dir, "expanded_uniform")
for (iter in 1:50) {
  p <- quiet(estimate_delay(as_rt(expand(p)), n = n_used))
  if (iter == 1) wm(p, dir, "pmf_iter1")
}
wm(p, dir, "pmf_iter50")
wm(expand(p), dir, "expanded_iter50")

message("fixtures written to ", root)
