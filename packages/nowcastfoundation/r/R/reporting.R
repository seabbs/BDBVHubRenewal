# The first and last reference dates of reporting shock `k` in `counts`.
shock_window <- function(counts, v, k = 1L) {
  start <- min(counts$reference_date) + round(v$shock_at[k] * v$weeks * 7)
  c(start, start + 7L * v$shock_weeks[k] - 1L)
}

# The first and last reference dates of a slowdown in `counts`.
slowdown_window <- function(counts, v) {
  start <- min(counts$reference_date) + round(v$slow_at * v$weeks * 7)
  c(start, start + 7L * v$slow_weeks - 1L)
}

# Revisions (layer 4) on counts by report date (`mean`, as
# `simulate_outbreak` builds them), with each stratum's rates: cases reported in error (a Poisson
# share `revise_over` of each count) stay in the releases for
# `revise_keep` days and are then removed; cases filed under a nearby
# reference day (a binomial share `revise_move`, 1 + Poisson(1) days
# either way, never after the report date) are moved back after the same
# time. A correction past the last report date falls on it, so the final
# counts are unchanged; cumulative counts by report date can fall but not
# below zero.
revise_counts <- function(pp, v) {
  last <- max(pp$report_date)
  first <- min(pp$reference_date)
  group <- if (".group" %in% names(pp)) pp[mean > 0, .group] else NULL
  base <- pp[mean > 0, .(draw, location, reference_date, report_date, mean)]
  n <- nrow(base)
  # Each stratum's rates (`.group`), or the one rate if no strata given.
  rate <- function(x) if (is.null(group)) rep(x[1], n) else x[group]
  keep <- pmax(1L, as.integer(round(rate(v$revise_keep))))
  later <- pmin(base$report_date + keep, last)
  err <- rpois(n, rate(v$revise_over) * base$mean)
  moved <- rbinom(n, base$mean, rate(v$revise_move))
  shift <- (2L * rbinom(n, 1, 0.5) - 1L) * (1L + rpois(n, 1))
  to <- pmin(pmax(base$reference_date + shift, first), base$report_date)
  at <- function(ref, rep, x) {
    data.table(
      draw = base$draw, location = base$location, reference_date = ref,
      report_date = rep, mean = x
    )
  }
  ref <- base$reference_date
  rep <- base$report_date
  out <- rbind(
    base,
    at(ref, rep, err), at(ref, later, -err),
    at(ref, rep, -moved), at(to, rep, moved),
    at(ref, later, moved), at(to, later, -moved)
  )
  out <- out[, .(mean = sum(mean)),
    by = .(draw, location, reference_date, report_date)
  ]
  out[mean != 0]
}
