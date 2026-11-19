# The delay's epinowcast parameters (`refp_mean_int`, `refp_sd_int`), its
# 99th centile and its probabilities on 0, ..., max_delay - 1 days.
# Lognormal is meanlog and sdlog; gamma is shape exp(mean_int) and rate
# sd_int; the exponential rate is exp(-mean_int). With no parametric
# delay ("none"), the hazard's intercept is the constant hazard of a
# geometric delay with the drawn mean.
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

# The longest delay in days: the shortest of 3, 4, 6, 8 and 12 weeks at
# or above the delay's 99th centile, at most 12 weeks.
max_delay_for <- function(q99) {
  options <- c(21L, 28L, 42L, 56L, 84L)
  options[min(which(c(options >= q99, TRUE))[1], length(options))]
}
