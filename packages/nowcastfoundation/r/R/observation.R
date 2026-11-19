# The infection-to-reference-event delay (incubation, or infection to
# test or admission) on 0, 1, ... days: a discretised gamma with CV `cv`
# and mean `mean`, to its 99.9% quantile.
incubation_pmf <- function(mean, cv = 0.5) {
  shape <- 1 / cv^2
  q <- qgamma(0.999, shape, shape / mean)
  pmf <- diff(pgamma(0:max(1L, as.integer(ceiling(q))), shape, shape / mean))
  pmf / sum(pmf)
}

# Each column of `x` (days by strata) convolved with its own pmf in
# `pmfs` on delays 0, 1, ...
convolve_rows <- function(x, pmfs) {
  out <- matrix(0, nrow(x), ncol(x))
  for (j in seq_len(ncol(x))) {
    p <- pmfs[[j]]
    for (d in seq_along(p) - 1L) {
      if (d >= nrow(x)) break
      rows <- (d + 1L):nrow(x)
      out[rows, j] <- out[rows, j] + p[d + 1L] * x[rows - d, j]
    }
  }
  out
}

# Draws of the reports with mean `mu` in stratum `group`: NB2 with
# phi = 1 / sqrt_phi^2, NB1 with variance mu (1 + 1 / phi) on the NB2
# scale (epinowcast #937), or Poisson.
observe <- function(mu, group, family, sqrt_phi_obs) {
  f <- family[group]
  phi <- 1 / sqrt_phi_obs[group]^2
  out <- numeric(length(mu))
  on <- mu > 0
  pois <- on & f == "poisson"
  nb2 <- on & f == "negbin"
  nb1 <- on & f == "negbin1d"
  out[pois] <- rpois(sum(pois), mu[pois])
  out[nb2] <- rnbinom(sum(nb2), size = phi[nb2], mu = mu[nb2])
  out[nb1] <- rnbinom(sum(nb1), size = mu[nb1] * phi[nb1], mu = mu[nb1])
  out
}
