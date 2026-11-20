#' Read a simulation grid
#'
#' @param path A grid config file (YAML), as `config/finetune/grid.yaml`.
#' @return The grid as a list: `factors`, `distributions`, `shares`,
#'   `rates`, `counts`, `floors`, `constants` and the rest of the file.
#' @export
read_grid <- function(path) {
  yaml::read_yaml(path)
}

#' Draw a simulation's categorical settings
#'
#' Each factor's level from its categorical distribution (the levels'
#' weights in `grid$factors`), independently of the others.
#'
#' @param grid A grid (`read_grid`).
#' @return A named list, one level per factor.
#' @export
draw_cell <- function(grid) {
  lapply(grid$factors, function(f) {
    w <- unlist(f)
    names(w)[sample.int(length(w), 1, prob = w)]
  })
}

#' Draw a setting from its sampling distribution
#'
#' `lognormal` takes its median and sdlog, `normal` its mean and sd,
#' `halfnormal` its scale, `gamma` its mean and sd, `beta` its shapes on
#' its support `lower` to `upper`, and `uniform` its range (only the cyclic
#' phase). None is truncated.
#'
#' @param grid A grid (`read_grid`).
#' @param name The setting's name in `grid$distributions`.
#' @param n Draws.
#' @return `n` draws.
#' @export
draw_setting <- function(grid, name, n = 1L) {
  d <- grid$distributions[[name]]
  if (is.null(d)) stop("no sampling distribution for ", name)
  switch(d$dist,
    lognormal = rlnorm(n, log(d$median), d$sdlog),
    normal = rnorm(n, d$mean, d$sd),
    halfnormal = abs(rnorm(n, 0, d$scale)),
    gamma = rgamma(n, (d$mean / d$sd)^2, d$mean / d$sd^2),
    beta = d$lower + (d$upper - d$lower) * rbeta(n, d$a, d$b),
    uniform = runif(n, d$lower, d$upper),
    stop("unknown distribution ", d$dist)
  )
}

#' A count at least `base` by definition: base plus a Poisson
#' @param grid A grid (`read_grid`).
#' @param name The count's name in `grid$counts`.
#' @param n Draws.
#' @return `n` integer counts.
#' @export
draw_count <- function(grid, name, n = 1L) {
  k <- grid$counts[[name]]
  if (is.null(k)) stop("no count for ", name)
  as.integer(k$base) + rpois(n, k$mean)
}

#' A Bernoulli choice with the grid's share
#' @param grid A grid (`read_grid`).
#' @param name The share's name in `grid$shares`.
#' @param n Draws.
#' @return `n` logicals.
#' @export
draw_share <- function(grid, name, n = 1L) {
  p <- grid$shares[[name]]
  if (is.null(p)) stop("no share for ", name)
  rbinom(n, 1, p) == 1
}
