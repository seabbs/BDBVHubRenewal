# Reference daily pmfs from primarycensored for tests/model/test_components.py.
# Regenerate with: Rscript tests/model/fixtures/dpcens.R
# Uniform primary over one day, daily secondary censoring.
# Generated with primarycensored 1.6.0.
library(primarycensored)
library(flexsurv)
out <- list()
run <- function(case, family, mean, cv, x, L, D) {
  sd <- mean * cv
  p <- if (family == "gamma") {
    dpcens(x, pgamma, pwindow = 1, swindow = 1, L = L, D = D,
           shape = 1 / cv^2, scale = sd^2 / mean)
  } else {
    sdlog <- sqrt(log1p(cv^2))
    dpcens(x, plnorm, pwindow = 1, swindow = 1, L = L, D = D,
           meanlog = log(mean) - sdlog^2 / 2, sdlog = sdlog)
  }
  out[[length(out) + 1]] <<- data.frame(case, family, mean, cv, delay = x, p)
}
# Generation interval on lags 1..42: left truncated at a day.
run("gi", "gamma", 15.3, 9.3 / 15.3, 1:42, L = 1, D = 43)
# Incubation period on 0..28 (MacNeil et al. 2010, as BVDOutbreakSize).
run("incubation", "lognormal", 6.3, 3.5 / 6.3, 0:28, L = -Inf, D = 29)
# Onset-to-report delays on 0..60.
for (family in c("gamma", "lognormal")) {
  for (mean in c(2, 5, 8, 15)) {
    for (cv in c(0.3, 0.8, 1.5)) {
      run("delay", family, mean, cv, 0:60, L = -Inf, D = 61)
    }
  }
}
# The hub's imputation delays: generalised gamma fits in hours
# (HUB_GENGAMMA in src/bdbvhubrenewal/data/delays.py), on days 0..60.
hub <- list(
  exact = c(5.12406648030701, 0.97184390713028, 1.07646963898928),
  double_censored = c(4.99016267662049, 0.936895070027868, 0.715805359543864)
)
for (fit in names(hub)) {
  par <- hub[[fit]]
  p <- dpcens(0:60, function(q, ...) pgengamma(24 * q, par[1], par[2], par[3]),
              pwindow = 1, swindow = 1, L = -Inf, D = 61)
  out[[length(out) + 1]] <- data.frame(
    case = "hub", family = fit, mean = NA, cv = NA, delay = 0:60, p
  )
}
out <- do.call(rbind, out)
out$p <- sprintf("%.17g", out$p)
write.csv(out, "tests/model/fixtures/dpcens.csv", row.names = FALSE,
          quote = FALSE)
cat("primarycensored", as.character(packageVersion("primarycensored")), "\n")
