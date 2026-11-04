# Reference daily pmfs from primarycensored for tests/model/test_components.py.
# Regenerate with: Rscript tests/model/fixtures/dpcens.R
# Uniform primary over one day, daily secondary censoring.
# Generated with primarycensored 1.6.0.
library(primarycensored)
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
# Onset-to-report delays on 0..60.
for (family in c("gamma", "lognormal")) {
  for (mean in c(2, 5, 8, 15)) {
    for (cv in c(0.3, 0.8, 1.5)) {
      run("delay", family, mean, cv, 0:60, L = -Inf, D = 61)
    }
  }
}
out <- do.call(rbind, out)
out$p <- sprintf("%.17g", out$p)
write.csv(out, "tests/model/fixtures/dpcens.csv", row.names = FALSE,
          quote = FALSE)
cat("primarycensored", as.character(packageVersion("primarycensored")), "\n")
