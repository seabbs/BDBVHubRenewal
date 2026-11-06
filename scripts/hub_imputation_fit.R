# Refit the hub's onset imputation delays (HUB_GENGAMMA in
# src/bdbvhubrenewal/data/delays.py).
#
# The hub fits a generalised gamma in hours to the delays published at
# 2026-10-17 (auxiliary-data/imputation-details.md). Releases before
# 2026-11-01 used flexsurv with status 1, so each delay is exact; from
# 2026-11-01 it is double interval censored with primarycensored.
# Run from the repo root:
#
#   Rscript scripts/hub_imputation_fit.R

suppressPackageStartupMessages({
  library(flexsurv)
  library(primarycensored)
})

url <- paste0(
  "https://raw.githubusercontent.com/InsightNet-US/BDBV-Modeling-Hub/",
  "d67f1b74e0ecd1b32bb3d57554d871502acb2eca/auxiliary-data/",
  "archived-delay-data/weekly-reporting-delays-by-province-2026-10-17.csv"
)
d <- read.csv(url)$days_sym_to_report
d <- d[d >= 0]

exact <- flexsurvreg(
  Surv(ifelse(d == 0, 1, 24 * d), 24 * (d + 1) - 1, rep(1, length(d)),
       type = "interval") ~ 1,
  dist = "gengamma"
)
cens <- data.frame(left = 24 * d, right = 24 * (d + 1), pwindow = 24, D = Inf)
double_censored <- fitdistdoublecens(
  cens, distr = "gengamma", start = list(mu = 5, sigma = 1, Q = 0.5)
)

print(rbind(
  exact = c(
    exact$coefficients["mu"], exp(exact$coefficients["sigma"]),
    exact$coefficients["Q"]
  ),
  double_censored = double_censored$estimate[c("mu", "sigma", "Q")]
), digits = 15)
