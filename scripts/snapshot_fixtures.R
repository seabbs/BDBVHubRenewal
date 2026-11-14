# Generate fixtures for the snapshot extension (E1) of the JAX port.
#
# E1 takes the chain ladder ratio of sums over the cells observed in
# both releases of a pair and skips the others. Run from the repo root:
#
#   Rscript scripts/snapshot_fixtures.R
#
# Case recon: a right-truncated daily triangle. With a release on every
# day and every denominator above 0, E1 reduces to baselinenowcast
# 0.2.2. The fixtures hold the triangle, the pmf from estimate_delay and
# the row totals of the chain ladder fill.
#
# Case skip: release snapshots on irregular days with unobserved cells
# and a zero denominator. The pair sums come from a plain R loop over
# the cells, written independently of the Python code.
#
# Outputs go to tests/baselinenowcast/fixtures/<case>/ as CSV matrices
# without a header. NA marks an unobserved or unidentified value.

pinned_version <- "0.2.2"

suppressPackageStartupMessages(library(baselinenowcast))

if (as.character(packageVersion("baselinenowcast")) != pinned_version) {
  stop(
    "baselinenowcast ", pinned_version, " is required, found ",
    packageVersion("baselinenowcast")
  )
}

root <- file.path("tests", "baselinenowcast", "fixtures")
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
quiet <- function(expr) suppressWarnings(suppressMessages(expr))

# ---- recon -----------------------------------------------------------

dir <- file.path(root, "recon")
dir.create(dir, recursive = TRUE, showWarnings = FALSE)
set.seed(20261102)
n_ref <- 40
max_delay <- 10
n_use <- 25
pmf <- diff(pgamma(0:(max_delay + 1), shape = 2, rate = 0.5))
pmf <- pmf / sum(pmf)
level <- 40 + 2 * seq_len(n_ref)
full <- t(sapply(seq_len(n_ref), function(i) {
  rpois(max_delay + 1, level[i] * pmf)
}))
tri <- full
for (i in seq_len(n_ref)) {
  for (d in 0:max_delay) if (i - 1 + d > n_ref - 1) tri[i, d + 1] <- NA
}
wm(tri, dir, "tri")
writeLines(as.character(c(max_delay, n_ref, n_use)),
  file.path(dir, "settings.txt"))
rt <- as_reporting_triangle(
  tri,
  delays_unit = "days",
  reference_dates = as.Date("2026-01-01") + seq_len(n_ref) - 1
)
wm(quiet(estimate_delay(rt, n = n_use)), dir, "pmf")
last <- tri[(n_ref - n_use + 1):n_ref, , drop = FALSE]
filled <- baselinenowcast:::.chainladder_fill_triangle(last)
wm(rowSums(filled), dir, "row_totals")

# ---- skip ------------------------------------------------------------

dir <- file.path(root, "skip")
dir.create(dir, recursive = TRUE, showWarnings = FALSE)
set.seed(7)
D <- 8
n_loc <- 2
n_day <- 30
cut <- c(2, 11, 13, 19, 22, 24, 29) # 0-based latest onset day
n_rel <- length(cut)
delay_pmf <- diff(pgamma(0:(D + 4), shape = 2, rate = 0.6))
final <- sapply(1:n_loc, function(l) {
  rpois(n_day, (15 + 10 * l) * exp(0.02 * (1:n_day)))
})
# counts[[l]][r, t + 1]: day t in release r, NA where unobserved
counts <- lapply(1:n_loc, function(l) {
  m <- matrix(NA_real_, n_rel, n_day)
  for (r in 1:n_rel) {
    for (t in 0:cut[r]) {
      share <- sum(delay_pmf[1:min(cut[r] - t + 1, length(delay_pmf))])
      m[r, t + 1] <- rbinom(1, final[t + 1, l], min(share, 1))
    }
  }
  m
})
# Cells missing inside the data, and a zero denominator.
for (l in 1:n_loc) counts[[l]][3, 1:3] <- NA
counts[[1]][4, cut[4] + 1] <- 0
counts[[1]][5, cut[5] + 1 - 2] <- 0
observed <- !is.na(counts[[1]]) & !is.na(counts[[2]])
for (l in 1:n_loc) {
  counts[[l]][!observed] <- NA
  wm(counts[[l]], dir, paste0("counts_", l))
}
wm(observed * 1, dir, "observed")
writeLines(as.character(cut), file.path(dir, "cut.txt"))
writeLines(as.character(c(D, n_loc, n_day)), file.path(dir, "settings.txt"))

# Cell by cell sums over the pairs.
sums <- list()
for (l in 1:n_loc) {
  for (p in seq_len(n_rel - 1)) {
    num <- den <- rep(0, D)
    for (x in 0:(D - 1)) {
      t <- cut[p] - x
      if (t < 0) next
      a <- counts[[l]][p, t + 1]
      b <- counts[[l]][p + 1, t + 1]
      if (!is.na(a) && !is.na(b)) {
        den[x + 1] <- a
        num[x + 1] <- b
      }
    }
    sums[[paste(l, p)]] <- rbind(num = num, den = den)
  }
}
wm(do.call(rbind, lapply(1:n_loc, function(l) {
  do.call(rbind, lapply(seq_len(n_rel - 1), function(p) {
    sums[[paste(l, p)]]
  }))
})), dir, "pair_sums")

message("snapshot fixtures written to ", root)
