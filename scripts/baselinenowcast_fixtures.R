# Generate test fixtures from the R package baselinenowcast.
#
# The Python port (src/bdbvhubrenewal/data/triangle.py and
# src/bdbvhubrenewal/forecasters/persistence.py) is checked against
# these fixtures. Run from the repo root:
#
#   Rscript scripts/baselinenowcast_fixtures.R
#
# Inputs (tests/data/baselinenowcast/):
#   triangles.csv  triangle, row, delay, count (NA = unobserved)
#   settings.csv   triangle, n_history_delay, n_retrospective_nowcasts
# Outputs (same directory):
#   pmf.csv        triangle, delay, pmf
#   point.csv      triangle, row, point_final
#   nb_size.csv    triangle, horizon, nb_size (1 = most recent open row)
#   VERSION        package version and the CRAN snapshot it came from
#
# baselinenowcast is installed from CRAN, pinned through a Posit Package
# Manager snapshot (the date below and in the workflow).

pinned_version <- "0.2.0"
snapshot <- "2026-10-27"

suppressPackageStartupMessages(library(baselinenowcast))

if (as.character(packageVersion("baselinenowcast")) != pinned_version) {
  stop(
    "baselinenowcast ", pinned_version, " is required, found ",
    packageVersion("baselinenowcast")
  )
}

dir <- "tests/data/baselinenowcast"
triangles <- read.csv(file.path(dir, "triangles.csv"))
settings <- read.csv(file.path(dir, "settings.csv"))

fmt <- function(x) sprintf("%.12g", x)

write_csv <- function(df, name) {
  # Quote nothing: all fields are plain identifiers or numbers.
  write.table(
    df, file.path(dir, name),
    sep = ",", row.names = FALSE, quote = FALSE, na = "NA"
  )
}

pmf_out <- list()
point_out <- list()
size_out <- list()

for (i in seq_len(nrow(settings))) {
  name <- settings$triangle[i]
  n_hist <- settings$n_history_delay[i]
  n_retro <- settings$n_retrospective_nowcasts[i]
  tri <- triangles[triangles$triangle == name, ]

  rows <- sort(unique(tri$row))
  delays <- sort(unique(tri$delay))
  m <- matrix(NA_real_, length(rows), length(delays))
  m[cbind(match(tri$row, rows), match(tri$delay, delays))] <- tri$count
  colnames(m) <- delays

  ref_dates <- as.Date("2026-01-03") + 7 * (seq_along(rows) - 1)
  rt <- as_reporting_triangle(
    m,
    delays_unit = "days",
    reference_dates = ref_dates
  )

  pmf <- estimate_delay(rt, n = n_hist)
  point <- apply_delay(rt, pmf)
  sizes <- estimate_uncertainty_retro(
    rt,
    n_history_delay = n_hist,
    n_retrospective_nowcasts = n_retro
  )

  pmf_out[[name]] <- data.frame(
    triangle = name, delay = delays, pmf = fmt(as.numeric(pmf))
  )
  pt <- as.matrix(point)
  point_out[[name]] <- data.frame(
    triangle = name, row = rows, point_final = fmt(rowSums(pt))
  )
  size_out[[name]] <- data.frame(
    triangle = name,
    horizon = seq_along(sizes),
    nb_size = fmt(as.numeric(sizes))
  )
}

write_csv(do.call(rbind, pmf_out), "pmf.csv")
write_csv(do.call(rbind, point_out), "point.csv")
write_csv(do.call(rbind, size_out), "nb_size.csv")

writeLines(
  c(
    paste0("baselinenowcast ", pinned_version),
    paste0("CRAN snapshot https://packagemanager.posit.co/cran/", snapshot)
  ),
  file.path(dir, "VERSION")
)
