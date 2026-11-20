# Generate test fixtures from the R simulator's infection side.
#
# The JAX port (src/bdbvhubrenewal/model/simulate_latent.py) is checked
# against these fixtures. Run from the repo root:
#
#   Rscript scripts/latent_fixtures.R
#
# The functions come from packages/nowcastfoundation/r/R/infections.R and
# observation.R, loaded with pkgload. Outputs go to
# tests/model/fixtures/latent/<case>.json with the inputs and outputs of
# each case. For `latent_onsets`, every normal draw (as a standard normal)
# and every path kind drawn are recorded in the order R makes them, so the
# port can replay them exactly.

suppressPackageStartupMessages({
  pkgload::load_all("packages/nowcastfoundation/r", quiet = TRUE)
  library(jsonlite)
  library(data.table)
})

ns <- asNamespace("nowcastfoundation")
grid <- read_grid(file.path("config", "finetune", "grids", "r7.yaml"))
root <- file.path("tests", "model", "fixtures", "latent")
dir.create(root, recursive = TRUE, showWarnings = FALSE)

out <- function(name, x) {
  write_json(
    x, file.path(root, paste0(name, ".json")),
    auto_unbox = TRUE, digits = NA, matrix = "rowmajor", null = "null",
    na = "null"
  )
}

strata_of <- function(places, ages = 1L) {
  s <- CJ(
    place = sprintf("P%02d", seq_len(places)), age = sprintf("A%d", seq_len(ages))
  )
  s[, .group := .I][]
}

# Pmfs and the delay convolution.
set.seed(1)
x <- matrix(rpois(60, 50), 20, 3)
out("pmfs", list(
  gt = lapply(
    list(list(5, "gamma", 0.5), list(12, "lognormal", 0.8), list(2, "gamma", 1.2)),
    function(a) list(mean = a[[1]], family = a[[2]], cv = a[[3]], pmf = gt_pmf(a[[1]], a[[2]], a[[3]]))
  ),
  incubation = lapply(c(5, 1.2, 20), function(m) {
    list(mean = m, cv = 0.5, pmf = incubation_pmf(m, 0.5))
  }),
  convolve = list(
    x = x,
    pmfs = list(incubation_pmf(2), incubation_pmf(5), incubation_pmf(9)),
    out = convolve_rows(x, list(incubation_pmf(2), incubation_pmf(5), incubation_pmf(9)))
  )
))

# Path shapes on log R.
set.seed(2)
v <- list(
  amplitude = 0.3, peak = 0.4, flip_days = 10, period = 40, phase = 1,
  waves = 3L, wave_scale = c(1, 0.8, 0.6, 0.5), walk_r = rnorm(30, 0, 0.1),
  control_days = 20, flat = 0.05
)
kinds <- c(
  "endemic", "controlled", "growth", "decline", "flat", "rise_fall_gradual",
  "fall_rise", "rise_fall_sharp", "oscillating", "multi_wave"
)
out("paths", list(
  n = 150L, values = v,
  paths = setNames(lapply(kinds, function(k) path_log_r(150L, k, v)), kinds),
  daily_walk = list(days = 0:40, walk = v$walk_r[1:5], out = daily_walk(0:40, v$walk_r[1:5]))
))

# Mixing between places.
strata <- strata_of(4L, 2L)
v <- list(
  mixing_order = c(3L, 1L, 4L, 2L), mixing_x = c(0.1, 0.5, 0.9, 0.3),
  mixing_y = c(0.2, 0.8, 0.4, 0.6), mixing_range = 0.4, mixing_eps = 0.2,
  quiet_places = 3L, mixing_switch_at = 0.3, mixing_switch_days = 9L
)
cases <- list()
for (kind in c("none", "symmetric", "asymmetric", "distance", "time_varying")) {
  for (switch in if (kind == "time_varying") c("on", "off") else "on") {
    v$mixing_switch <- switch
    m <- place_mixing(60L, strata, list(mixing = kind), v)
    cases[[length(cases) + 1]] <- list(
      mixing = kind, switch = switch, K = m$K, eps = m$eps
    )
  }
}
out("mixing", list(n = 60L, strata = strata, values = v, cases = cases))

# Imports.
strata <- strata_of(3L, 2L)
share <- c(2, 1, 0, 3, 1.5, 0.5)
v <- list(
  late_places = 2L, late_at = 0.4, late_days = 5L, import_rate = 0.01,
  import_at = c(0.2, 0.5, 0.5001), import_size = 4
)
cases <- list()
for (imports in c("none", "constant", "pulses")) {
  for (seeding in c("start", "mid")) {
    d <- list(imports = imports, seeding = seeding)
    cases[[length(cases) + 1]] <- list(
      imports = imports, seeding = seeding,
      out = stratum_imports(50L, share, strata, d, v)
    )
  }
}
out("imports", list(n = 50L, strata = strata, share = share, values = v, cases = cases))

# The renewal.
gt <- gt_pmf(5, "gamma", 0.5)
n <- 120L
k <- 3L
t <- seq_len(n)
log_r <- cbind(0.4 * cos(t / 20), 0.2 - t / 400, 0.6 * sin(t / 15))
population <- c(1e4, 5e3, 2e4)
start <- matrix(0, length(gt), k)
start[length(gt), ] <- c(5, 2, 0)
start[, 3] <- c(rep(0, length(gt) - 1), 1)
K <- matrix(c(0, 1, 0, 0.5, 0, 0.5, 0, 1, 0), 3, byrow = TRUE)
eps <- pmin(t / 60, 1) * 0.3
imports <- matrix(0, n, k)
imports[c(30, 31, 80), 2] <- c(3, 2, 5)
imports[, 3] <- 0.05
shifts <- list(
  at = c(40, 90), size = c(0.5, 0.3), days = c(14, 20),
  reactive = c(TRUE, FALSE), up = c(FALSE, TRUE)
)
variants <- list(at = 70, escape = 0.4, mult = 1.5, days = 12)
runs <- list(
  plain = renewal_infections(start, gt, log_r, population),
  full = renewal_infections(
    start, gt, log_r, population, mixing = list(K = K, eps = eps),
    imports = imports, shifts = shifts, variants = variants
  ),
  exhaust = renewal_infections(start * 50, gt, log_r + 1.5, population)
)
out("renewal", list(
  gt = gt, log_r = log_r, population = population, start = start,
  K = K, eps = eps, imports = imports, shifts = shifts, variants = variants,
  runs = runs
))

# Whole latent runs, with R's draws recorded.
tape <- new.env()
recorder <- list(
  rnorm = function(n, mean = 0, sd = 1) {
    x <- stats::rnorm(n, mean, sd)
    z <- if (all(sd == 0)) rep(0, n) else (x - mean) / sd
    tape$normal <- c(tape$normal, z)
    x
  },
  sample = function(x, size, ...) {
    y <- base::sample(x, size, ...)
    tape$choice <- c(tape$choice, y)
    y
  }
)
# The namespace is locked, so the functions that draw run as copies in an
# environment holding the recorders.
rec <- new.env(parent = ns)
for (f in names(recorder)) assign(f, recorder[[f]], envir = rec)
for (f in c("latent_onsets", "strata_log_r", "weekly_noise")) {
  g <- get(f, envir = ns)
  environment(g) <- rec
  assign(f, g, envir = rec)
}
record <- function(...) {
  tape$normal <- numeric(0)
  tape$choice <- character(0)
  rec$latent_onsets(...)
}

layer12 <- c(
  "weeks", "offset", "quiet", "coupling", "place_rho", "place_sd",
  "place_lags", "late_at", "late_days", "mixing_eps", "mixing_order",
  "mixing_x", "mixing_y", "mixing_range", "mixing_switch", "mixing_switch_at",
  "mixing_switch_days", "import_rate", "import_at", "import_size",
  "stratum_sd", "path", "gt_mean", "gt_cv", "amplitude", "flat", "peak",
  "period", "phase", "waves", "wave_scale", "flip_days", "walk_r",
  "control_days", "r_drift", "population", "seed", "level", "trickle",
  "shift_at", "shift_size", "shift_days", "shift_reactive", "shift_up",
  "variant_at", "variant_escape", "variant_r", "variant_days",
  "quiet_places", "late_places", "ascertainment", "incubation_mean"
)
design_keys <- c(
  "window", "mixing", "imports", "seeding", "place_rt", "spread",
  "gt_family", "n_places", "n_ages", "trajectory", "strata_link"
)
want <- list(
  mixing = c("none", "symmetric", "asymmetric", "distance", "time_varying"),
  place_rt = c("shared", "independent", "correlated", "lagged"),
  window = c("early", "established", "endemic"),
  imports = c("none", "constant", "pulses"),
  seeding = c("start", "mid")
)
seen <- lapply(want, function(x) character(0))
cases <- list()
for (s in 1:400) {
  set.seed(1000 + s)
  d <- draw_design(grid, weeks = 8L)
  if (d$n_places * d$n_ages > 8) next
  new <- any(vapply(names(want), function(f) !(d[[f]] %in% seen[[f]]), logical(1)))
  if (!new && length(cases) >= 12) next
  v <- draw_values(grid, d)
  gt <- gt_pmf(v$gt_mean, d$gt_family, v$gt_cv)
  n_days <- 7L * (8L + 4L + 4L)
  strata <- strata_of(d$n_places, d$n_ages)
  run <- tryCatch(
    record(grid, n_days - length(gt), gt, strata, d, v),
    error = function(e) list(error = conditionMessage(e))
  )
  case <- list(
    seed = 1000 + s, n_days = n_days, design = d[design_keys],
    values = v[layer12], strata = strata,
    tape = list(normal = tape$normal, choice = tape$choice)
  )
  if (!is.null(run$error)) {
    case$error <- run$error
  } else {
    case$n_gt <- length(gt)
    case$onsets <- run$onsets
    case$infections <- run$infections
    case$latent <- run$latent
    case$stratum_population <- run$values$stratum_population
    case$attack <- run$values$attack
  }
  cases[[length(cases) + 1]] <- case
  for (f in names(want)) seen[[f]] <- union(seen[[f]], d[[f]])
  if (length(cases) >= 20) break
}
# Multi-place designs with each window, mixing kind and place R relation,
# set on a drawn design as the R tests do.
forced <- list(
  list(window = "early", mixing = "distance", place_rt = "independent", imports = "pulses", seeding = "mid"),
  list(window = "early", mixing = "time_varying", place_rt = "correlated", imports = "constant", seeding = "start"),
  list(window = "established", mixing = "time_varying", place_rt = "lagged", imports = "none", seeding = "mid"),
  list(window = "established", mixing = "distance", place_rt = "correlated", imports = "pulses", seeding = "start"),
  list(window = "endemic", mixing = "symmetric", place_rt = "independent", imports = "constant", seeding = "mid"),
  list(window = "endemic", mixing = "time_varying", place_rt = "shared", imports = "pulses", seeding = "start")
)
for (i in seq_along(forced)) {
  set.seed(5000 + i)
  d <- draw_design(grid, weeks = 12L)
  d[names(forced[[i]])] <- forced[[i]]
  d$strata <- "places_ages"
  d$strata_link <- "linked"
  d$n_places <- 3L
  d$n_ages <- 2L
  v <- draw_values(grid, d)
  # Shifts and a variant, so the renewal's events are replayed too.
  v$shift_at <- c(0.3, 0.6)
  v$shift_size <- c(0.4, 0.3)
  v$shift_days <- c(10, 20)
  v$shift_reactive <- c(TRUE, FALSE)
  v$shift_up <- c(TRUE, FALSE)
  if (d$window != "early") {
    v$variant_at <- 0.5
    v$variant_escape <- 0.3
    v$variant_r <- 1.4
    v$variant_days <- 14
  }
  gt <- gt_pmf(v$gt_mean, d$gt_family, v$gt_cv)
  n_days <- 7L * (12L + 4L + 4L)
  strata <- strata_of(d$n_places, d$n_ages)
  run <- tryCatch(
    record(grid, n_days - length(gt), gt, strata, d, v),
    error = function(e) list(error = conditionMessage(e))
  )
  case <- list(
    seed = 5000 + i, n_days = n_days, design = d[design_keys],
    values = v[layer12], strata = strata,
    tape = list(normal = tape$normal, choice = tape$choice)
  )
  if (!is.null(run$error)) {
    case$error <- run$error
  } else {
    case$n_gt <- length(gt)
    case$onsets <- run$onsets
    case$infections <- run$infections
    case$latent <- run$latent
    case$stratum_population <- run$values$stratum_population
    case$attack <- run$values$attack
  }
  cases[[length(cases) + 1]] <- case
}
for (i in seq_along(cases)) out(sprintf("latent-%02d", i), cases[[i]])
message("wrote ", length(cases), " latent cases; designs seen:")
print(seen)
