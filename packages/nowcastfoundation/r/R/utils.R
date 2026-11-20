# Weekly AR(1) noise (autocorrelation 0.8) on `n` days.
weekly_noise <- function(n, sd) {
  weeks <- ceiling(n / 7)
  e <- numeric(weeks)
  for (w in seq_len(weeks)[-1]) e[w] <- 0.8 * e[w - 1] + rnorm(1, 0, sd)
  rep(e, each = 7)[seq_len(n)]
}

# A simulation's settings as one row: values with one entry per place
# become text joined by ";".
flat_values <- function(x) {
  lapply(x, function(e) {
    if (length(e) == 1L) e else paste(e, collapse = ";")
  })
}

# A rejected try's row: its design and values, flat.
failure_row <- function(id, try, reason, design, v) {
  flat_values(c(list(outbreak = id, try = try, reason = reason), design, v))
}

# An expectation that gives the latent infections `lambda` (days by
# strata) exactly: epinowcast's growth model (`generation_time = 1`)
# starts each stratum at its first day and grows it by the growth rate,
# so the growth rate is the daily change in log `lambda`. Days without
# infections are floored at 1e-8.
latent_expectation <- function(pobs, lambda, observation) {
  expectation <- epinowcast::enw_expectation(
    r = ~1, generation_time = 1, observation = observation, data = pobs
  )
  stopifnot(nrow(lambda) == expectation$data$expr_t + 1L)
  log_l <- log(pmax(lambda, 1e-8))
  list(
    expectation = expectation,
    growth_rate = as.numeric(apply(log_l, 2, diff)),
    lelatent = matrix(log_l[1, ], nrow = 1)
  )
}

# A formula from its terms, `~ 0` if there are none.
as_formula <- function(intercept, terms) {
  terms <- c(if (intercept) "1" else "0", terms)
  as.formula(paste("~", paste(terms, collapse = " + ")))
}

# Standard deviations of a module's random effects, named by effect. An
# interaction's effects (`day_of_week__placeP01`, one per level of the
# second factor) take the first factor's.
effect_sds <- function(rdesign, sds) {
  effects <- sub("__.*", "", colnames(rdesign)[-1])
  out <- sds[effects]
  if (any(vapply(out, is.null, TRUE))) {
    stop("no sd for effect ", paste(effects[vapply(out, is.null, TRUE)], collapse = ", "))
  }
  array(as.numeric(unlist(out)))
}

# Stan data as JSON, as cmdstanr writes it from a list (`process_data`):
# each array of the model's data in its declared shape and integer type,
# reals with a decimal point. Arrays of `min_size` values or more are
# written in slabs along their first dimension (`write_json_array`), so
# the whole of a large array's text is never held at once.
write_stan_data <- function(data, file, variables, min_size = 1e6,
                            slab = 2e6) {
  for (name in names(variables)) {
    x <- data[[name]]
    if (length(x) == 1 && variables[[name]]$dimensions == 1) {
      x <- array(x, dim = 1)
    }
    if (variables[[name]]$type == "int" && !is.integer(x)) {
      x <- round(x)
      mode(x) <- "integer"
    }
    data[[name]] <- x
  }
  large <- vapply(data, length, 1) >= min_size
  cmdstanr::write_stan_json(data[!large], file, always_decimal = TRUE)
  # The small arrays' object, reopened for the large ones.
  head <- readLines(file)
  head[length(head)] <- sub("}$", "", head[length(head)])
  writeLines(head, file)
  sep <- if (any(!large)) ",\n" else ""
  for (name in names(data)[large]) {
    cat(sep, "\"", name, "\": ", file = file, sep = "", append = TRUE)
    write_json_array(data[[name]], file, slab)
    data[[name]] <- NULL
    sep <- ",\n"
  }
  cat("\n}\n", file = file, append = TRUE)
  invisible(file)
}

# Append numeric array `x` to `file` as a JSON array nested in index order
# (first index outermost), as jsonlite writes it. Each distinct value is
# formatted once by jsonlite (`digits = NA`, reals with a decimal point),
# so every value has jsonlite's text. Written in slabs of about `slab`
# values along the first dimension, one innermost array a line.
write_json_array <- function(x, file, slab = 2e6) {
  d <- dim(x)
  if (is.null(d)) d <- length(x)
  n <- length(d)
  inner <- d[n]
  # Each line is an innermost array, at index (i_1, ..., i_{n-1}). Before
  # it open the arrays it starts (i_k, ..., i_{n-1} all first, k >= 2);
  # after it close those it ends (all last).
  mid <- if (n > 2) d[2:(n - 1)] else integer(0)
  per_row <- prod(mid)
  cat("[", file = file, append = TRUE)
  step <- max(1L, floor(slab / prod(d[-1])))
  for (s in seq(1L, d[1], by = step)) {
    rows <- s:min(d[1], s + step - 1L)
    part <- if (n == 1) {
      x[rows]
    } else {
      do.call(`[`, c(list(x, rows), rep(list(TRUE), n - 1L), drop = FALSE))
    }
    # Values in row-major order, `inner` to a line. `unique` and `match`
    # take -0 for 0 (so 0 is labelled from +0); jsonlite writes -0 as
    # "-0.0".
    values <- if (n == 1) part else as.vector(aperm(part, n:1))
    u <- unique(values + 0L)
    txt <- jsonlite::toJSON(u, always_decimal = TRUE, digits = NA)
    labels <- strsplit(
      substr(txt, 2L, nchar(txt) - 1L), ",", fixed = TRUE
    )[[1]]
    stopifnot(length(labels) == length(u))
    text <- labels[match(values, u)]
    if (is.double(values)) {
      text[which(values == 0 & 1 / values < 0)] <- "-0.0"
    }
    rm(part, values)
    if (n == 1) {
      cat(
        if (s > 1L) ",", paste(text, collapse = ","),
        file = file, append = TRUE, sep = ""
      )
      next
    }
    lines <- matrix(text, nrow = inner)
    rm(text)
    # Collect the last slab's text now: after a large simulation R's next
    # collection is gigabytes away.
    invisible(gc(full = FALSE))
    k <- ncol(lines)
    # Position of each line among the middle indices (0-based, row-major).
    pos <- (seq_len(k) - 1L) %% per_row
    opens <- trailing_run(pos, rev(mid), 0L)
    closes <- trailing_run(pos, rev(mid), -1L)
    prefix <- paste0(
      c(if (s > 1L) "," else "", rep(",", k - 1L)), strrep("[", opens + 1L)
    )
    lines[1, ] <- paste0(prefix, lines[1, ])
    lines[inner, ] <- paste0(lines[inner, ], strrep("]", closes + 1L))
    fwrite(
      as.data.table(t(lines)), file,
      append = TRUE, quote = FALSE, sep = ",", col.names = FALSE
    )
  }
  cat("]", file = file, append = TRUE)
}

# For row-major positions `pos` (0-based) over dimensions given fastest
# first (`dims`), how many of the fastest indices in a row are at their
# first value (`at = 0`) or their last (`at = -1`).
trailing_run <- function(pos, dims, at) {
  run <- integer(length(pos))
  on <- rep(TRUE, length(pos))
  for (m in dims) {
    i <- pos %% m
    on <- on & i == if (at == 0L) 0L else m - 1L
    run <- run + on
    pos <- pos %/% m
  }
  run
}

# The Stan data and initial values of `enw_simulate` (epinowcast): its
# steps up to sampling, in its order, so the draws (and the R random
# stream) are its own. The reporting triangle's cells come back too. The
# data hold their own copies of the modules' design matrices.
enw_simulate_data <- function(data, growth_rate, parameters, reference,
                              report, expectation, obs) {
  expr_len <- expectation$data$expr_t * data$groups[[1]]
  growth_rate <- epinowcast:::.resolve_growth_rate_length(
    growth_rate, expr_len
  )
  capture_sampler <- function(data, init, ...) {
    data.table(data = list(data), init = list(init))
  }
  capture_fit <- epinowcast::enw_fit_opts(
    sampler = capture_sampler, nowcast = TRUE, pp = TRUE,
    likelihood = FALSE
  )
  # `epinowcast` uses `data` only to bind it to its output, which copies
  # every table in it; the triangle's cells are the one table needed.
  prepared <- do.call(epinowcast::epinowcast, list(
    data = data[, .(new_confirm)], reference = reference, report = report,
    expectation = expectation,
    missing = epinowcast::enw_missing(formula = ~0, data = data), obs = obs,
    fit = capture_fit, model = NULL
  ))
  data_list <- epinowcast::enw_get_data(prepared, "data")
  data_list$expr_r_override <- 1L
  data_list$expr_r_override_value <- as.numeric(growth_rate)
  init <- utils::modifyList(epinowcast::enw_get_data(prepared, "init")(), parameters)
  init <- epinowcast:::.map_centred_intercepts(init)
  list(data = data_list, init = init, cells = prepared$new_confirm[[1]])
}

# One draw of the posterior predictions `pp_obs` from the Stan data of
# `enw_simulate_data` written to `stan$file`, as `enw_simulate` samples
# it. The data and output files are removed once read.
enw_simulate_draw <- function(stan, model, seed, ...) {
  on.exit(unlink(stan$file))
  fit <- model$sample(
    data = stan$file, fixed_param = TRUE, chains = 1, iter_sampling = 1L,
    iter_warmup = 0, seed = seed, threads_per_chain = 1,
    init = function() stan$init, ...
  )
  on.exit(unlink(fit$output_files()), add = TRUE)
  if (any(fit$return_codes() != 0)) {
    stop("the simulation's Stan run failed: ", paste(fit$output_files(), collapse = ", "))
  }
  read_pp_obs(fit$output_files())
}

# The first draw of `pp_obs` in a CmdStan output file, as
# `$draws("pp_obs")` gives it, from the header and the first draw's line
# alone: only the `pp_obs` fields are parsed.
read_pp_obs <- function(file) {
  con <- file(file, "r")
  on.exit(close(con))
  lines <- character(0)
  while (length(lines) < 2L) {
    line <- readLines(con, n = 1L)
    stopifnot(length(line) == 1L)
    if (!startsWith(line, "#")) lines <- c(lines, line)
  }
  # Field i of a line runs between its commas i - 1 and i (found on the
  # line's bytes: `gregexpr` takes quadratic time on a line this long).
  commas <- function(x) {
    c(0L, which(charToRaw(x) == charToRaw(",")), nchar(x, "bytes") + 1L)
  }
  h <- commas(lines[1])
  at <- which(
    substring(lines[1], h[-length(h)] + 1L, h[-length(h)] + 7L) == "pp_obs."
  )
  field <- function(i) substr(lines[1], h[i] + 1L, h[i + 1L] - 1L)
  stopifnot(
    all(diff(at) == 1L), field(at[1]) == "pp_obs.1",
    field(at[length(at)]) == paste0("pp_obs.", length(at))
  )
  d <- commas(lines[2])
  stopifnot(length(d) == length(h))
  as.numeric(strsplit(
    substr(lines[2], d[at[1]] + 1L, d[at[length(at)] + 1L] - 1L), ",",
    fixed = TRUE
  )[[1]])
}

# An epinowcast module, its builder's frame emptied. A module's `inits`
# closure keeps the frame it was built in (the reference model's full
# design matrices among it) though it reads only its arguments.
drop_builder_frame <- function(module) {
  env <- environment(module$inits)
  rm(list = ls(env, all.names = TRUE), envir = env)
  module
}
