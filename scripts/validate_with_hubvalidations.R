#!/usr/bin/env Rscript
# Run the hub's official hubValidations checks on submission and metadata
# files, without placing them in a hub clone by hand.
#
# Each file is copied into a fresh temporary hub that holds a copy of the
# real hub's `hub-config/`, an empty `model-output/` and the relevant
# metadata file, so the clone is never modified.
#
# Usage:
#   Rscript validate_with_hubvalidations.R --hub PATH [options] FILE...
#
# Options:
#   --hub PATH           Local clone of BDBV-Modeling-Hub (required).
#   --metadata-dir DIR   Extra directory searched for `<model_id>.yml`
#                        before the hub's own `model-metadata/`
#                        (repeatable).
#   --no-metadata        Do not copy any metadata file into the temp hub.
#   --skip-window        Skip the submission window check
#                        (`skip_submit_window_check = TRUE`).
#   --json PATH          Write per-check results as JSON.
#
# FILE is either a model-output file (`.csv`/`.parquet`) whose parent
# directory is the model_id (e.g. `model-output/team-model/2026-10-31-...`)
# or a metadata file (`.yml`/`.yaml`).
#
# Exit status is 1 if any file has a failure or execution error.
#
# Install: pak::pak("hubverse-org/hubValidations") or
#   install.packages("hubValidations",
#     repos = c("https://hubverse-org.r-universe.dev", getOption("repos")))

suppressPackageStartupMessages({
  library(hubValidations)
  library(jsonlite)
})

parse_args <- function(args) {
  opts <- list(
    hub = NULL, metadata_dir = NULL, no_metadata = FALSE,
    skip_window = FALSE, json = NULL, files = character()
  )
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    if (a == "--hub") {
      opts$hub <- args[[i + 1L]]
      i <- i + 1L
    } else if (a == "--metadata-dir") {
      opts$metadata_dir <- c(opts$metadata_dir, args[[i + 1L]])
      i <- i + 1L
    } else if (a == "--json") {
      opts$json <- args[[i + 1L]]
      i <- i + 1L
    } else if (a == "--skip-window") {
      opts$skip_window <- TRUE
    } else if (a == "--no-metadata") {
      opts$no_metadata <- TRUE
    } else if (startsWith(a, "--")) {
      stop("Unknown option: ", a)
    } else {
      opts$files <- c(opts$files, a)
    }
    i <- i + 1L
  }
  if (is.null(opts$hub) || length(opts$files) == 0L) {
    stop("Usage: validate_with_hubvalidations.R --hub PATH [opts] FILE...")
  }
  opts
}

make_temp_hub <- function(hub) {
  tmp <- tempfile("hub-")
  dir.create(file.path(tmp, "model-output"), recursive = TRUE)
  dir.create(file.path(tmp, "model-metadata"))
  file.copy(file.path(hub, "hub-config"), tmp, recursive = TRUE)
  tmp
}

find_metadata <- function(model_id, metadata_dir, hub) {
  dirs <- c(metadata_dir, file.path(hub, "model-metadata"))
  for (d in dirs) {
    for (ext in c("yml", "yaml")) {
      p <- file.path(d, paste0(model_id, ".", ext))
      if (file.exists(p)) {
        return(p)
      }
    }
  }
  NULL
}

# Flatten a hub_validations object into a data frame of checks.
tidy_checks <- function(v, file) {
  rows <- lapply(names(v), function(nm) {
    x <- v[[nm]]
    if (inherits(x, "hub_validations") || !inherits(x, "condition")) {
      return(tidy_checks(x, file))
    }
    cls <- class(x)[1]
    status <- switch(cls,
      check_success = "pass",
      check_failure = "fail",
      check_error = "error",
      check_exec_error = "error",
      check_info = "skip",
      check_exec_warn = "warn",
      check_warning = "warn",
      cls
    )
    msg <- gsub("\\s+", " ", cli::ansi_strip(conditionMessage(x)))
    data.frame(file = file, check = nm, status = status, message = msg)
  })
  do.call(rbind, rows)
}

validate_one <- function(file, opts) {
  hub <- normalizePath(opts$hub)
  tmp <- make_temp_hub(hub)
  on.exit(unlink(tmp, recursive = TRUE), add = TRUE)
  ext <- tools::file_ext(file)
  if (ext %in% c("yml", "yaml")) {
    file.copy(file, file.path(tmp, "model-metadata", basename(file)))
    v <- validate_model_metadata(
      hub_path = tmp, file_path = basename(file)
    )
  } else {
    model_dir <- basename(dirname(normalizePath(file)))
    dest <- file.path(tmp, "model-output", model_dir)
    dir.create(dest)
    file.copy(file, file.path(dest, basename(file)))
    if (!opts$no_metadata) {
      # hubValidations looks metadata up by the model_id in the file name.
      model_id <- sub(
        "^\\d{4}-\\d{2}-\\d{2}-(.*)\\.[^.]+$", "\\1", basename(file)
      )
      md <- find_metadata(model_id, opts$metadata_dir, hub)
      if (!is.null(md)) {
        file.copy(md, file.path(tmp, "model-metadata", basename(md)))
      }
    }
    v <- validate_submission(
      hub_path = tmp,
      file_path = file.path(model_dir, basename(file)),
      skip_submit_window_check = opts$skip_window
    )
  }
  tidy_checks(v, file)
}

main <- function() {
  opts <- parse_args(commandArgs(trailingOnly = TRUE))
  results <- lapply(opts$files, function(f) {
    tryCatch(validate_one(f, opts), error = function(e) {
      data.frame(
        file = f, check = "r_error", status = "error",
        message = conditionMessage(e)
      )
    })
  })
  res <- do.call(rbind, results)
  for (f in unique(res$file)) {
    r <- res[res$file == f, ]
    bad <- r[r$status %in% c("fail", "error"), ]
    verdict <- if (nrow(bad) == 0L) "VALID" else "INVALID"
    cat(sprintf("%s: %s\n", verdict, f))
    for (j in seq_len(nrow(bad))) {
      cat(sprintf("  [%s] %s: %s\n", bad$status[j], bad$check[j],
                  bad$message[j]))
    }
  }
  if (!is.null(opts$json)) {
    write_json(res, opts$json, auto_unbox = TRUE, pretty = TRUE)
  }
  if (any(res$status %in% c("fail", "error"))) quit(status = 1L)
}

main()
