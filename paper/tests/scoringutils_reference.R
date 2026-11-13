# Score tests/fixtures/scoring-fixture.csv with scoringutils and write the
# expected scores to tests/fixtures/scoringutils-expected.csv.
# Run from paper/: Rscript tests/scoringutils_reference.R
# Uses scoringutils' defaults: WIS with weigh = TRUE and the median
# counted once; CRPS by scoringRules::crps_sample ("edf"); bias_quantile
# and bias_sample; coverage of central intervals, from type-7 sample
# quantiles for sample forecasts. Log scores use log_shift(offset = 1).
suppressPackageStartupMessages({
  library(scoringutils)
  library(data.table)
})
fixture <- fread("tests/fixtures/scoring-fixture.csv")
setnames(fixture, c("value", "truth"), c("predicted", "observed"))
keys <- c("study", "model", "location", "origin_date", "target_date", "horizon")

log1p_forecast <- function(fc) {
  transform_forecasts(fc, fun = log_shift, offset = 1, append = FALSE)
}

quant <- fixture[!is.na(quantile_level)][, sample_id := NULL]
qfc <- as_forecast_quantile(quant, forecast_unit = keys)
qs <- score(qfc)
qlog <- score(log1p_forecast(qfc), metrics = list(wis = wis))
q_out <- qs[, c(keys, "wis", "bias", "interval_coverage_50",
  "interval_coverage_90"), with = FALSE]
q_out[qlog, log_score := i.wis, on = keys]
setnames(q_out, "wis", "score")
q_out[, metric := "wis"]

samp <- fixture[!is.na(sample_id)][, quantile_level := NULL]
sfc <- as_forecast_sample(samp, forecast_unit = keys)
ss <- score(sfc, metrics = list(crps = crps_sample, bias = bias_sample))
slog <- score(log1p_forecast(sfc), metrics = list(crps = crps_sample))
scov <- score(
  as_forecast_quantile(sfc, probs = c(0.05, 0.25, 0.5, 0.75, 0.95), type = 7),
  metrics = get_metrics(qfc, select = c(
    "interval_coverage_50", "interval_coverage_90"
  ))
)
s_out <- ss[, c(keys, "crps", "bias"), with = FALSE]
s_out[slog, log_score := i.crps, on = keys]
s_out[scov, `:=`(
  interval_coverage_50 = i.interval_coverage_50,
  interval_coverage_90 = i.interval_coverage_90
), on = keys]
setnames(s_out, "crps", "score")
s_out[, metric := "crps"]

out <- rbind(q_out, s_out, use.names = TRUE)
setnames(
  out,
  c("interval_coverage_50", "interval_coverage_90"),
  c("coverage_50", "coverage_90")
)
out[, coverage_50 := as.numeric(coverage_50)]
out[, coverage_90 := as.numeric(coverage_90)]
setcolorder(out, c(keys, "metric", "score", "log_score", "bias"))
fwrite(out, "tests/fixtures/scoringutils-expected.csv")
cat("scoringutils", as.character(packageVersion("scoringutils")), "\n")
