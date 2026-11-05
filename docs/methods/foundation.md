# Foundation model

`fm_delay`, the pretrained Toto 1.0 time-series model used zero-shot on the reporting delays.
{ .page-subtitle }

The data and notation are on the [Data methods page](data.md#target-and-notation).
`fm_delay` is a candidate, not submitted ([issue #20](https://github.com/seabbs/BDBVHubRenewal/issues/20)).

## Model

Toto 1.0 ([Cohen et al. 2025](https://arxiv.org/abs/2505.14766)) is a 151M-parameter decoder-only transformer pretrained on observability metrics and public time series.
It reads several series together, with one variate-attention layer in twelve, and draws sample paths autoregressively, so each draw is one joint trajectory of every series.
`fm_delay` uses the published checkpoint with no fine-tuning.

## Inputs

Each case in the vintage's delay line data has an onset week $w$ and a delay of $d$ days, in weekly delay class $k = \lfloor d / 7 \rfloor$ for $k < K = 9$.
Negative delays go to class 0 and delays of 63 days or more (about 0.2%) are left out.
$n_{p,w,k}$ counts the cases of province $p$ in cell $(w, k)$ at vintage $T$.
A cell is complete when its last possible report day, $w + 7k + 6$, is on or before $T$.

Toto forecasts every series from the same last time step.
Its padding mask only weights its scaler, so it cannot mask a ragged end.
Each delay class therefore goes in on the report clock: series $(p, k)$ holds $n_{p, s - k, k}$ at report week $s$.
Cell $(w, k)$ is complete exactly when $w + k \le s^*$, the latest week whose Saturday is at least six days before $T$, so every series is complete up to $s^*$ and ends there.
The $26 K$ series are one group, so variate attention sees every province and delay together.

No delay covariate is added.
Toto takes known-future covariates only through labels learnt in fine-tuning, and its scaler maps a constant series to zero.

## Trajectories

Toto draws every series jointly for the weeks past $s^*$ needed by the last delay class of the last target week.
Draws are rounded and floored at zero.
Sample $i$ of week $w$ is the reported total plus the new reports of the incomplete cells,

$$
y^{(i)}_{p,w} = R_{p,w} + \sum_{k : w + k > s^*} \max\bigl(\hat n^{(i)}_{p,w,k} - n_{p,w,k}, 0\bigr),
$$

with $R_{p,w}$ the reported total of known and imputed onsets and $\hat n^{(i)}_{p,w,k}$ the draw of series $(p, k)$ at report week $w + k$.
Weeks after the data have $R = 0$ and $n = 0$.
National is the sum of provinces by sample.

## Limitations

The delay line data hold known onsets only, so onsets the hub imputes later are not forecast, and forecast weeks miss them entirely.
Toto reads patches of 64 time steps, so the whole weekly history is one left-padded patch and the forecast comes from one decoding step.
Toto's output is continuous, so small counts are rounded draws of a continuous mixture.
Health zones have no delay line data, so the model runs at province level only.

## Running it

Toto is an optional dependency, with torch built for CUDA 12.6 on Linux.

```bash
uv sync --extra toto
uv run bdbvhubrenewal backtest --models fm_delay
```

Without it, the backtest skips `fm_delay` and says so.
