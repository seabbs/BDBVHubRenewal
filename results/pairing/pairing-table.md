## Germany, development half (80 dates, states task)

| model | target | arm | n | WIS | rel. WIS | 50% cov. | 90% cov. |
|---|---|---|---|---|---|---|---|
| randomwalknowcast | national | off | 400 | 139.9 | 1.000 | 0.35 | 0.75 |
| randomwalknowcast | national | fixed | 400 | 145.0 | 1.036 | 0.70 | 0.99 |
| randomwalknowcast | national | rolling | 400 | 126.6 | 0.905 | 0.39 | 0.87 |
| randomwalknowcast | states | off | 6400 | 13.9 | 1.000 | 0.55 | 0.87 |
| randomwalknowcast | states | fixed | 6400 | 13.9 | 1.000 | 0.55 | 0.87 |
| randomwalknowcast | states | rolling | 6400 | 13.9 | 1.000 | 0.55 | 0.87 |

## BDBV release backtest

| model | target | arm | n | CRPS | rel. CRPS | 50% cov. | 90% cov. |
|---|---|---|---|---|---|---|---|
| randomwalknowcast | cell | off | 156 | 0.4 | 1.000 | 0.96 | 0.97 |
| randomwalknowcast | cell | fixed | 156 | 0.4 | 1.000 | 0.96 | 0.97 |
| randomwalknowcast | cell | rolling | 156 | 0.4 | 1.000 | 0.96 | 0.97 |
| randomwalknowcast | national | off | 6 | 8.2 | 1.000 | 0.33 | 0.50 |
| randomwalknowcast | national | fixed | 6 | 7.4 | 0.912 | 0.33 | 0.50 |
| randomwalknowcast | national | rolling | 6 | 7.4 | 0.913 | 0.33 | 0.50 |
| randomwalknowcast | national_total | off | 3 | 14.0 | 1.000 | 0.33 | 0.67 |
| randomwalknowcast | national_total | fixed | 3 | 11.7 | 0.832 | 0.33 | 1.00 |
| randomwalknowcast | national_total | rolling | 3 | 11.7 | 0.833 | 0.33 | 1.00 |
| randomwalknowcast | province_total | off | 78 | 0.7 | 1.000 | 0.96 | 0.97 |
| randomwalknowcast | province_total | fixed | 78 | 0.7 | 0.942 | 0.96 | 0.97 |
| randomwalknowcast | province_total | rolling | 78 | 0.7 | 0.943 | 0.96 | 0.97 |
| randomwalknowcast | provinces_sum | off | 6 | 8.2 | 1.000 | 0.33 | 0.50 |
| randomwalknowcast | provinces_sum | fixed | 6 | 7.4 | 0.912 | 0.33 | 0.50 |
| randomwalknowcast | provinces_sum | rolling | 6 | 7.4 | 0.913 | 0.33 | 0.50 |

| model | scale | arm | energy | rel. energy | log energy |
|---|---|---|---|---|---|
| randomwalknowcast | national | off | 11.5 | 1.000 | 0.027 |
| randomwalknowcast | national | fixed | 10.5 | 0.913 | 0.024 |
| randomwalknowcast | national | rolling | 10.6 | 0.914 | 0.024 |
| randomwalknowcast | provinces | off | 11.9 | 1.000 | 0.070 |
| randomwalknowcast | provinces | fixed | 11.7 | 0.980 | 0.066 |
| randomwalknowcast | provinces | rolling | 11.7 | 0.982 | 0.066 |

## BDBV health zones (zones forecast directly)

| model | target | arm | n | score | 50% cov. | 90% cov. |
|---|---|---|---|---|---|---|
| randomwalknowcast | cell | fixed | 39 | CRPS 1.27 | 0.64 | 0.72 |
| randomwalknowcast | cell | off | 39 | CRPS 1.27 | 0.64 | 0.72 |
| randomwalknowcast | cell | rolling | 39 | CRPS 1.27 | 0.64 | 0.72 |
| randomwalknowcast | energy | fixed | 6 | energy 6.03, log 0.534 |  |  |
| randomwalknowcast | energy | off | 6 | energy 5.99, log 0.530 |  |  |
| randomwalknowcast | energy | rolling | 6 | energy 6.03, log 0.533 |  |  |
| randomwalknowcast | zone_sum | fixed | 6 | CRPS 5.55 | 0.50 | 0.83 |
| randomwalknowcast | zone_sum | off | 6 | CRPS 5.75 | 0.50 | 0.83 |
| randomwalknowcast | zone_sum | rolling | 6 | CRPS 5.53 | 0.50 | 0.83 |
| randomwalknowcast | zone_total | fixed | 39 | CRPS 1.27 | 0.64 | 0.72 |
| randomwalknowcast | zone_total | off | 39 | CRPS 1.27 | 0.64 | 0.72 |
| randomwalknowcast | zone_total | rolling | 39 | CRPS 1.27 | 0.64 | 0.72 |
