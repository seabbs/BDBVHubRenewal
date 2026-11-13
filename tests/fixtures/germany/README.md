# German hospitalisation fixture

`germany_covid19_hosp.csv.gz` is part of epinowcast's `germany_covid19_hosp` data set: COVID-19 hospitalisations in Germany by Meldedatum (`reference_date`) and report date, with the count reported by each report date (`confirm`).
It keeps Meldedatum 2021-08-26 to 2021-10-20 for Germany (all ages and the six age groups) and for Berlin and Bavaria (all ages).

Source: `data/germany_covid19_hosp.rda` in [epinowcast](https://github.com/epinowcast/epinowcast) at commit `0075d950c9322affc440db88ec2b235e34892671` (MIT licence).
epinowcast builds it from the [German hospitalisation nowcasting hub](https://github.com/KITmetricslab/hospitalization-nowcast-hub) truth data, which come from the Robert Koch Institute (CC BY 4.0).

It was read without R, with the [rdata](https://pypi.org/project/rdata/) package:

```python
import pandas as pd
import rdata

d = rdata.read_rda("germany_covid19_hosp.rda")["germany_covid19_hosp"]
epoch = pd.Timestamp("1970-01-01")
for col in ("reference_date", "report_date"):
    days = pd.to_timedelta(d[col].astype(int), "D")
    d[col] = (epoch + days).dt.date.astype(str)
keep = (d.location == "DE") | (
    (d.age_group == "00+") & d.location.isin(["DE-BE", "DE-BY"])
)
d = d.loc[keep & (d.reference_date >= "2021-08-26")]
d = d.astype({"location": str, "age_group": str, "confirm": int})
d = d.sort_values(["location", "age_group", "reference_date", "report_date"])
d.to_csv("germany_covid19_hosp.csv.gz", index=False)
```
