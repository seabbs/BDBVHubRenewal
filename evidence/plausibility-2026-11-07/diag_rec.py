"""Raw zone walk sums against the province walk, before reconcile."""
import argparse, sys
from pathlib import Path
import pandas as pd
from bdbvhubrenewal.cli import _task
from bdbvhubrenewal.data.tidy import load_daily_zones
from bdbvhubrenewal.data.zones import hub_zones, province_of, weekly_zone_counts
from bdbvhubrenewal.forecasters import get_model
from bdbvhubrenewal.forecasters.zones import zone_task

data = Path(sys.argv[1])
args = argparse.Namespace(data=str(data), vintage="latest", backtest_config="config/backtest.toml")
task, _ = _task(args)
daily = load_daily_zones(str(data))
counts = weekly_zone_counts(daily, task.vintage)
zones = hub_zones(data / "hub-config" / "tasks.json")
for name in sys.argv[2].split(","):
    m = get_model(name)
    prov = m.forecast(task)
    zt = zone_task(task, counts, zones, pd.concat([task.daily_history, daily]))
    raw = m.forecast(zt)
    raw["province"] = raw.location.map(province_of)
    zs = raw.groupby(["province", "target_end_date", "sample"]).value.sum()
    pv = prov.loc[prov.location.str.len() == 5].set_index(["location", "target_end_date", "sample"]).value
    pv.index.names = zs.index.names
    d = pd.DataFrame({"zones": zs, "prov": pv}).dropna().groupby(["province", "target_end_date"]).median()
    d["ratio"] = (d.prov / d.zones).round(2)
    print(name); print(d.unstack("province")["ratio"].to_string())
    raw.to_csv(data.parent.parent.parent.parent / f"raw-zones-{name}.csv.gz", index=False)
