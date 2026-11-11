"""The maximum delay D per origin: the age by which 99% of the hub
target is reported under randomwalknowcast's chain ladder delay at the
origin, for the country."""
import numpy as np

from bdbvhubrenewal.forecasters.chain import delay_summary
from bdbvhubrenewal.forecasters.randomwalk import RandomWalk


def max_delay(task, q=0.99):
    origin = RandomWalk().origin_at(task, errors=False)
    s = delay_summary(origin, "x", task.vintage)
    nat = s.loc[s["location"] == "CD"].sort_values("x")
    hit = nat.loc[nat["q50"] >= q, "x"]
    return int(hit.iloc[0]) + 1 if len(hit) else int(nat["x"].max()) + 1, nat


if __name__ == "__main__":
    from bdbvhubrenewal.data import load_daily, load_delays, load_target
    from bdbvhubrenewal.evaluation.backtest import load_config
    from bdbvhubrenewal.evaluation.store import origin_tasks
    d = "data"
    tasks = origin_tasks(load_target(d), load_delays(d), load_config(), load_daily(d))
    for o, t in tasks.items():
        try:
            D, nat = max_delay(t)
        except Exception as e:
            print(o, "no delay:", type(e).__name__); continue
        sh = nat.set_index("x")["q50"]
        print(o, "D", D, "max age", int(sh.index.max()), "share at 0/7/14/21/28/35:", [round(float(sh.get(a, np.nan)), 3) for a in (0, 7, 14, 21, 28, 35)])
