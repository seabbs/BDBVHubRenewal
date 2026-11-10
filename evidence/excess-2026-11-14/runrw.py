"""uv run python -I runrw.py VINTAGE REF NSAMP OUTCSV"""
import argparse, dataclasses, sys
import pandas as pd
from bdbvhubrenewal import cli
from bdbvhubrenewal.forecasters import get_model
v, ref, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
ns = argparse.Namespace(vintage=v, reference_date=ref, data="data", backtest_config="config/backtest.toml")
task, _ = cli._task(ns)
task = dataclasses.replace(task, n_samples=n)
s = get_model("randomwalknowcast").forecast(task)
s.to_csv(out, index=False)
