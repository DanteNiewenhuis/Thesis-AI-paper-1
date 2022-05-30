# %%

from __project_path import _

from datetime import datetime

from project.Benchmarks_2d.Branin import Branin
from project.Benchmarks_2d.Easom import Easom
from project.Benchmarks_2d.GoldsteinPrice import GoldsteinPrice
from project.Benchmarks_2d.MartinGaddy import MartinGaddy
from project.Benchmarks_2d.Mishra4 import Mishra4
from project.Benchmarks_2d.SixHump import SixHump

import numpy as np
import pandas as pd
import os

from project.Benchmarks_base.benchmark import Benchmark

# %%


def make_unique_df(benchmark: Benchmark, folder: str = "results", session: str = "latest"):
    if session == "latest":
        sessions = [datetime.strptime(x, '%Y-%m-%d_%H:%M:%S')
                    for x in os.listdir(f"../{folder}/{benchmark.__name__}")]

        session = datetime.strftime(max(sessions), '%Y-%m-%d_%H:%M:%S')

    df = pd.read_csv(
        f"../{folder}/{benchmark.__name__}/{session}/run.csv")

    df['iteration'] = df.index

    df = pd.DataFrame([g[1].iloc[0] for g in df.groupby('val')])

    df.to_csv(f"../{folder}/{benchmark.__name__}/{session}/run-unique.csv")


for benchmark in [Branin, Easom, GoldsteinPrice, MartinGaddy, Mishra4, SixHump]:
    make_unique_df(benchmark, folder="results")

# %%
