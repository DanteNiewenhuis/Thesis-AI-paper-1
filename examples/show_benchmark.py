# %%

from __project_path import _

from project.Benchmarks_2d.Branin import Branin
from project.Benchmarks_2d.Easom import Easom
from project.Benchmarks_2d.GoldsteinPrice import GoldsteinPrice
from project.Benchmarks_2d.MartinGaddy import MartinGaddy
from project.Benchmarks_2d.Mishra4 import Mishra4
from project.Benchmarks_2d.SixHump import SixHump

import pandas as pd
from datetime import datetime
import os
import argparse

from project.Benchmarks_base.benchmark import Benchmark

# %%

benchmark_dict = {"Branin": Branin, "Easom": Easom, "GoldsteinPrice": GoldsteinPrice,
                  "MartinGaddy": MartinGaddy, "Mishra4": Mishra4, "SixHump": SixHump}


parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('-benchmark', type=str, default="Branin",
                    choices=benchmark_dict.keys())
parser.add_argument('-i', type=int, default=0)

args = parser.parse_args()

print(args)

# %%

benchmark_name = args.benchmark
benchmark = benchmark_dict[benchmark_name]
i = args.i
session = "latest"

if session == "latest":
    sessions = [datetime.strptime(x, '%Y-%m-%d_%H:%M:%S')
                for x in os.listdir(f"../results/{benchmark.__name__}")]

    session = datetime.strftime(max(sessions), '%Y-%m-%d_%H:%M:%S')

df = pd.read_csv(
    f"../results/{benchmark.__name__}/{session}/run-unique.csv")


if i >= len(df):
    raise ValueError(f"{benchmark_name} only has {len(df)} frames.")

param_columns = [c for c in df.columns if "p_" in c]
params = df.iloc[i][param_columns]

bench: Benchmark = benchmark(params)
bench.plot_benchmark()
# %%
