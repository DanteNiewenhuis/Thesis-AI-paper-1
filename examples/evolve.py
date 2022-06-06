
from __project_path import _

from project.Benchmarks_2d.Branin import Branin
from project.Benchmarks_2d.Easom import Easom
from project.Benchmarks_2d.GoldsteinPrice import GoldsteinPrice
from project.Benchmarks_2d.MartinGaddy import MartinGaddy
from project.Benchmarks_2d.Mishra4 import Mishra4
from project.Benchmarks_2d.SixHump import SixHump

from project.measurements import mean_objective_deficiency
from project.Algorithms.FunctionEvolver import FunctionEvolver

import argparse

benchmark_dict = {"Branin": Branin, "Easom": Easom, "GoldsteinPrice": GoldsteinPrice,
                  "MartinGaddy": MartinGaddy, "Mishra4": Mishra4, "SixHump": SixHump}


parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('-benchmark', type=str,
                    choices=benchmark_dict.keys())
parser.add_argument('-n', type=int, default=2000)

args = parser.parse_args()

benchmark_name = args.benchmark
benchmark = benchmark_dict[benchmark_name]
n = args.n

h = FunctionEvolver(benchmark, mean_objective_deficiency, base=True)
h.evolve(n)
