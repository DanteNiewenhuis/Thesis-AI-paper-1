# %%

from project.Algorithms.FunctionEvolver import FunctionEvolver
from project.measurements import *
from project.Benchmarks_nd.Tablet import Tablet
from project.Benchmarks_nd.Sphere import Sphere
from project.Benchmarks_nd.Schwefel import Schwefel
from project.Benchmarks_nd.RosenBrock import RosenBrock
from project.Benchmarks_nd.Rastrigin import Rastrigin
from project.Benchmarks_nd.Griewank import Griewank
from project.Benchmarks_nd.Ellipse import Ellipse
from project.Benchmarks_nd.Cigar import Cigar
from project.Benchmarks_nd.Ackley import Ackley
from project.Benchmarks_2d.SixHump import SixHump
from project.Benchmarks_2d.MartinGaddy import MartinGaddy
from project.Benchmarks_2d.GoldsteinPrice import GoldsteinPrice
from project.Benchmarks_2d.Easom import Easom
from project.Benchmarks_2d.Branin import Branin
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import numpy as np
import pandas as pd
from typing import Annotated
import scipy
import re
from genericpath import samefile
import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))


# %%


def get_error(g: Benchmark, s: int = 1_000_000):
    sample_min, sample_max = g.get_extremes(sample_size=s)
    true_min, true_max = g.get_true_extremes()

    sample_val_range = sample_max[-1] - sample_min[-1]
    true_val_range = true_max[-1] - true_min[-1]

    min_domain_error = scipy.spatial.distance.cdist(
        true_min[:-1].reshape(1, -1), sample_min[:-1].reshape(1, -1))[0]
    min_domain_error /= g.max_dis

    max_domain_error = scipy.spatial.distance.cdist(
        true_max[:-1].reshape(1, -1), sample_max[:-1].reshape(1, -1))[0]
    max_domain_error /= g.max_dis

    range_error = np.abs(true_val_range - sample_val_range) / true_val_range

    min_error = np.abs(true_min[-1] - sample_min[-1]) / true_val_range
    max_error = np.abs(true_max[-1] - sample_max[-1]) / true_val_range

    return min_error, max_error, range_error, min_domain_error[0], max_domain_error[0]


def range_experiment(benchmark: Benchmark, dimensions: list[int] = [2, 3, 4, 5, 6, 7, 8, 9, 10,
                                                                    15, 20, 30, 40], sample_size: int = 1_000_000) -> pd.DataFrame:

    columns = ["d", "min_error_mean", "max_error_mean", "range_error_mean", "min_domain_error_mean", "max_domain_error_mean",
               "min_error_std", "max_error_std", "range_error_std", "min_domain_error_std", "max_domain_error_std",
               "min_error_min", "max_error_min", "range_error_min", "min_domain_error_min", "max_domain_error_min",
               "min_error_max", "max_error_max", "range_error_max", "min_domain_error_max", "max_domain_error_max", ]
    res = []
    for d in dimensions:
        print(f"dimenionality: {d}")
        min_errors = []
        max_errors = []
        range_errors = []
        min_domain_errors = []
        max_domain_errors = []

        for i in range(5):
            g = benchmark(D=d)
            min_error, max_error, range_error, min_domain_error, max_domain_error = get_error(
                g, sample_size)

            min_errors.append(min_error*100)
            max_errors.append(max_error*100)
            range_errors.append(range_error*100)
            min_domain_errors.append(min_domain_error*100)
            max_domain_errors.append(max_domain_error*100)

        res.append([
            d, np.mean(min_errors), np.mean(max_errors),
            np.mean(range_errors), np.mean(
                min_domain_errors), np.mean(max_domain_errors),
            np.std(min_errors), np.std(max_errors),
            np.std(range_errors), np.std(
                min_domain_errors), np.std(max_domain_errors),
            np.min(min_errors), np.min(max_errors),
            np.min(range_errors), np.min(
                min_domain_errors), np.min(max_domain_errors),
            np.max(min_errors), np.max(max_errors),
            np.max(range_errors), np.max(min_domain_errors), np.max(max_domain_errors)])

    df = pd.DataFrame(res, columns=columns)

    plot_results(df, benchmark.__name__, sample_size)


def plot_results(df: pd.DataFrame, bench_name: str, sample_size: int):
    data = []

    colors = ["255, 0, 0", "0, 255, 0",
              "0, 0, 255", "255, 0, 255", "0, 255, 255"]
    names = ["min", "max", "range", "min_domain"]

    for name, color in zip(names, colors):

        line_data = [
            go.Scatter(
                name=f'{name}',
                x=df["d"],
                y=df[f"{name}_error_mean"],
                mode="lines",
                line=dict(color=f"rgb({color})"),
            ),
            go.Scatter(
                name='Upper Bound',
                x=df["d"],
                y=df[f"{name}_error_mean"] + df[f"{name}_error_std"],
                mode="lines",
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False
            ),
            go.Scatter(
                name='Lower Bound',
                x=df['d'],
                y=df[f"{name}_error_mean"] - df[f"{name}_error_std"],
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                fillcolor=f'rgba({color}, 0.3)',
                fill='tonexty',
                showlegend=False
            )
        ]

        data += line_data

    df.to_csv(f"results/range_experiments/{bench_name}_{sample_size}.png")

    fig = go.Figure(data=data,
                    layout=dict(xaxis=dict(title=f"Dimenions"),
                                yaxis=dict(title=f"Error(%)")))

    fig.write_image(
        f"results/range_experiments/{bench_name}_{sample_size}.png")

    # fig.show()

# %%


benchmarks = [Ackley, Cigar, Ellipse, Griewank,
              Rastrigin, RosenBrock, Schwefel, Sphere, Tablet]
benchmarks = [Ackley, Cigar, Ellipse,
              Rastrigin, RosenBrock, Schwefel, Sphere, Tablet]

for bench in benchmarks:
    print(bench)
    range_experiment(bench)
