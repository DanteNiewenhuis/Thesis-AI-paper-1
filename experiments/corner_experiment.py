# %%


from project.Benchmarks_2d.SixHump import SixHump
from project.Benchmarks_2d.Mishra4 import Mishra4
from project.Benchmarks_2d.MartinGaddy import MartinGaddy
from project.Benchmarks_2d.GoldsteinPrice import GoldsteinPrice
from project.Benchmarks_2d.Easom import Easom
from project.Benchmarks_2d.Branin import Branin
from datetime import datetime
import numpy as np
import pandas as pd
from project.Benchmarks_base.benchmark import Benchmark
import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))


# %%

def test_sample(benchmark, num_params):
    params = np.random.uniform(-10, 10, num_params)
    b: Benchmark = benchmark(params=params)

    # Get Sample
    sample_all = b.sample_benchmark(edges=True, corners=True)
    sample_edges = sample_all[:-4, :]
    sample_nothing = sample_edges[:-40_000, :]

    # Get indices
    max_idx_all = np.argmax(sample_all[:, -1])
    min_idx_all = np.argmin(sample_all[:, -1])
    max_idx_edges = np.argmax(sample_edges[:, -1])
    min_idx_edges = np.argmin(sample_edges[:, -1])
    max_idx_nothing = np.argmax(sample_nothing[:, -1])
    min_idx_nothing = np.argmin(sample_nothing[:, -1])

    # Get values
    max_val_all = sample_all[max_idx_all, -1]
    min_val_all = sample_all[min_idx_all, -1]
    max_val_edges = sample_edges[max_idx_edges, -1]
    min_val_edges = sample_edges[min_idx_edges, -1]
    max_val_nothing = sample_nothing[max_idx_nothing, -1]
    min_val_nothing = sample_nothing[min_idx_nothing, -1]

    # Get ranges
    range_all = max_val_all - min_val_all
    range_edges = max_val_edges - min_val_edges
    range_nothing = max_val_nothing - min_val_nothing

    corners_diff = np.abs(range_all - range_nothing) / range_all
    edges_diff = np.abs(range_nothing - range_edges) / range_all

    return params, range_all, range_edges, range_nothing, corners_diff, edges_diff

    # print(max_val_all)
    # print(max_val_edges)
    # print(max_val_no_edges)

    # print(min_val_all)
    # print(min_val_edges)
    # print(min_val_no_edges)

    # print(range_all)
    # print(range_edges)
    # print(range_no_edges)

    # print(f"{corners_diff = }")
    # print(f"{edges_diff = }")


def test_corners(benchmark, num_params, loops=10):
    corners_diffs = []
    edges_diffs = []

    edge_samples = 0
    corner_samples = 0
    other_samples = 0

    param_list = []

    for _ in range(loops):
        params, range_all, range_edges, range_no_edges, corners_diff, edges_diff = test_sample(
            benchmark, num_params)

        param_list.append(params)
        corners_diffs.append(corners_diff)
        edges_diffs.append(edges_diff)

        if range_all == range_no_edges:
            other_samples += 1
            continue

        if range_all == range_edges:
            edge_samples += 1
            continue

        corner_samples += 1

    corner_samples = float(corner_samples) / loops * 100
    edge_samples = float(edge_samples) / loops * 100
    other_samples = float(other_samples) / loops * 100
    return param_list, corners_diffs, edges_diffs, corner_samples, edge_samples, other_samples


# %%

benchmarks = [Branin, Easom, GoldsteinPrice, MartinGaddy, Mishra4, SixHump]
num_params_list = [11, 11, 28, 9, 14, 12]

for benchmark, num_params in zip(benchmarks, num_params_list):
    print(benchmark.__name__)

    param_list, corners_diffs, edges_diffs, corner_samples, edge_samples, other_samples = test_corners(
        benchmark, num_params, loops=100)

    print(f"{corner_samples}, {edge_samples}, {other_samples}, {np.mean(corners_diffs) * 100:.3f}")

# %%
