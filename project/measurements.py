from cgitb import small
from project.Algorithms import PPA
from multiprocessing import Pool
import numpy as np
import numpy.typing as npt
from typing import Callable, Tuple, Any

from project.Benchmarks_base.benchmark import Benchmark
import time
import plotly.graph_objects as go

import scipy

number = int | float | complex

################################################################################
# measurments
################################################################################


def divide_success(instance_results: npt.ArrayLike, minima: npt.ArrayLike,
                   max_dis: number, r: number = 1) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
    """Divide the results of multiple runs into successful, and unsuccessful 

    Args:
        instance_results (npt.ArrayLike): _description_
        minima (npt.ArrayLike): _description_
        max_dis (number): _description_
        r (number, optional): _description_. Defaults to 1.

    Returns:
        Tuple[npt.ArrayLike, npt.ArrayLike]: _description_
    """
    x = instance_results[:, :-1]
    minima = minima[:, :-1]

    dis = scipy.spatial.distance.cdist(x, minima)

    correct_idx = np.where(dis < max_dis/100)[0]
    incorrect_idx = np.array(list(set(range(100)) - set(correct_idx)))

    return correct_idx, incorrect_idx


def failure_rate(instance_results: npt.ArrayLike, minima: npt.ArrayLike,
                 max_dis: number, r: number = 1) -> number:
    """Determine the Failure rate based on the domain distance to the global minima

    Args:
        instance_results (npt.ArrayLike): The results of multiple runs on the 
            used benchmark
        minima (npt.ArrayLike): The global minima of the used benchmark
        max_dis (number): The maximal domain distance possible in the used benchmark
        r (number, optional): The allowed distance to the global minima. Defaults to 1.

    Returns:
        number: The percentage of unsuccessful runs.
    """

    correct_idx, incorrect_idx = divide_success(
        instance_results, minima, max_dis, r)

    return (len(incorrect_idx) / len(instance_results)) * 100


def success_rate(instance_results: npt.ArrayLike, minima: npt.ArrayLike,
                 max_dis: number, r: number = 1) -> number:
    """Determine the success rate based on the domain distance to the global minima

    Args:
        instance_results (npt.ArrayLike): The results of multiple runs on the 
            used benchmark
        minima (npt.ArrayLike): The global minima of the used benchmark
        max_dis (number): The maximal domain distance possible in the used benchmark
        r (number, optional): The allowed distance to the global minima. Defaults to 1.

    Returns:
        number: The percentage of successful runs.
    """

    correct_idx, incorrect_idx = divide_success(
        instance_results, minima, max_dis, r)

    return (len(correct_idx) / len(instance_results)) * 100


def mean_objective_deficiency(instance_results: npt.ArrayLike, bench_min: npt.ArrayLike,
                              val_range: number, minima, pos_range: number,  r: number = 1) -> number:
    """The mean of the normalized fitness of the instance results

    Args:
        zs (npt.NDArray): z values of the instance results
        val_range (number): The max range of the benchmark

    Returns:
        number: The normalized mean of results
    """

    zs = instance_results[:, -1]
    return normalize(np.mean(zs), bench_min[-1], val_range)


def best_fitness(instance_results: npt.ArrayLike, x_min: number, y_min: number, z_min: number,
                 val_range: number, minima, pos_range: number,  r: number = 1) -> number:
    """The best normalized fitness of the instance results

    Args:
        zs (npt.NDArray): z values of the instance results
        val_range (number): The max range of the benchmark

    Returns:
        number: The best instance result normalized
    """
    zs = instance_results[:, -1]
    return normalize(np.min(zs), z_min, val_range)


def worst_fitness(instance_results: npt.ArrayLike, x_min: number, y_min: number, z_min: number,
                  val_range: number, minima, pos_range: number,  r: number = 1) -> number:
    """The worst normalized fitness of the instance results

    Args:
        zs (npt.NDArray): z values of the instance results
        val_range (number): The max range of the benchmark

    Returns:
        number: The worst instance result normalized
    """
    zs = instance_results[:, -1]
    return normalize(np.max(zs), z_min, val_range)

################################################################################
# Functions
################################################################################


def evolve_ppa(inp: Tuple[int, Benchmark]) -> Tuple[number, number, number]:
    """Evolve a single PPA instance on the benchmark

    Args:
        inp ([type]): Initial seed

    Returns:
        Tuple[number, number, number]: x, y, z
    """
    i, bench = inp

    np.random.seed(i)

    ppa = PPA.PPA(bench)

    ppa.evolve(5_000)

    return ppa.get_current_fitness()


def run_PPAs(benchmark: Benchmark, loops: int = 100, threads: int = 10):
    inp = ((np.random.randint(10000), benchmark) for _ in range(loops))
    with Pool(threads) as p:
        res = np.array(p.map(evolve_ppa, inp))

    return np.array(res)


def normalize(val: number, val_min: number, val_range: number):
    """Normalize value using min-max normalizetion. Return value as a percentage

    Args:
        val (number): z-value given
        val_min (number): global minimum of benchmark
        val_range (number): Difference between global minimum and maximum of the 
            used benchmark 

    Returns:
        _type_: normalized z-value
    """
    return (val - val_min) / val_range * 100


def get_objective(bench: Benchmark,
                  oj: Callable[[npt.NDArray, number, number, number, number], number],
                  sampling: bool = True, loops: int = 100, threads: int = 10) \
        -> number:

    instance_results = run_PPAs(bench, loops, threads)

    if sampling:
        val_range = bench.get_val_range()
        bench_min, bench_max = bench.get_extremes()

    else:
        val_range = bench.get_val_range()
        bench_min, bench_max = bench.get_extremes()

    # TODO: Look into get_minima()
    return oj(instance_results, bench_min, val_range, [], bench.max_dis)


def get_results(bench: Benchmark, loops: int = 100, threads: int = 10):

    instance_results = run_PPAs(bench, loops, threads)

    val_range = bench.get_val_range()
    bench_min, bench_max = bench.get_extremes()

    minima = bench.get_minima()

    fr = failure_rate(instance_results, minima, bench.max_dis)

    mbf = mean_objective_deficiency(
        instance_results, bench_min, val_range, bench.get_minima(), bench.max_dis)

    return fr, mbf


# TODO: move to _plotting
def plot_runs(bench: Benchmark, instance_results=None, minima=None, loops: int = 100, threads: int = 10):

    if instance_results == None:
        instance_results = run_PPAs(bench, loops, threads)

    if minima == None:
        minima = bench.get_minima()

    correct_idx, incorrect_idx = divide_success(
        instance_results, minima, bench.max_dis)

    correct = instance_results[correct_idx, :]
    incorrect = instance_results[incorrect_idx, :]

    data = bench.get_points_data([correct, incorrect])

    data[1]["marker"]["color"] = "green"
    data[1]["name"] = "Success"
    data[1]["marker"]["size"] = 3

    data[2]["marker"]["color"] = "red"
    data[2]["name"] = "Failure"
    data[2]["marker"]["size"] = 3

    return go.Figure(data=data, layout=bench.get_layout())
