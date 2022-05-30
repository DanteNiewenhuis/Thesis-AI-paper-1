# %%
from multiprocessing import Pool
import numpy as np
from PPA import PPA
from FunctionEvolver import FunctionEvolver
from Benchmarks_base.benchmark import Schwefel

from rich import print
import plotly as plt

import time

# %%

pop_size = 20
n_max = 5

D = 2
lower_bounds = np.array([0, 0, 0, 0])
upper_bounds = np.array([3, 3, 3, 3])

# %%

hc = FunctionEvolver(Schwefel, 4, lower_bounds, upper_bounds)

# %%

hc.train(10)
# %%

hc.parameters
# %%
