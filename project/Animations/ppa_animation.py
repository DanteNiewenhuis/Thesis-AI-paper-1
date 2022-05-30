# %%

import numpy as np
from rich import print
from PPA import PPA

import plotly as plt

# %%


def sphere(x):
    return np.sum(x**2, axis=1)


def schwefel(x):
    return 418.9829*x.shape[1] - np.sum((x * np.sin(np.sqrt(abs(x)))), axis=1)


pop_size = 20
n_max = 5
r = 0.9

D = 2
lower_bounds = np.array([-500, -500])
upper_bounds = np.array([500, 500])


# %%

ppa = PPA(pop_size, n_max, D, lower_bounds, upper_bounds, schwefel)
f, F = ppa.calculate_fitness(ppa.current_population)

print(f"loss at the start: {f.sum()}")

# %%
losses = []
for i in range(20):
    ppa.next_generation()

    f, F = ppa.calculate_fitness(ppa.current_population)

    losses.append(f.sum())

print(f"loss after training: {f.sum()}")
# %%

print(f"{len(ppa.populations)=}")

ppa.animate_populations()

# %%
