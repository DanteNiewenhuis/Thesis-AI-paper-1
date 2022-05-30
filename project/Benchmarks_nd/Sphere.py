# %%

from project.Benchmarks_base.benchmark import Benchmark

import numpy as np
from typing import Annotated, Tuple

number = int | float | complex

# %%
base_params = (1, 0, 0)


class Sphere(Benchmark):

    def __init__(self, params: Annotated[Tuple[number], 3] = base_params,
                 D=2,
                 lower: list[number] = None,
                 upper: list[number] = None):

        if lower == None:
            lower = np.array([-100 for _ in range(D)])
            upper = np.array([100 for _ in range(D)])
        self.D = D

        super(Sphere, self).__init__(
            params, lower, upper, D)

    def get_true_extremes(self):
        true_minimum = np.zeros([1, self.D])

        true_minimum = np.concatenate(
            [true_minimum[0, :], self.get_value(true_minimum)])

        true_maximum = np.array([self.upper_bounds])

        true_maximum = np.concatenate(
            [true_maximum[0, :], self.get_value(true_maximum)])

        return true_minimum, true_maximum

    def _get_value(self, inp: list[number]):
        p = self.params

        return p[0] * np.sum((inp + p[1])**2, axis=1) + p[2]
