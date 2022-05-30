# %%

from project.Benchmarks_base.benchmark import Benchmark
import numpy as np
import numpy.typing as npt
from typing import Annotated, Tuple

number = int | float | complex

# %%

base_params = (100, 1, 1, 0, 1, 1, 0)


class RosenBrock(Benchmark):

    def __init__(self, params: Annotated[Tuple[number], 7] = base_params,
                 D=2,
                 lower: list[number] = None,
                 upper: list[number] = None):

        if not isinstance(lower, np.ndarray):
            lower = np.array([-30 for _ in range(D)])
            upper = np.array([30 for _ in range(D)])

        super(RosenBrock, self).__init__(params, lower, upper, D)

    def get_true_extremes(self):
        true_minimum = np.ones([1, self.D])

        true_minimum = np.concatenate(
            [true_minimum[0, :], self.get_value(true_minimum)])

        true_maximum = np.array([self.lower_bounds])

        true_maximum = np.concatenate(
            [true_maximum[0, :], self.get_value(true_maximum)])

        return true_minimum, true_maximum

    def _get_value(self, inp: npt.NDArray) -> npt.NDArray:
        """Get the value of a list of points

        Args:
            inp (npt.ArrayLike[npt.NDArray]): point matrix of shape (N, 2)

        Returns:
            npt.NDArray: list of values of shape (N)
        """

        p = self.params

        s_1 = np.sum((p[1]*inp[:, 1:] - p[2]*(inp[:, :-1]+p[3])**2)
                     ** 2 + p[4]*(inp[:, :-1] - p[5])**2, axis=1)

        return s_1 + p[6]

    def deriv_x(self, point: Tuple[number, number]) -> number:
        """Get the derivative towards dimension x at a given point

        Args:
            point (Tuple[number, number]): x, y

        Returns:
            number: derivative towards x at the given point
        """
        raise NotImplementedError

    def deriv_y(self, point: Tuple[number, number]) -> number:
        """Get the derivative towards dimension y at a given point

        Args:
            point (Tuple[number, number]): x, y

        Returns:
            number: derivative towards y at the given point
        """

        raise NotImplementedError
