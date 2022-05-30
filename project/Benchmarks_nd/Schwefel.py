# %%

from project.Benchmarks_base.benchmark import Benchmark
import numpy as np
import numpy.typing as npt
from typing import Union, Tuple, Annotated

number = int | float | complex

# %%

base_params = (1, 1, 1, 0, 0, 0)


class Schwefel(Benchmark):

    def __init__(self, params: Annotated[Tuple[number], 7] = base_params,
                 D=2,
                 lower: list[number] = None,
                 upper: list[number] = None):

        if not isinstance(lower, np.ndarray):
            lower = np.array([-500 for _ in range(D)])
            upper = np.array([500 for _ in range(D)])

        super(Schwefel, self).__init__(params, lower, upper, D)

    def get_true_extremes(self):
        true_minimum = np.ones([1, self.D]) * 420.9687

        true_minimum = np.concatenate(
            [true_minimum[0, :], self.get_value(true_minimum)])

        true_maximum = np.ones([1, self.D]) * -420.9687

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

        s_1 = -p[0]/self.D * \
            np.sum(
                inp*np.sin(p[1]*np.sqrt(np.abs(p[2]*inp + p[3])) + p[4]), axis=1)
        return s_1 + p[5]

    def deriv_x(self, point: Tuple[number, number]) -> number:
        """Get the derivative towards dimension x at a given point

        Args:
            point (Tuple[number, number]): x, y

        Returns:
            number: derivative towards x at the given point
        """
        x = point[0]

        abs_val = np.abs(self.params[1] * x)
        abs_sqrt = np.sqrt(abs_val)

        p_0 = self.params[1]**2 * x**2 * np.cos(abs_sqrt + self.params[2])
        p_1 = 2 * abs_val * abs_sqrt
        p_2 = np.sin(abs_sqrt + self.params[2])

        return self.params[0]*(p_0/p_1 + p_2)

    def deriv_y(self, point: Tuple[number, number]) -> number:
        """Get the derivative towards dimension y at a given point

        Args:
            point (Tuple[number, number]): x, y

        Returns:
            number: derivative towards y at the given point
        """

        y = point[1]

        abs_val = np.abs(self.params[1] * y)
        abs_sqrt = np.sqrt(abs_val)

        p_0 = self.params[1]**2 * y**2 * np.cos(abs_sqrt + self.params[2])
        p_1 = 2 * abs_val * abs_sqrt
        p_2 = np.sin(abs_sqrt + self.params[2])

        return self.params[0]*(p_0/p_1 + p_2)

    # def _get_value_2(self, inp: list[number]):
    #     return self.get_value(np.array([inp]))
