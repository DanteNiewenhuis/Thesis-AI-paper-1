# %%

from project.Benchmarks_base.benchmark import Benchmark
import numpy as np
import numpy.typing as npt
from typing import Annotated, Tuple

number = int | float | complex

# %%
base_params = (1/4000, 0,
               1, 1, 0,
               1)


class Griewank(Benchmark):

    def __init__(self, params: Annotated[Tuple[number], 6] = base_params,
                 D=2,
                 lower: list[number] = None,
                 upper: list[number] = None):

        if not isinstance(lower, np.ndarray):
            lower = np.array([-600 for _ in range(D)])
            upper = np.array([600 for _ in range(D)])

        super(Griewank, self).__init__(params, lower, upper, D)

    def get_true_extremes(self):
        true_minimum = np.zeros([1, self.D])

        true_minimum = np.concatenate(
            [true_minimum[0, :], self.get_value(true_minimum)])

        raise NotImplementedError
        true_maximum = np.array([self.upper_bounds])

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

        s_1 = p[0] * np.sum((inp + p[1])**2, axis=1)
        s_2 = -np.prod(p[2] * np.cos(p[3] /
                       np.sqrt(np.arange(self.D) + 1) * inp + p[4]), axis=1)

        return (s_1 + s_2) + p[5]

    def deriv_x(self, point: Tuple[number, number]) -> number:
        """Get the derivative towards dimension x at a given point

        Args:
            point (Tuple[number, number]): x, y

        Returns:
            number: derivative towards x at the given point
        """
        x = point[0]
        y = point[1]

        p_1 = -np.cos(self.params[1]*x)*np.cos(self.params[2]*y)
        p_2 = np.exp(self.params[3]*(-(self.params[4]*x - self.params[5]*np.pi)**2
                                     - (self.params[6]*y - self.params[7]*np.pi)**2))

        p_1_d = self.params[1] * \
            np.sin(self.params[1]*x)*np.cos(self.params[2]*y)
        p_2_d = -2*self.params[3]*self.params[4] * \
            (self.params[4]*x - self.params[5]*np.pi) * p_2

        return self.params[0]*(p_1*p_2_d + p_2*p_1_d)

    def deriv_y(self, point: Tuple[number, number]) -> number:
        """Get the derivative towards dimension y at a given point

        Args:
            point (Tuple[number, number]): x, y

        Returns:
            number: derivative towards y at the given point
        """

        x = point[0]
        y = point[1]

        p_1 = -np.cos(self.params[1]*x)*np.cos(self.params[2]*y)
        p_2 = np.exp(self.params[3]*(-(self.params[4]*x - self.params[5]*np.pi)**2
                                     - (self.params[6]*y - self.params[7]*np.pi)**2))

        p_1_d = self.params[2] * \
            np.sin(self.params[2]*y)*np.cos(self.params[1]*x)
        p_2_d = -2*self.params[3]*self.params[6] * \
            (self.params[6]*y - self.params[7]*np.pi) * p_2

        return self.params[0]*(p_1*p_2_d + p_2*p_1_d)

    # def _get_value_2(self, inp: list[number]):
    #     return self.get_value(np.array([inp]))
