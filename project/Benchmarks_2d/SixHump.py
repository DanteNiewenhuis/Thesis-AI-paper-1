# %%

import re
from project.Benchmarks_base.benchmark import Benchmark
import numpy as np
import numpy.typing as npt
from typing import Union, Tuple, Annotated


number = int | float | complex

# %%

base_params = (1/3, 0, 2.1, 0, 4, 0, 4, 0, 4, 0, 1, 0)


class SixHump(Benchmark):

    def __init__(self, params: Annotated[Tuple[number], 12] = base_params,
                 lower: npt.NDArray = np.array([-2, -1]),
                 upper: npt.NDArray = np.array([2, 1]), D=2):
        super(SixHump, self).__init__(params, lower, upper)

    def _get_value(self, inp: npt.NDArray) -> npt.NDArray:
        """Get the value of a list of points

        Args:
            inp (npt.ArrayLike[npt.NDArray]): point matrix of shape (N, 2)

        Returns:
            npt.NDArray: list of values of shape (N)
        """
        x_1 = inp[:, 0]
        x_2 = inp[:, 1]
        p = self.params

        s_1 = p[0]*(x_1+p[1])**6 - p[2]*(x_1+p[3])**4 + p[4]*(x_1+p[5])**2
        s_2 = p[6]*(x_2+p[7])**4 - p[8]*(x_2+p[9])**2
        s_3 = p[10]*x_1*x_2

        return s_1 + s_2 + s_3 + p[11]

    def _to_latex(self, r=5, line_breaks=True, parameterized=False, double=False):
        p_f = np.round(self.params, r)

        # all parameters that multiply
        p = ["" for _ in p_f]
        for i in [0, 2, 4, 6, 8, 10]:
            p[i] = "" if p_f[i] == 1 else str(p_f[i])

        # all parameters that add
        for i in [1, 3, 5, 7, 9, 11]:
            p[i] = "" if p_f[i] == 0 else str(p_f[i])

        if parameterized:
            p = [f"p_{'{'}{i}{'}'}" for i in range(len(p))]

        res = f"${p[0]}(x_1+{p[1]})^6 - {p[2]}(x_1+{p[3]})^4 + {p[4]}(x_1+{p[5]})^2 +{p[6]}(x_2+{p[7]})^4 - {p[8]}(x_2+{p[9]})^2 + {p[10]}x_1x_2 + {p[11]}$"

        if line_breaks:
            res = f"${p[0]}(x_1+{p[1]})^6 - {p[2]}(x_1+{p[3]})^4 + {p[4]}(x_1+{p[5]})^2 +$ \\\\ ${p[6]}(x_2+{p[7]})^4 - {p[8]}(x_2+{p[9]})^2 + {p[10]}x_1x_2 + {p[11]}$"

        return res

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
