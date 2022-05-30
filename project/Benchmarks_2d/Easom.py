# %%

import re
from project.Benchmarks_base.benchmark import Benchmark
import numpy as np
import numpy.typing as npt
from typing import Annotated, Tuple

number = int | float | complex

# %%
base_params = (1, 1, 0, 1, 0, 1, -np.pi, 1, -np.pi, 0, 0)


class Easom(Benchmark):

    def __init__(self, params: Annotated[Tuple, 11] = base_params,
                 lower: npt.NDArray = np.array([-100, -100]),
                 upper: npt.NDArray = np.array([100, 100]), D=2):
        super(Easom, self).__init__(params, lower, upper)

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

        s_0 = -p[0] * np.cos(p[1]*x_1 + p[2]) * np.cos(p[3]*x_1 + p[4])
        s_1 = np.exp(-np.abs(p[5]*(x_1+p[6])**2 + p[7]*(x_2+p[8])**2 + p[9]))

        return (s_0 * s_1) + p[10]

    def _to_latex(self, r=5, line_breaks=True, parameterized=False, double=False):
        p_f = np.round(self.params, r)

        # all parameters that multiply
        p = ["XX" for _ in p_f]
        for i in [0, 1, 3, 5, 7]:
            p[i] = "" if p_f[i] == 1 else str(p_f[i])

        # all parameters that add
        for i in [2, 4, 6, 8, 9, 10]:
            p[i] = "" if p_f[i] == 0 else str(p_f[i])

        if parameterized:
            p = [f"p_{'{'}{i}{'}'}" for i in range(len(p))]

        res = f"$-{p[0]}\cos({p[1]}x_1 + {p[2]})\cos({p[3]}x_2 + {p[4]})\exp[-(|({p[5]}(x_1+{p[6]})^2 + {p[7]}(x_2+{p[8]})^2 + {p[9]}|)] + {p[10]}$"

        if line_breaks:
            res = f"$-{p[0]}\cos({p[1]}x_1 + {p[2]})\cos({p[3]}x_2 + {p[4]})\exp[-(|({p[5]}(x_1+{p[6]})^2 +$ \\\\ ${p[7]}(x_2+{p[8]})^2 + {p[9]}|)] + {p[10]}$"

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
