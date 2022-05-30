# %%

import re
from project.Benchmarks_base.benchmark import Benchmark
import numpy as np
import numpy.typing as npt
from typing import Annotated, Union, Tuple

number = int | float | complex

# %%

base_params = (1, 1, 5.1/(4*np.pi**2), 0, 5/np.pi, -
               6, 10*(1-1/(8*np.pi)), 1, 0, 10)


class Branin(Benchmark):

    def __init__(self, params: Annotated[Tuple, 10] = base_params,
                 lower: npt.NDArray = np.array([-5, 0]),
                 upper: npt.NDArray = np.array([10, 15]), D=2):
        super(Branin, self).__init__(params, lower, upper)

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

        s_1 = p[0]*(p[1]*x_2 - p[2]*(x_1+p[3])**2 + p[4]*x_1 + p[5])**2
        s_2 = p[6]*np.cos(p[7]*x_1 + p[8])

        return s_1 + s_2 + p[9]

    def _to_latex(self, r=5, line_breaks=True, parameterized=False, double=False):
        p_f = np.round(self.params, r)

        # all parameters that multiply
        p = ["" for _ in p_f]
        for i in [0, 1, 2, 4, 6, 7]:
            p[i] = "" if p_f[i] == 1 else str(p_f[i])

        # all parameters that add
        for i in [3, 5, 8, 9]:
            p[i] = "" if p_f[i] == 0 else str(p_f[i])

        if parameterized:
            p = [f"p_{'{'}{i}{'}'}" for i in range(len(p))]

        res = f"${p[0]}({p[1]}x_2 - {p[2]}(x_1 + {p[3]})^2 + {p[4]}x_1 + {p[5]})^2 + {p[6]}\cos({p[7]}x_1 + {p[8]}) + {p[9]}$"

        if line_breaks:
            res = f"${p[0]}({p[1]}x_2 - {p[2]}(x_1 + {p[3]})^2 + {p[4]}x_1 + {p[5]})^2 +$ \\\\ ${p[6]}\cos({p[7]}x_1 + {p[8]}) + {p[9]}$"

        return res

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
