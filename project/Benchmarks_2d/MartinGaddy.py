# %%

import re
from project.Benchmarks_base.benchmark import Benchmark
import numpy as np
import numpy.typing as npt
from typing import Annotated, Tuple

number = int | float | complex

# %%
base_params = (1, 1, 1, 0,
               1, 1/3, 1/3, 10/3, 0)


class MartinGaddy(Benchmark):

    def __init__(self, params: Annotated[Tuple, 9] = base_params,
                 lower: npt.NDArray = np.array([-20, -20]),
                 upper: npt.NDArray = np.array([20, 20]), D=2):
        super(MartinGaddy, self).__init__(params, lower, upper)

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

        s_1 = p[0] * (p[1]*x_1 - p[2]*x_2 + p[3])**2
        s_2 = p[4] * (p[5]*x_1 + p[6]*x_2 - p[7])**2

        return s_1 + s_2 + p[8]

    def _to_latex(self, r=5, line_breaks=True, parameterized=False, double=False):
        p_f = np.round(self.params, r)

        # all parameters that multiply
        p = ["" for _ in p_f]
        for i in [0, 1, 2, 4, 5, 6]:
            p[i] = "" if p_f[i] == 1 else str(p_f[i])

        # all parameters that add
        for i in [3, 7, 8]:
            p[i] = "" if p_f[i] == 0 else str(p_f[i])

        if parameterized:
            p = [f"p_{'{'}{i}{'}'}" for i in range(len(p))]

        res = f"${p[0]}({p[1]}x_1 - {p[2]}x_2 + {p[3]})^2 + {p[4]}({p[5]}x_1 + {p[6]}x_2 - {p[7]})^2 + {p[8]}$"

        if line_breaks:
            res = f"${p[0]}({p[1]}x_1 - {p[2]}x_2 + {p[3]})^2 + {p[4]}({p[5]}x_1 + {p[6]}x_2 - {p[7]})^2 + {p[8]}$"
            # TODO
            # res = "\\begin{tabular}[c]{@{}l@{}}$"
            # res += f"{p[0]}({p[1]}y - {p[2]}(x + {p[3]})^2 + {p[4]}x + {p[5]})^2 +$ \\\\ ${p[6]}\cos({p[7]}x + {p[8]}) + {p[9]}$"
            # res += "\end{tabular}"

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

# %%
