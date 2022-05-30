# %%

import re
from project.Benchmarks_base.benchmark import Benchmark
import numpy as np
import numpy.typing as npt
from typing import Annotated, Tuple

number = int | float | complex

# %%

base_params = (1, 1, 1, 1, 1, 1,
               19, 14, 14, 3, 0, 3, 0, 6,
               30, 1, 2, 3, 0,
               18, 32, 48, 27, 0, 12, 0, 36,
               0)


class GoldsteinPrice(Benchmark):

    def __init__(self, params: Annotated[Tuple[number], 28] = base_params,
                 lower: npt.NDArray = np.array([-2, -2]),
                 upper: npt.NDArray = np.array([2, 2]), D=2):
        super(GoldsteinPrice, self).__init__(params, lower, upper)

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

        s_1 = p[0]*(p[1] + p[2]*(p[3]*x_1 + p[4]*x_2 + p[5])**2 *
                    (p[6] - p[7]*x_1 - p[8]*x_2 + p[9]*(x_1+p[10])**2 + p[11]*(x_2+p[12])**2 + p[13]*x_1*x_2))
        s_2 = (p[14] + p[15]*(p[16]*x_1 - p[17]*x_2 + p[18])**2 *
               (p[19] - p[20]*x_1 + p[21]*x_2 + p[22]*(x_2+p[23])**2 + p[24]*(x_1+p[25])**2 - p[26]*x_1*x_2))
        return s_1 * s_2 + p[27]

    def _to_latex(self, r=5, line_breaks=True, parameterized=False, double=False):
        p_f = np.round(self.params, r)

        # all parameters that multiply
        p = ["XX" for _ in p_f]
        for i in [0, 2, 3, 4, 7, 8, 9, 11, 13, 15, 16, 17, 20, 21, 22, 24, 26]:
            p[i] = "" if p_f[i] == 1 else str(p_f[i])

        # all parameters that add
        for i in [1, 5, 6, 10, 12, 14, 18, 19, 23, 25, 27]:
            p[i] = "" if p_f[i] == 0 else str(p_f[i])

        if parameterized:
            p = [f"p_{'{'}{i}{'}'}" for i in range(len(p))]

        res = f"$[{p[0]}({p[1]} + {p[2]}({p[3]}x_1 + {p[4]}x_2 + {p[5]})^2 ({p[6]} - {p[7]}x_1 - {p[8]}x_2 + p_9(x_1+{p[10]})^2 + {p[11]}(x_2+{p[12]})^2 + {p[13]}x_1x_2)] \\times [{p[14]} + {p[15]}({p[16]}x_1 - {p[17]}x_2 + {p[18]})^2 ({p[19]} - {p[20]}x_1 + {p[21]}x_2 + {p[22]}(x_2+{p[23]})^2 + {p[24]}(x_1+{p[25]})^2  - {p[26]}x_1x_2)] + {p[27]}$"

        if line_breaks:
            res = f"$[{p[0]}({p[1]} + {p[2]}({p[3]}x_1 + {p[4]}x_2 + {p[5]})^2$\\\\ $ ({p[6]} - {p[7]}x_1 - {p[8]}x_2 + p_9(x_1+{p[10]})^2 + {p[11]}(x_2+{p[12]})^2 + {p[13]}x_1x_2)] \\times$\\\\ $ [{p[14]} + {p[15]}({p[16]}x_1 - {p[17]}x_2 + {p[18]})^2$\\\\ $ ({p[19]} - {p[20]}x_1 + {p[21]}x_2 + {p[22]}(x_2+{p[23]})^2 + {p[24]}(x_1+{p[25]})^2  - {p[26]}x_1x_2)] + {p[27]}$"

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
