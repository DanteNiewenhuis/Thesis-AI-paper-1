# %%

import re
import numpy as np
import numpy.typing as npt
import scipy.spatial

from rich import print
from typing import Tuple, Any

from project.Algorithms import PPA
import project.Benchmarks_base._plotting as _plotting
import project.Benchmarks_base._value as _value
import project.Benchmarks_base._extremes_sample as _extremes_sample

number = int | float | complex

# %%


class Benchmark(_plotting.Mixin, _value.Mixin, _extremes_sample.Mixin):
    def __init__(self, params: list[number], lower: npt.NDArray, upper: npt.NDArray, D: number = 2):
        self.params = params

        # Initialize bounds
        self.D = D
        self.lower_bounds = lower
        self.upper_bounds = upper

        # Check for correct bounds size
        if len(self.lower_bounds) != self.D:
            raise ValueError(
                "The lower bounds should have the same length as the dimensions")

        # Determine ranges
        self.range = upper - lower
        self.max_dis = scipy.spatial.distance.cdist(
            lower.reshape(1, -1), upper.reshape(1, -1))[0]

        # initialize variables
        self.zero_points = None
        self.sample = None

    def to_latex(self, r=5, line_breaks=False, parameterized=False):
        res: str = self._to_latex(r, line_breaks, parameterized)

        res = res.replace("+ -", "-")
        res = res.replace("+-", "-")
        res = res.replace("- -", "+")
        res = res.replace("--", "+")

        res = re.sub("\|\s*\+\s*", r"|", res)
        res = re.sub("\(\s*\+\s*", r"(", res)
        res = re.sub("^\$\s*\+\s*", r"", res)

        res = re.sub("\s*\+\s*\|", r"|", res)
        res = re.sub("\s*\+\s*\)", r")", res)
        res = re.sub("\s*\+\s*\$\s*$", r"$", res)

        res = re.sub("\s*\-\s*\|", r"|", res)
        res = re.sub("\s*\-\s*\)", r")", res)
        res = re.sub("\s*\-\s*\$\s*$", r"$", res)

        res = re.sub("[^\w](\()(\w\_\d)(\))", r"\g<2>", res)
        res = re.sub("(\.0)([^\d])", r"\g<2>", res)

        if line_breaks:
            res = "\\begin{tabular}[c]{@{}l@{}}" + res + "\end{tabular}"

        return res

    def _to_latex(self):
        raise NotImplementedError

    def get_domains(self, steps: int = 100) -> Tuple[npt.NDArray, npt.NDArray]:
        """Return a linspace for both the x and the y dimension

        Args:
            steps (int, optional): number of steps in the range. 
                                   Defaults to 100.

        Returns:
            Tuple[npt.NDArray, npt.NDArray]: The x_domain and y_domain respectively
        """

        return np.linspace(self.lower_bounds, self.upper_bounds, steps).T
