import numpy as np

# typing imports
import numpy.typing as npt
from typing import Tuple, Any
number = int | float | complex


class Mixin():

    def get_value(self, inp: npt.NDArray) -> npt.NDArray:
        if inp.shape[1] != self.D:
            raise ValueError(f"array should have the shape [n, {self.D}]")

        return self._get_value(inp)

    def _get_value(self, inp: npt.NDArray) -> npt.NDArray:

        raise NotImplementedError("This function is mandatory for this class")

    def sample_benchmark(self, sample_size: int = 1_000_000, edges=False, corners=False) -> npt.NDArray:
        """Take a sample of the benchmark and return the results

        Args:
            sample_size (int, optional): number of samples to take. 
                                         Defaults to 1_000_000.

        Returns:
            npt.NDArray: Sampled points with values as the final column.
        """

        if isinstance(self.sample, np.ndarray):
            return self.sample

        self.sample = np.random.uniform(
            low=self.lower_bounds, high=self.upper_bounds,
            size=(sample_size, self.D))
        self.sample = np.column_stack(
            (self.sample, self.get_value(self.sample)))

        if edges:
            edges = self.get_edges(int(sample_size/100))

            self.sample = np.row_stack([self.sample, edges])

        if corners:
            corners = self.get_corners()

            self.sample = np.row_stack([self.sample, corners])

        return self.sample

    def get_corners(self) -> npt.NDArray:
        r = np.row_stack([self.lower_bounds, self.upper_bounds])
        corners = np.array(np.meshgrid(*np.hsplit(r, self.D))
                           ).T.reshape(-1, self.D)
        corners = np.column_stack([corners, self.get_value(corners)])

        return corners

    def get_edges(self, sample_size: int) -> npt.NDArray:
        s_1 = np.column_stack([np.random.uniform(
            self.lower_bounds[0], self.upper_bounds[0], sample_size), np.ones(sample_size) * self.lower_bounds[1]])
        s_2 = np.column_stack([np.random.uniform(
            self.lower_bounds[0], self.upper_bounds[0], sample_size), np.ones(sample_size) * self.upper_bounds[1]])

        s_3 = np.column_stack([np.ones(sample_size) * self.lower_bounds[0], np.random.uniform(
            self.lower_bounds[1], self.upper_bounds[1], sample_size)])
        s_4 = np.column_stack([np.ones(sample_size) * self.upper_bounds[0], np.random.uniform(
            self.lower_bounds[1], self.upper_bounds[1], sample_size)])

        edges = np.row_stack([s_1, s_2, s_3, s_4])

        return np.column_stack([edges, self.get_value(edges)])
