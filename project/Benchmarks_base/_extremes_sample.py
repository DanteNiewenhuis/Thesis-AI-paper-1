import scipy.spatial
import numpy as np

# typing imports
import numpy.typing as npt
from typing import Tuple, Any
number = int | float | complex


class Mixin():
    def get_extremes(self, sample_size: int = 1_000_000) -> Tuple[number, number, number, number, number, number]:
        """Get the global minimum and global maximum of the benchmark

        Args:
            sample_size (int, optional): Number of samples to take 

        Returns:
            Tuple[number, number, number, number, number, number]: _description_
        """

        sample = self.sample_benchmark(sample_size)

        a_min = np.argmin(sample[:, -1])
        a_max = np.argmax(sample[:, -1])

        return sample[a_min, :], sample[a_max, :]

    def get_minima(self, sample_size: int = 1_000_000):
        s = self.sample_benchmark(sample_size=sample_size)

        minimum = np.min(s[:, -1])
        s_max = np.sort(s[:, -1])[int(len(s)/10)]

        val_dis = (s_max - np.min(s[:, 2]))/1_000
        args = np.where((s[:, -1] - minimum) < val_dis)[0]
        s_min = s[args]

        s_dis = scipy.spatial.distance.cdist(s_min[:, :-1], s_min[:, :-1])

        final_minima = []
        seen = []
        for i in range(len(s_min)):
            if i in seen:
                continue

            close = list(np.where(s_dis[i] < (self.max_dis/50))[0])
            close = [x for x in close if x not in seen]
            seen += close

            v = s_min[close]
            idx = np.argmin(v[:, -1])

            final_minima.append(v[idx])

        return np.array(final_minima)

    def get_val_range(self, sample_size: int = 1_000_000) -> number:
        """Get a value range of the benchmark using a sample

        Args:
            sample_size ([int], optional): Number of points to sample. 
                                           Defaults to 10_000.

        Returns:
            number: The value range of the benchmark
        """
        z = self.sample_benchmark(sample_size)[:, -1]
        return np.max(z) - np.min(z)

    def _search_grid(self, middle, grid_size, steps=10, selection_size=5):
        domains, step_size = np.linspace(
            middle-(grid_size/2), middle+(grid_size/2), steps, retstep=True)

        points = np.array(np.meshgrid(
            *np.hsplit(domains, domains.shape[1]))).T.reshape(-1, domains.shape[1])

        values = self.get_value(points)

        minima = np.argsort(values)[:selection_size]

        return points[minima], step_size

    def _check_seen(self, seen, point):
        seen = np.array(seen)
        point = np.array(point)

        for p in seen:
            if (p == point).all():
                return True

        return False

    def _find_points(self, middle, grid_size, res, depth=3, steps=10, selection_size=5):
        if depth == 0:
            return res

        # Skip point if already processed
        if self._check_seen(res, middle):
            return res

        res.append(middle)
        points, step_size = self._search_grid(
            middle, grid_size, steps=steps, selection_size=selection_size)

        for p in points:
            self._find_points(p, step_size, res, depth-1)

        return res

    def get_minimum(self, steps=10, depth=3, selection_size=5):
        middle = self.lower_bounds + self.range/2

        res = np.array(self._find_points(middle, self.range, [], depth=depth,
                                         steps=steps, selection_size=selection_size))

        values = self.get_value(res)

        i_min = np.argmin(values)

        return res[i_min, :], values[i_min]
