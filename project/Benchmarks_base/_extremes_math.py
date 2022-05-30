from scipy.optimize import fsolve
from multiprocessing import Pool
import numpy as np

# typing imports
import numpy.typing as npt
from typing import Tuple, Any
number = int | float | complex


class Mixin():
    def get_zero_points(self, steps: int = 100) -> npt.NDArray:
        """Gather the coords of all minima and maxima. 

        Args:
            steps (int, optional): number of steps to take in each dimension. 
                                   Defaults to 100.

        Returns:
            npt.NDArray: coords of the minima and maxima
        """
        if self.zero_points is not None:
            return self.zero_points

        # Initial list consisting of the four corners
        zero_points = [(self.lower_bounds[0], self.lower_bounds[1]),
                       (self.upper_bounds[0], self.lower_bounds[1]),
                       (self.lower_bounds[0], self.upper_bounds[1]),
                       (self.upper_bounds[0], self.upper_bounds[1])]

        x_domain, y_domain = self.get_domains(steps)
        y_domain = np.linspace(
            self.lower_bounds[1], self.upper_bounds[1], steps)

        x_domain_1 = [x for i, x in enumerate(x_domain) if i % 2 == 0]
        x_domain_2 = [x for i, x in enumerate(x_domain) if i % 2 == 1]

        y_domain_1 = [y for i, y in enumerate(y_domain) if i % 2 == 0]
        y_domain_2 = [y for i, y in enumerate(y_domain) if i % 2 == 1]

        x_1 = np.repeat(x_domain_1, int(steps/2))
        y_1 = np.tile(y_domain_1, int(steps/2))
        points_1 = np.column_stack((x_1, y_1))

        x_2 = np.repeat(x_domain_2, int(steps/2))
        y_2 = np.tile(y_domain_2, int(steps/2))
        points_2 = np.column_stack((x_2, y_2))

        points = np.row_stack((points_1, points_2))

        x = np.repeat(x_domain, int(steps))
        y = np.tile(y_domain, int(steps))
        points = np.column_stack((x, y))
        # Get all zero_points within the bounds
        with Pool(10) as p:
            res = np.array(p.map(self.solve_deriv, points))

        zero_points += list(set([tuple(x) for x in res]))

        # Get all zero_points on y=lower and y=upper
        for x in x_domain:
            r_low, d, ier, mesg = fsolve(
                self.deriv_x_low, x, full_output=True)
            if (ier == 1):
                zero_points.append(
                    tuple([r_low[0].round(3), self.lower_bounds[1]]))

            r_high, d, ier, mesg = fsolve(
                self.deriv_x_high, x, full_output=True)
            if (ier == 1):
                zero_points.append(
                    tuple([r_high[0].round(3), self.upper_bounds[1]]))

        # Get all zero_points on x=lower and x=upper
        for y in y_domain:
            r_low, d, ier, mesg = fsolve(
                self.deriv_y_low, y, full_output=True)
            if (ier == 1):
                zero_points.append(
                    tuple([self.lower_bounds[0], r_low[0].round(3)]))

            r_high, d, ier, mesg = fsolve(
                self.deriv_y_high, y, full_output=True)
            if (ier == 1):
                zero_points.append(
                    tuple([self.upper_bounds[0], r_high[0].round(3)]))

        # Remove all points that are duplicate, or are outside the bounds
        zero_points = np.array(list(set([x for x in zero_points if (
            x[0] >= self.lower_bounds[0]) and (x[0] <= self.upper_bounds[0]) and
            (x[1] >= self.lower_bounds[1]) and (x[1] <= self.upper_bounds[1])])))

        self.zero_points = zero_points
        return zero_points

    def get_val_range(self) -> number:
        _, _, z_min, _, _, z_max = self.get_extremes()

        return (z_max - z_min)

    def get_extremes(self, steps: int = 100) -> Tuple[number, number, number, number, number, number]:
        """Get the global minimum and global maximum of the benchmark

        Args:
            steps (int, optional): number of steps to take when gathering points. 
                                   Defaults to 100.

        Returns:
            Tuple[number, number, number, number, number, number]:      
                x_min, y_min, z_min, x_max, y_max, z_max 
        """
        zero_points = self.get_zero_points(steps)

        z = self.get_value(zero_points)

        i_min = np.argmin(z)
        i_max = np.argmax(z)

        return zero_points[i_min, 0], zero_points[i_min, 1], z[i_min], \
            zero_points[i_max, 0], zero_points[i_max, 1], z[i_max]

    ############################################################################
    # Deriv functions
    ############################################################################

    def solve_deriv(self, p):
        return fsolve(self.deriv, p).round(3)

    def deriv(self, inp: Tuple[number, number]) -> Tuple[number, number]:
        """Return the derivative of the functions in a point.

        Args:
            inp (Tuple[number, number]): x, y

        Returns:
            Tuple[number, number]: derivative towards x and y
        """
        return [self.deriv_x(inp), self.deriv_y(inp)]

    def deriv_x(self, inp: list[number, number]) -> number:
        raise NotImplementedError("This function is mandatory for this class")

    def deriv_y(self, inp: list[number, number]) -> number:
        raise NotImplementedError("This function is mandatory for this class")

    def deriv_x_low(self, x: number) -> number:
        """The derivative towards x, when y is at the lower bound.

        Args:
            x (number)

        Returns:
            number: derivative towards x
        """
        return self.deriv_x([x, self.lower_bounds[1]])

    def deriv_x_high(self, x: number) -> number:
        """The derivative towards x, when y is at the upper bound.

        Args:
            x (number)

        Returns:
            number: derivative towards x
        """
        return self.deriv_x([x, self.upper_bounds[1]])

    def deriv_y_low(self, y: number) -> number:
        """The derivative towards y, when x is at the lower bound.

        Args:
            y (number)

        Returns:
            number: derivative towards y
        """
        return self.deriv_y([self.lower_bounds[0], y])

    def deriv_y_high(self, y: number) -> number:
        """The derivative towards y, when x is at the upper bound.

        Args:
            y (number)

        Returns:
            number: derivative towards y
        """
        return self.deriv_y([self.upper_bounds[0], y])
