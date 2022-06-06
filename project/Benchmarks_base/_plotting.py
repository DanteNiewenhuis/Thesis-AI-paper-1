import numpy as np

# plot imports
import plotly.graph_objects as go
import plotly as plt

# typing imports
import numpy.typing as npt
from typing import Tuple, Any
number = int | float | complex


class Mixin():

    def check_dimensions(self):
        """Checks if benchmark is 2 dimensional, raises error otherwise.

        Raises:
            ValueError: Benchmark needs to be 2 dimensional
        """

        if self.D != 2:
            raise ValueError(
                "Plotting can only be done on 2 dimensional benchmarks.")

    def get_benchmark_surface(self, x_domain, y_domain, steps: int = 100) -> npt.NDArray:
        """Get point values of the benchmark that can be used to create a surface

        Args:
            x_domain ([type]): list of number to sample from the x_axis
            y_domain ([type]): list of number to sample from the y_axis
            steps (int, optional): number of steps taken in each dimension. 
                                   Defaults to 100.

        Returns:
            npt.NDArray: A matrix of value on the given ranges
        """

        self.check_dimensions()

        steps = x_domain.shape[0]

        x = np.repeat(x_domain, steps)
        y = np.tile(y_domain, steps)

        points = np.column_stack((x, y))

        return np.transpose(self.get_value(points).reshape((steps, steps)))

    def get_benchmark_data(self, steps: int = 100) -> list:

        self.check_dimensions()

        domains = self.get_domains(steps)

        x_domain, y_domain = domains[0], domains[1]
        z = self.get_benchmark_surface(x_domain, y_domain, steps)

        return [go.Surface(z=z, x=x_domain, y=y_domain)]

    def get_points_data(self, pops: list[list[number]], steps: int = 100,
                        marker_styles=[], names=[]) -> list:

        self.check_dimensions()

        domains = self.get_domains(steps)
        x_domain, y_domain = domains[0], domains[1]

        z_surface = self.get_benchmark_surface(x_domain, y_domain, steps)

        data = [go.Surface(z=z_surface, x=x_domain, y=y_domain, opacity=0.8)]

        for i, pop in enumerate(pops):
            markers_x = pop[:, 0]
            markers_y = pop[:, 1]
            markers_z = self.get_value(pop)

            ms = marker_styles[i] if len(
                marker_styles) > i else dict(size=3, color="green", opacity=0.9)
            name = names[i] if len(names) > i else ""

            data.append(go.Scatter3d(z=markers_z, x=markers_x, y=markers_y,
                                     mode="markers", marker=ms, name=name))

        return data

    def get_layout(self, title="") -> Any:
        return go.Layout(title=title, autosize=True,
                         width=500, height=500,
                         margin=dict(l=65, r=50, b=65, t=90),
                         legend=dict(
                             yanchor="top",
                             y=0.99,
                             xanchor="right",
                             x=1
                         ),
                         scene={
                             "aspectmode": "cube"
                         })

    def plot_benchmark(self, steps: int = 100, title: str = "", save=False, name=None):
        self.check_dimensions()

        fig = go.Figure(
            data=self.get_benchmark_data(steps),
            layout=self.get_layout(title))

        if save:
            with open(f"results/{name}.php", "w") as wf:
                wf.write(plt.offline.plot(fig, include_plotlyjs=False,
                                          output_type='div'))
        fig.show()

    def plot_points(self, pop: list[number], steps: int = 100, title: str = "") -> Any:
        self.check_dimensions()

        fig = go.Figure(data=self.get_points_data(
            [pop], steps), layout=self.get_layout(title))

        return fig

    def plot_zero_points(self, show=False) -> Any:
        self.check_dimensions()

        fig = self.plot_points(self.get_zero_points())

        if show:
            fig.show()
            return

        return fig

    def plot_extremes(self):
        self.check_dimensions()

        x_min, y_min, z_min, x_max, y_max, z_max = self.get_extremes()
        data = [np.array([[x_min, y_min]]), np.array([[x_max, y_max]])]
        styles = [dict(size=6, color="green"), dict(size=6, color="blue")]
        names = ["Min", "Max"]
        points = self.get_points_data(data, marker_styles=styles, names=names)

        fig = go.Figure(data=points, layout=self.get_layout())
        fig.show()

    def plot_extremes_sample(self):
        self.check_dimensions()

        x_min, y_min, z_min, x_max, y_max, z_max = self.get_extremes()
        data = [np.array([[x_min, y_min]]), np.array([[x_max, y_max]])]
        styles = [dict(size=6, color="green"), dict(size=6, color="blue")]
        names = ["Min", "Max"]
        points = self.get_points_data(data, marker_styles=styles, names=names)

        fig = go.Figure(data=points, layout=self.get_layout())
        fig.show()

    def plot_population(self, steps: int = 100, title: str = ""):
        self.check_dimensions()

        self.plot_points(self.current_population, steps, title)
