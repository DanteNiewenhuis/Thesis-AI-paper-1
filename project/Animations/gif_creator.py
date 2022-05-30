import os
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import gif

from project.Benchmarks_2d.Branin import Branin
from project.Benchmarks_2d.Easom import Easom
from project.Benchmarks_2d.SixHump import SixHump
from project.Benchmarks_2d.MartinGaddy import MartinGaddy
from project.Benchmarks_2d.GoldsteinPrice import GoldsteinPrice

from project.Benchmarks_nd.Schwefel import Schwefel
from project.Benchmarks_nd.Griewank import Griewank


def get_params(benchmark, session):
    df = pd.read_csv(f"results/{benchmark}/{session}/run.csv")
    num_parameters = len(df.columns) - 2

    iteration_ranges = []
    params = []
    values = []

    for val, series in df.groupby("val"):
        iteration_ranges.append([series.index[0], series.index[-1]])
        params.append(series.head(
            1)[[f"p_{i}" for i in range(num_parameters)]].to_numpy()[0])
        values.append(val)

    # options = [{'label': f"Model {i} => used in iterations {x[0]} - {x[1]}",
    #                 'value': i} for i, x in enumerate(iteration_ranges)]

    return params, values


def get_domains(benchmark):
    bench = eval(f"{benchmark}()")

    return bench.get_domains(100)


def get_surfaces(benchmark, params, x_domain, y_domain):
    z_min = 0
    z_max = 0
    z_list = []

    for p in params:
        s = eval(f"{benchmark}({list(p)})")

        z = s.get_benchmark_surface(x_domain, y_domain)

        z_min = np.min(z) if np.min(z) < z_min else z_min
        z_max = np.max(z) if np.max(z) > z_max else z_max

        z_list.append(z)

    z_min *= 1.1 if z_min < 0 else 0.9
    z_max *= 1.1 if z_max > 0 else 0.9

    return z_list, z_min, z_max


def make_gif(benchmark, session):
    x_domain, y_domain = get_domains(benchmark)

    x_min, x_max = x_domain[0], x_domain[-1]
    y_min, y_max = y_domain[0], y_domain[-1]

    params, values = get_params(benchmark, session)

    z_list, z_min, z_max = get_surfaces(benchmark, params, x_domain, y_domain)

    @gif.frame
    def plot(i):
        data = [go.Surface(x=x_domain, y=y_domain,
                           z=z_list[i], showscale=False)]
        fig = go.Figure(data=data)

        fig.update_layout(autosize=True,

                          scene_camera_eye=dict(x=1.3, y=1.3, z=0.85),
                          scene_camera_up=dict(x=0, y=0, z=1),
                          scene_camera_center=dict(x=0, y=0, z=-0.15),
                          width=1000, height=1000, margin=dict(l=0, r=0, t=0, b=0),


                          scene=dict(
                              aspectmode="cube",
                              annotations=[
                                  dict(
                                      xanchor="left", yanchor="bottom",
                                      yshift=70,
                                      x=x_max, y=y_min, z=np.max(z_list[i]),
                                      text=f'Model {i+1}/{len(params)}', textangle=0, showarrow=False,
                                      font=dict(
                                          color="black",
                                          size=55
                                      ),
                                  ),
                                  dict(
                                      xanchor="right", yanchor="bottom",
                                      yshift=70,
                                      x=x_min, y=y_max, z=np.max(z_list[i]),
                                      text=f'MOD: {values[i]:.5f}', textangle=0, showarrow=False,
                                      font=dict(
                                          color="black",
                                          size=55
                                      ),
                                  ),
                              ],
                          ))

        if not os.path.exists(f"results/{benchmark}/{session}/frames"):
            os.mkdir(f"results/{benchmark}/{session}/frames")
            os.mkdir(f"results/{benchmark}/{session}/frames/text")
            os.mkdir(f"results/{benchmark}/{session}/frames/no_text")

        fig.write_image(
            f"results/{benchmark}/{session}/frames/no_text/{i}.png")

        # fig.update_layout(
        #     title=f'Model {i+1}/{len(params)}, value: {values[i]:.5f}',
        #     title_font=dict(
        #         family="Courier New, monospace",
        #         size=50
        #     ), margin=dict(l=0, r=0, t=80, b=0))

        # fig.write_image(
        #     f"results/{benchmark}/{session}/frames/text/{i}.png")
        return fig

    frames = [plot(i) for i in range(len(params))]

    # frames = [plot(i) for i in range(1)]

    # gif.save(
    #     frames, f"results/{benchmark}/{session}/evolution.gif", duration=1000, loop=False)
