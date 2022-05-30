# %%
from __project_path import _

from project.Benchmarks_2d.Branin import Branin
from project.Benchmarks_2d.Easom import Easom
from project.Benchmarks_2d.GoldsteinPrice import GoldsteinPrice
from project.Benchmarks_2d.MartinGaddy import MartinGaddy
from project.Benchmarks_2d.Mishra4 import Mishra4
from project.Benchmarks_2d.SixHump import SixHump

from project.measurements import get_results, plot_runs

import plotly.graph_objects as go
from datetime import datetime

import numpy as np
import pandas as pd
import os

from project.Benchmarks_base.benchmark import Benchmark


def get_annotations(surface, val, x_min, x_max, y_min, y_max, i, num_evolutions):
    return [
        dict(
            bgcolor="lightgray",
            bordercolor="black",
            borderwidth=3,
            borderpad=10,
            xanchor="left",
            yshift=-320, xshift=-100,
            x=x_max, y=y_min, z=np.min(surface),
            text=f'{i+1}/{num_evolutions}', textangle=0, showarrow=False,
            font=dict(
                color="black",
                size=55
            ),
        ),
        dict(
            bgcolor="lightgray",
            bordercolor="black",
            borderwidth=3,
            borderpad=10,
            xanchor="right",
            yshift=-280, xshift=100,
            x=x_min, y=y_max, z=np.min(surface),
            text=f'{val:.5f}', textangle=0, showarrow=False,
            font=dict(
                color="black",
                size=55
            ),
        ),
    ]


def get_titles(surface, x_min, x_max, y_min, y_max, name):
    return [
        dict(
            bgcolor="lightgray",
            bordercolor="black",
            borderwidth=3,
            borderpad=10,
            xanchor="right",
            yanchor="bottom",
            yshift=-325, xshift=80,
            x=x_min, y=y_max, z=np.min(surface),
            text=f'{name}', textangle=0, showarrow=False,
            font=dict(
                color="black",
                size=45,
                family="Balto"
            ),
            align="left",
        )
    ]


def get_df(benchmark: Benchmark, session: str = "latest"):
    if session == "latest":
        sessions = [datetime.strptime(x, '%Y-%m-%d_%H:%M:%S')
                    for x in os.listdir(f"../results/{benchmark.__name__}")]

        session = datetime.strftime(max(sessions), '%Y-%m-%d_%H:%M:%S')

    df = pd.read_csv(
        f"../results/{benchmark.__name__}/{session}/run.csv")

    df['iteration'] = df.index

    return pd.DataFrame([g[1].iloc[0] for g in df.groupby('val')])


def get_surface(benchmark: Benchmark, df: pd.DataFrame, i: int = 0):
    param_columns = [c for c in df.columns if "p_" in c]

    params = df.iloc[i][param_columns].to_numpy()

    b: Benchmark = benchmark(params)

    x_domain, y_domain = b.get_domains()

    surface = b.get_benchmark_surface(x_domain, y_domain)

    return x_domain, y_domain, surface

# %%


layout = dict(autosize=True,

              scene_camera_eye=dict(x=1.2, y=1.45, z=0.7),
              scene_camera_up=dict(x=0, y=0, z=1),
              scene_camera_center=dict(x=0, y=0, z=-0.15),
              width=1000, height=1000, margin=dict(l=0, r=0, t=0, b=0),

              scene=dict(
                  aspectmode="cube",
                  xaxis=dict(
                      tickfont=dict(size=20),
                      title=dict(text="x_1", font=dict(size=30))
                  ),
                  yaxis=dict(
                      tickfont=dict(size=20),
                      title=dict(text="x_2", font=dict(size=30))
                  ),
                  zaxis=dict(
                      tickfont=dict(size=20),
                      title=dict(text="", font=dict(size=30))
                  )

              ))


def plot_surface_MOD(benchmark, i):

    df = get_df(benchmark)
    val = df.iloc[i]['val']
    num_evolutions = len(df)

    x_domain, y_domain, surface = get_surface(benchmark, df, i=i)

    data = [go.Surface(x=x_domain, y=y_domain,
                       z=surface, showscale=False)]
    x_min, x_max = x_domain[0], x_domain[-1]
    y_min, y_max = y_domain[0], y_domain[-1]

    annotations = get_annotations(
        surface, val, x_min, x_max, y_min, y_max, i, num_evolutions)

    fig = go.Figure(data=data)

    # current_layout['scene']['annotations'] = annotations

    if benchmark == Mishra4:
        layout["scene_camera_eye"]["z"] = 1.5

    fig.update_layout(layout)

    fig.write_image(
        f"../results_paper_1/{benchmark.__name__}/frames/{i}.png")


names = {Branin: "<b>Branin</b>", Easom: "<b>Easom</b>", GoldsteinPrice: '<b>Goldstein-<br>Price</b>',
         MartinGaddy: "<b>Martin-Gaddy</b>", Mishra4: "<b>Mishra4</b>", SixHump: "<b>Six-Hump<br>Camel</b>"}


def plot_surface_base(benchmark, i):

    df = get_df(benchmark)
    val = df.iloc[i]['val']
    num_evolutions = len(df)

    x_domain, y_domain, surface = get_surface(benchmark, df, i=i)

    data = [go.Surface(x=x_domain, y=y_domain,
                       z=surface, showscale=False)]
    x_min, x_max = x_domain[0], x_domain[-1]
    y_min, y_max = y_domain[0], y_domain[-1]

    name = names[benchmark]
    annotations = get_titles(
        surface, x_min, x_max, y_min, y_max, name)

    if benchmark == Mishra4:
        annotations[0]["yshift"] = -365
        annotations[0]["xshift"] = 100

    fig = go.Figure(data=data)

    layout['scene']['annotations'] = annotations
    if benchmark == Mishra4:
        layout["scene_camera_eye"]["z"] = 1.5
    else:
        layout["scene_camera_eye"]["z"] = 0.85

    fig.update_layout(layout)
    fig.write_image(
        f"../results_paper_1/base_benchmarks/{benchmark.__name__}.png")


# for benchmark in [Branin, Easom, GoldsteinPrice, MartinGaddy, Mishra4, SixHump]:
#     plot_surface_base(benchmark, 0)

# plot_surface_base(GoldsteinPrice, 0)

# %%

# for benchmark in [Branin, Easom, GoldsteinPrice, MartinGaddy, Mishra4, SixHump]:
for benchmark in [Easom]:
    df = get_df(benchmark)
    l = len(df)

    for i in range(l):
        plot_surface_MOD(benchmark, i)

# %%

df = get_df(Mishra4)
# %%

param_columns = [c for c in df.columns if "p_" in c]
params = df.iloc[-1][param_columns]

# %%

m = Mishra4(params)

# %%

plot_runs(m)
# %%
