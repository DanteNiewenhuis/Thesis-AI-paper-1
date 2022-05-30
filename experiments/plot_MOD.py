# %%

from plotly.subplots import make_subplots
from __project_path import _

from project.Benchmarks_2d.Branin import Branin
from project.Benchmarks_2d.Easom import Easom
from project.Benchmarks_2d.GoldsteinPrice import GoldsteinPrice
from project.Benchmarks_2d.MartinGaddy import MartinGaddy
from project.Benchmarks_2d.Mishra4 import Mishra4
from project.Benchmarks_2d.SixHump import SixHump

from project.measurements import get_results, plot_runs

import plotly.graph_objects as go
import plotly.express as px

from datetime import datetime

import numpy as np
import pandas as pd
import os

from project.Benchmarks_base.benchmark import Benchmark


def get_df(benchmark: Benchmark, session: str = "latest"):
    if session == "latest":
        sessions = [datetime.strptime(x, '%Y-%m-%d_%H:%M:%S')
                    for x in os.listdir(f"../results/{benchmark.__name__}")]

        session = datetime.strftime(max(sessions), '%Y-%m-%d_%H:%M:%S')

    df = pd.read_csv(
        f"../results/{benchmark.__name__}/{session}/run.csv")

    return df

    df['iteration'] = df.index

    return pd.DataFrame([g[1].iloc[0] for g in df.groupby('val')])


def plot_MOD(benchmark: Benchmark, df: pd.DataFrame, last_session: str):

    fig = px.line(df, y="val")
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="Iterations",
                      yaxis_title="Difficulty",
                      width=1000, height=600,
                      xaxis=dict(titlefont=dict(size=35)),
                      yaxis=dict(titlefont=dict(size=35)))

    fig.update_traces(line=dict(width=5))

    fig.write_image(
        f"results/{benchmark.__name__}/{last_session}/training_{benchmark.__name__}.png")

# %%


benchmarks = [Branin, Easom, GoldsteinPrice, MartinGaddy, Mishra4, SixHump]

# subplot_titles=([b.__name__ for b in benchmarks])
fig = make_subplots(rows=2, cols=3, shared_xaxes=True, shared_yaxes=False,
                    horizontal_spacing=0.025, vertical_spacing=0.02,
                    x_title="Iterations", y_title="MOD")

placement = [[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3]]

for i in range(6):
    row = placement[i][0]
    col = placement[i][1]
    df = get_df(benchmarks[i])

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['val'],
            line=dict(color="blue", width=2),
            mode="lines",

        ),
        row=row, col=col
    )
    fig.add_annotation(text=f"{benchmarks[i].__name__}", x=2000, y=0.5,
                       bgcolor="lightgray", xanchor="right", font=dict(size=15),
                       yshift=-165*(row-1), xref=f'x{col}',  showarrow=False)

fig.update_layout(height=400, width=1000,
                  showlegend=False, margin=dict(l=60, r=0, t=20, b=60))


# fig.update_yaxes(ticklabelposition="inside top")

# names = {'Plot 1':'2016', 'Plot 2':'2017', 'Plot 3':'2018', 'Plot 4':'2019'}
# fig.for_each_annotation(lambda a: a.update(text = a.text + ': ' + names[a.text]))

fig.write_image(
    f"../results_paper_1/MODovertime.png")
fig.show()


# %%

fig = go.Figure()
fig.add_trace(
    go.Scatter(x=df.index, y=df['val'])
)
fig.update_yaxes(tickvals=[round(x, 2)
                 for x in np.linspace(0, df['val'].iloc[-1], 5)])
fig.show()
# %%

print(df['val'].iloc[0], df['val'].iloc[-1])


# %%

np.linspace(df['val'].iloc[0], df['val'].iloc[-1], 5)

# %%
