import plotly.io as pio
import pandas as pd
import plotly.graph_objects as go
import numpy as np


def sphere(x, i):
    power = 2 + i/10
    return np.sum(np.abs(x) * power, axis=1)


steps = 100

x_range = np.linspace(-10, 10, steps)
y_range = np.linspace(-10, 10, steps)

z = np.ones((steps))
z_data_list = [z]
for frame in range(10):
    new_z = np.copy(z)
    new_z += np.random.uniform(low=0, high=0.1, size=(steps))
    z_data_list.append(new_z)

    z = new_z

fig = go.Figure(
    data=[go.Scatter3d(x=x_range, y=y_range,
                       z=z_data_list[0], mode="markers")],
    layout=go.Layout(updatemenus=[dict(type="buttons", buttons=[
                     dict(label="Play", method="animate", args=[None])])]),
    frames=[go.Frame(data=[go.Scatter3d(x=x_range, y=y_range, z=k, mode="markers")], name=str(i))
            for i, k in enumerate(z_data_list)]
)

fig["layout"]["xaxis"] = {"range": [-10, 10]}
fig["layout"]["yaxis"] = {"range": [-10, 10]}

# fig.update_traces(contours_z=dict(show=True, usecolormap=True,
#                   highlightcolor="tomato", project_z=True), colorscale='portland')

fig.update_layout(title='data HEATPILES', autosize=False,
                  width=650, height=500, margin=dict(l=0, r=0, b=0, t=0))


def frame_args(duration):
    return {
        "frame": {"duration": duration},
        "mode": "immediate",
        "fromcurrent": True,
        "transition": {"duration": duration, "easing": "linear"},
    }


sliders = [
    {
        "pad": {"b": 10, "t": 60},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [
            {
                "args": [[f.name], frame_args(0)],
                "label": str(k),
                "method": "animate",
            }
            for k, f in enumerate(fig.frames)
        ],
    }
]

fig.update_layout(sliders=sliders)

fig.show()
