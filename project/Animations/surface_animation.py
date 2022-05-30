# %%

import numpy as np
from rich import print

import plotly as plt
import plotly.graph_objects as go

# %%

lower_bounds = np.array([-10, -10])
upper_bounds = np.array([10, 10])
steps = 10
marker_style = dict(size=3, color="green")
surface_opacity = 1

x_range = np.linspace(
    lower_bounds[0], upper_bounds[0], steps)
y_range = np.linspace(
    lower_bounds[1], upper_bounds[1], steps)


# %%

def get_surface(steps):
    return np.random.uniform(low=0, high=1, size=(steps, steps))


z_surface = get_surface(steps)

# %%


data = [go.Surface(x=x_range, y=y_range,
                   z=z_surface, opacity=surface_opacity)]

frames = [go.Frame(data=data, name=str(0))]
for i in [x for x in range(5)]:
    z_surface = get_surface(steps)

    frames.append(go.Frame(data=[go.Surface(x=x_range, y=y_range, z=z_surface, opacity=surface_opacity)],
                           name=str(i+1)))

fig = go.Figure(data=data, frames=frames)

fig.update_layout(title='data HEATPILES', autosize=True,
                  width=500, height=500, margin=dict(l=65, r=50, b=65, t=90))

################################################################################
# Create Slider
################################################################################


def frame_args(duration):
    return {
        "frame": {"duration": duration, 'redraw': True},
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
                "label": f.name,
                "method": "animate",
            }
            for f in fig.frames
        ],
    }
]

fig.update_layout(sliders=sliders)

fig.show()

# %%

for i in [x/10 for x in range(50)]:
    print(i)

# %%
