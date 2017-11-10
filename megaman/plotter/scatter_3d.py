# Author: Yu-Chia Chen <yuchaz@uw.edu>
# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE

import numpy as np
from .utils import _check_backend

@_check_backend('matplotlib')
def scatter_plot3d_matplotlib(embedding, coloring=None, fig=None):
    from mpl_toolkits.mplot3d import art3d, Axes3D
    if fig is None:
        import matplotlib.pyplot as plt
        fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    s = [2 for i in range(embedding.shape[0])]
    x,y,z = embedding[:,:3].T

    if isinstance(coloring, str) and coloring.lower() in 'xyz':
        color_idx = 'xyz'.find(coloring)
        coloring = embedding[:,color_idx].flatten()

    if coloring is None:
        ax.scatter(x,y,z,s=s)
    else:
        sc = ax.scatter(x,y,z,c=coloring,cmap='gist_rainbow',s=s)
        fig.colorbar(sc)

    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0

    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    return fig, ax

@_check_backend('plotly')
def scatter_plot3d_plotly(embedding, coloring=None, colorscale='Rainbow', marker=None):
    import plotly.graph_objs as go
    x,y,z = embedding[:,:3].T
    if isinstance(coloring, str) and coloring.lower() in 'xyz':
        color_idx = 'xyz'.find(coloring)
        coloring = embedding[:,color_idx].flatten()
    scatter_plot = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=2,
            opacity=0.8,
        ),
        name='Embedding'
    )
    if coloring is not None:
        scatter_plot['marker'].update(dict(
            color=coloring,
            colorscale=colorscale,
            showscale=True,
        ))
    elif marker is not None:
        scatter_plot['marker'].update(marker)

    return [scatter_plot]
