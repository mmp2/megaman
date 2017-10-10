import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d, Axes3D
import plotly.graph_objs as go
import numpy as np

def scatter_plot3d_matplotlib(embedding, coloring=None, fig=None):
    if fig is None:
        fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    s = [2 for i in range(embedding.shape[0])]
    x,y,z = embedding[:,:3].T
    coloring = x if coloring is None else coloring
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

def scatter_plot3d_plotly(embedding, coloring=None, colorscale='Rainbow'):
    x,y,z = embedding[:,:3].T
    coloring = x if coloring is None else coloring
    scatter_plot = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=2,
            opacity=0.8,
            color=coloring,
            colorscale=colorscale,
            showscale=True,
        ),
        name='Embedding'
    )
    return [scatter_plot]
