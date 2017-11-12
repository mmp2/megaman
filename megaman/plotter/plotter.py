# Author: Yu-Chia Chen <yuchaz@uw.edu>
# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE

import numpy as np
from .utils import *
from .utils import _check_backend
from .scatter_3d import scatter_plot3d_plotly, scatter_plot3d_matplotlib
from .covar_plotter3 import covar_plotter3d_plotly, covar_plotter3d_matplotlib

@_check_backend('plotly')
def plot_with_plotly( embedding, rieman_metric, nstd=2,
                      color_by_ratio=True, if_ellipse=False ):
    from plotly.offline import iplot
    import plotly.graph_objs as go
    sigma_norms = get_top_two_sigma_norm(rieman_metric, color_by_ratio)
    colors, colorscale = generate_colors_and_colorscale('gist_rainbow',
                                                        sigma_norms)
    scatter_pt = scatter_plot3d_plotly(embedding, coloring=sigma_norms,
                                       colorscale=colorscale)
    index = generate_grid(embedding.shape[0])

    if if_ellipse:
        ellipses_pt = covar_plotter3d_plotly(embedding,
                                             rieman_metric, index, colors)
        scatter_pt = ellipses_pt + scatter_pt

    layout = plotly_layout(embedding)
    fig = go.Figure(data=scatter_pt,layout=layout)
    iplot(fig,filename='scatter-3d-plotly')

def plot_embedding_with_plotly(trace_var,idx,if_ellipse=False):
    plot_with_plotly(trace_var.Y[idx],trace_var.H[idx]/30,if_ellipse=if_ellipse)

@_check_backend('matplotlib')
def plot_with_matplotlib(embedding, rieman_metric, nstd=2,
                         color_by_ratio=True, if_ellipse=False):
    import matplotlib.pyplot as plt
    sigma_norms = get_top_two_sigma_norm(rieman_metric, color_by_ratio)
    colors, _ncor = get_colors_array('gist_rainbow', sigma_norms, base255=False)
    fig,ax = scatter_plot3d_matplotlib(embedding, sigma_norms)

    index = generate_grid(embedding.shape[0])
    if if_ellipse:
        ax = covar_plotter3d_matplotlib(embedding, rieman_metric,
                                        index, ax, colors)
    plt.show()

def plot_embedding_with_matplotlib(trace_var,idx,if_ellipse=False):
    plot_with_matplotlib(trace_var.Y[idx],trace_var.H[idx]/30,if_ellipse=if_ellipse)
