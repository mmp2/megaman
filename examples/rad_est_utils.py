import numpy as np
import scipy
import scipy.linalg
import os
import plotly.graph_objs as go

try:
    from tqdm import *
    tqdm_installed = True
except ImportError as e:
    tqdm_installed = False
    print('tqdm not installed, will not show the progress bar')

def find_neighbors(idx, dist):
    nbr = dist[idx, :].nonzero()[1]
    if idx not in nbr:
        return np.append(nbr, idx)
    else:
        return nbr


def find_local_singular_values(data, idx, dist,  dim=15):
    nbr = find_neighbors(idx, dist)
    if nbr.shape[0] == 1:
        return np.zeros(dim)
    else:
        local_pca_data = data[nbr, :]
        local_center = np.mean(local_pca_data, axis=0)
        local_pca_data -= local_center[None, :]

        sing = scipy.linalg.svd(local_pca_data, compute_uv=False)
        sing_return = sing[:dim]
        return np.pad(sing_return, (0, dim - sing_return.shape[0]), 'constant')


def find_all_singular_values(data, rad, dist):
    dist_copy = dist.copy()
    dist_copy[dist_copy > rad] = 0.0
    dist_copy.eliminate_zeros()
    dim = data.shape[1]
    singular_list = np.array([find_local_singular_values(data, idx, dist_copy, dim)
                              for idx in range(data.shape[0])])
    return singular_list


def find_mean_singular_values(data, rad, dist):
    singular_list = find_all_singular_values(data, rad, dist)
    return np.mean(singular_list, axis=0)


def find_argmax_dimension(data, dist, optimal_rad):
    singular_list = find_all_singular_values(data, optimal_rad, dist)
    singular_gap = np.hstack(
        (-1 * np.diff(singular_list, axis=1), singular_list[:, -1, None]))
    return np.argmax(singular_gap, axis=1) + 1


def ordinal (n):
    return "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])


def estimate_dimension(data, dist, rad_search_space=None):
    if rad_search_space is None:
        rad_search_space = np.logspace(np.log10(1e-1), np.log10(5), 50)

    rad_iterator = rad_search_space if not tqdm_installed else tqdm(
        rad_search_space)
    sgv = np.array([find_mean_singular_values(data, rad, dist)
                    for rad in rad_iterator])

    return rad_search_space, sgv


def plot_singular_values_versus_radius(singular_values, rad_search_space, start_idx, end_idx):
    all_trace = []
    singular_gap = -np.diff(singular_values,axis=1)
    for idx, sing in enumerate(singular_values.T):
        singular_line = go.Scatter(
            x=rad_search_space, y=sing, name='{} singular value'.format(ordinal(idx+1))
        )
        if idx <= 2:
            singular_line['text'] = [ 'Singular gap: {:.2f}'.format(singular_gap[rid, idx]) for rid in range(50) ]
        if idx > 3:
            singular_line['hoverinfo'] = 'none'
        all_trace.append(singular_line)
        if idx == 2:
            # HACK: just specify the color manually, need to generate each later.
            all_trace.append(go.Scatter(
                x=rad_search_space[start_idx:end_idx], y=singular_values[start_idx:end_idx,2],
                mode='lines',marker=dict(color='green'),
                showlegend=False, hoverinfo='none'
            ))
            all_trace.append(go.Scatter(
                x=rad_search_space[start_idx:end_idx], y=singular_values[start_idx:end_idx,1],
                fill='tonexty', mode='none', showlegend=False, hoverinfo='none'
            ))
    return all_trace

def generate_layouts(start_idx, end_idx, est_rad_dim1, est_rad_dim2, rad_search_space):
    return go.Layout(
        title='Singular values - radii plot',
        xaxis=dict(
            title='$\\text{Radius } r $',
            # type='log',
            autorange=True
        ),
        yaxis=dict(title='$\\text{Singular value } \\sigma$'),
        shapes=[{
            'type': 'rect',
            'xref': 'x',
            'yref': 'paper',
            'x0': rad_search_space[start_idx],
            'y0': 0,
            'x1': rad_search_space[end_idx-1],
            'y1': 1,
            'fillcolor': '#d3d3d3',
            'opacity': 0.4,
            'line': {
                'width': 0,
            }
        }],
        annotations=[
        dict(
            x=est_rad_dim1,
            y=0,
            xref='x',
            yref='y',
            text='$\\hat{r}_{d=1}$',
            font = dict(size = 30),
            showarrow=True,
            arrowhead=7,
            ax=20,
            ay=30
        ),
        dict(
            x=est_rad_dim2,
            y=0,
            xref='x',
            yref='y',
            text='$\\hat{r}_{d=2}$',
            font = dict(size = 30),
            showarrow=True,
            arrowhead=7,
            ax=-20,
            ay=30
        )
    ])
