# Author: Yu-Chia Chen <yuchaz@uw.edu>
# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE

import numpy as np

def _check_backend(backend):
    def decorator(func):
        def wrapper(*args,**kwargs):
            import warnings
            warnings.warn(
                'Be careful in using megaman.plotter modules'
                ' API will change in the next release.',
                FutureWarning
            )
            import pkgutil
            package = pkgutil.find_loader(backend)
            if package is not None:
                return func(*args,**kwargs)
            else:
                raise ImportError('plotting backend {} not installed'.format(backend))
        return wrapper
    return decorator

@_check_backend('matplotlib')
def get_colors_array(name,coloring,base255=True):
    from matplotlib import colors, cm
    cmap = cm.get_cmap(name=name)
    norm = colors.Normalize()
    normalized_coloring = norm(coloring)
    colors_array = (cmap(normalized_coloring)[:,:3]*255).astype(np.uint8) \
                   if base255 else cmap(normalized_coloring)
    return colors_array, normalized_coloring

def generate_plotly_colorscale(name,num=256):
    colormap, normalized_coloring = get_colors_array(name,np.arange(num))
    return [ [n_coloring, 'rgb({},{},{})'.format(*colormap[idx])] \
             for idx, n_coloring in enumerate(normalized_coloring) ]

def generate_colors_and_colorscale(name,coloring,**kwargs):
    colors_array, _ncor = get_colors_array(name,coloring)
    colorscale = generate_plotly_colorscale(name,**kwargs)
    return colors_array, colorscale

def generate_grid(size,num_groups=100):
    return np.arange(0,size,num_groups)

@_check_backend('plotly')
def plotly_layout(embedding):
    import plotly.graph_objs as go
    max_value = 1.2*np.max(np.absolute(embedding[:,:3]))
    axis_range = [-max_value,max_value]
    layout = go.Layout(
        title='Plot with ellipse',
        height=600,
        width=600,
        scene=dict(
            xaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)',
                range=axis_range,
            ),
            yaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)',
                range=axis_range,
            ),
            zaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)',
                range=axis_range,
            ),
        )
    )
    return layout

def get_top_two_sigma_norm(H,color_by_ratio=True):
    eigen_vals = np.array([ sorted_eigh(Hk)[0][:2] for Hk in H ])
    if color_by_ratio == True:
        toptwo_eigen_vals_norm = eigen_vals[:,1] / eigen_vals[:,0]
    else:
        toptwo_eigen_vals_norm = eigen_vals[:,0]
    return toptwo_eigen_vals_norm

def sorted_eigh(M):
    vals, vecs = np.linalg.eigh(M)
    return vals[::-1], vecs[:,::-1]
