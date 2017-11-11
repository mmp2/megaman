# Author: Yu-Chia Chen <yuchaz@uw.edu>
#
#
# Transformed 2D patches to 3D after:
#         Till Hoffmann <t.hoffmann13@imperial.ac.uk>
#         https://stackoverflow.com/questions/18228966/how-can-matplotlib-2d-patches-be-transformed-to-3d-with-arbitrary-normals
# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE

import numpy as np
from .utils import _check_backend

def covar_plotter3d_matplotlib(embedding, rieman_metric, inspect_points_idx, ax,
                               colors):
    """3 Dimensional Covariance plotter using matplotlib backend."""
    for pts_idx in inspect_points_idx:
        plot_ellipse_matplotlib( cov=rieman_metric[pts_idx],
                             pos=embedding[pts_idx],
                             ax=ax, ec='k', lw=1,
                             color=colors[pts_idx] )
    return ax

@_check_backend('matplotlib')
def plot_ellipse_matplotlib(cov, pos, ax, nstd=2, **kwargs):
    """
    Plot 2d ellipse in 3d using matplotlib backend
    """
    from matplotlib.patches import Ellipse
    from mpl_toolkits.mplot3d import art3d, Axes3D
    ellipse_param, normal = calc_2d_ellipse_properties(cov,nstd)
    ellipse_kwds = merge_keywords(ellipse_param, kwargs)
    ellip = Ellipse(xy=(0,0), **ellipse_kwds)
    ax.add_patch(ellip)

    ellip = pathpatch_2d_to_3d(ellip, normal=normal)
    ellip = pathpatch_translate(ellip,pos)
    return ellip

def merge_keywords(x,y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z

def pathpatch_2d_to_3d(pathpatch, z = 0, normal = 'z'):
    """
    Transforms a 2D Patch to a 3D patch using the given normal vector.

    The patch is projected into they XY plane, rotated about the origin
    and finally translated by z.
    """
    if type(normal) is str: #Translate strings to normal vectors
        index = "xyz".index(normal)
        normal = np.roll((1.0,0,0), index)

    normal /= np.linalg.norm(normal) #Make sure the vector is normalised

    path = pathpatch.get_path() #Get the path and the associated transform
    trans = pathpatch.get_patch_transform()

    path = trans.transform_path(path) #Apply the transform

    pathpatch.__class__ = art3d.PathPatch3D #Change the class
    pathpatch._code3d = path.codes #Copy the codes
    pathpatch._facecolor3d = pathpatch.get_facecolor #Get the face color

    verts = path.vertices #Get the vertices in 2D

    d = np.cross(normal, (0, 0, 1)) #Obtain the rotation vector
    M = rotation_matrix(d) #Get the rotation matrix

    pathpatch._segment3d = \
        np.array([np.dot(M, (x, y, 0)) + (0, 0, z) for x, y in verts])
    return pathpatch

def pathpatch_translate(pathpatch, delta):
    """
    Translates the 3D pathpatch by the amount delta.
    """
    pathpatch._segment3d += delta
    return pathpatch

def covar_plotter3d_plotly(embedding, rieman_metric, inspect_points_idx,
                           colors, **kwargs):
    """3 Dimensional Covariance plotter using matplotlib backend."""
    def rgb2hex(rgb):
        return '#%02x%02x%02x' % tuple(rgb)
    return [ plt_data for idx in inspect_points_idx
             for plt_data in plot_ellipse_plotly(
                rieman_metric[idx], embedding[idx],
                color=rgb2hex(colors[idx]), **kwargs) ]

def plot_ellipse_plotly(cov, pos, nstd=2,**kwargs):
    """Plot 2d ellipse in 3d using matplotlib backend"""
    ellipse_param, normal = calc_2d_ellipse_properties(cov,nstd)
    points = create_ellipse(**ellipse_param)

    points = transform_to_3d(points, normal=normal)
    points = translate_in_3d(points,pos)
    return create_ellipse_mesh(points,**kwargs)

def calc_2d_ellipse_properties(cov,nstd=2):
    """Calculate the properties for 2d ellipse given the covariance matrix."""
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    vals, vecs = eigsorted(cov)
    width, height = 2 * nstd * np.sqrt(vals[:2])
    normal = vecs[:,2] if vecs[2,2] > 0 else -vecs[:,2]
    d = np.cross(normal, (0, 0, 1))
    M = rotation_matrix(d)
    x_trans = np.dot(M,(1,0,0))
    cos_val = np.dot(vecs[:,0],x_trans)/np.linalg.norm(vecs[:,0])/np.linalg.norm(x_trans)
    theta = np.degrees(np.arccos(np.clip(cos_val, -1, 1))) # if you really want the angle
    return { 'width': width, 'height': height, 'angle': theta }, normal

def rotation_matrix(d):
    """
    Calculates a rotation matrix given a vector d. The direction of d
    corresponds to the rotation axis. The length of d corresponds to
    the sin of the angle of rotation.

    Variant of: http://mail.scipy.org/pipermail/numpy-discussion/2009-March/040806.html
    """
    sin_angle = np.linalg.norm(d)

    if sin_angle == 0:
        return np.identity(3)

    d /= sin_angle

    eye = np.eye(3)
    ddt = np.outer(d, d)
    skew = np.array([[    0,  d[2],  -d[1]],
                     [-d[2],     0,   d[0]],
                     [ d[1], -d[0],     0]], dtype=np.float64)

    M = ddt + np.sqrt(1 - sin_angle**2) * (eye - ddt) + sin_angle * skew
    return M

def create_ellipse(width,height,angle):
    """Create parametric ellipse from 200 points."""
    angle = angle / 180.0 * np.pi
    thetas = np.linspace(0,2*np.pi,200)
    a = width / 2.0
    b = height / 2.0

    x = a*np.cos(thetas)*np.cos(angle) - b*np.sin(thetas)*np.sin(angle)
    y = a*np.cos(thetas)*np.sin(angle) + b*np.sin(thetas)*np.cos(angle)
    z = np.zeros(thetas.shape)
    return np.vstack((x,y,z)).T

def transform_to_3d(points,normal,z=0):
    """Project points into 3d from 2d points."""
    d = np.cross(normal, (0, 0, 1))
    M = rotation_matrix(d)
    transformed_points = M.dot(points.T).T + z
    return transformed_points

def translate_in_3d(points,delta):
    """Translate the points in 2d by amount of delta"""
    return points + delta

@_check_backend('plotly')
def create_ellipse_mesh(points,**kwargs):
    """Visualize the ellipse by using the mesh of the points."""
    import plotly.graph_objs as go
    x,y,z = points.T
    return (go.Mesh3d(x=x,y=y,z=z,**kwargs),
            go.Scatter3d(x=x, y=y, z=z,
                         marker=dict(size=0.01),
                         line=dict(width=2,color='#000000'),
                         showlegend=False,
                         hoverinfo='none'
            )
    )
