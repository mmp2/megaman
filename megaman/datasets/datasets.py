"""Some sample datasets"""
from __future__ import division

import os

import numpy as np
from scipy import ndimage
from sklearn.utils import check_random_state

import collections

def get_megaman_image(factor=1):
    """Return an RGBA representation of the megaman icon"""
    imfile = os.path.join(os.path.dirname(__file__), 'megaman.png')
    data = ndimage.imread(imfile) / 255
    if factor > 1:
        data = data.repeat(factor, axis=0).repeat(factor, axis=1)
    return data


def generate_megaman_data(sampling=2):
    """Generate 2D point data of the megaman image"""
    data = get_megaman_image()
    x = np.arange(sampling * data.shape[1]) / float(sampling)
    y = np.arange(sampling * data.shape[0]) / float(sampling)
    X, Y = map(np.ravel, np.meshgrid(x, y))
    C = data[np.floor(Y.max() - Y).astype(int),
             np.floor(X).astype(int)]
    return np.vstack([X, Y]).T, C


def _make_S_curve(x, range=(-0.75, 0.75)):
    """Make a 2D S-curve from a 1D vector"""
    assert x.ndim == 1
    x = x - x.min()
    theta = 2 * np.pi * (range[0] + (range[1] - range[0]) * x / x.max())
    X = np.empty((x.shape[0], 2), dtype=float)
    X[:, 0] = np.sign(theta) * (1 - np.cos(theta))
    X[:, 1] = np.sin(theta)
    X *= x.max() / (2 * np.pi * (range[1] - range[0]))
    return X


def generate_megaman_manifold(sampling=2, nfolds=2,
                              rotate=True, random_state=None):
    """Generate a manifold of the megaman data"""
    X, c = generate_megaman_data(sampling)
    for i in range(nfolds):
        X = np.hstack([_make_S_curve(x) for x in X.T])

    if rotate:
        rand = check_random_state(random_state)
        R = rand.randn(X.shape[1], X.shape[1])
        U, s, VT = np.linalg.svd(R)
        X = np.dot(X, U)

    return X, c

def generate_noisefree_hourglass(n_size, scaling_factor=1.75, seed=None):
    if seed is not None:
        np.random.seed(seed)
    fz = lambda z: -4*z**4 + 4*z**2 + 1
    X = np.random.normal(0,1,[n_size,3])
    sphere = X / np.linalg.norm(X,axis=1)[:,None]
    r = np.linalg.norm(sphere,axis=1)

    x,y,z = sphere.T
    theta = np.arctan2(y,x)
    phi = np.arccos(z/r)

    r_hour = fz(z)
    theta_hour = theta
    z_hour = z
    phi_hour = np.arccos(z_hour/r_hour)

    x_hour = r_hour*np.cos(theta_hour)*np.sin(phi_hour)
    y_hour = r_hour*np.sin(theta_hour)*np.sin(phi_hour)
    z_hour = r_hour*np.cos(phi_hour)

    x_hour *= 0.5
    y_hour *= 0.5

    hourglass = np.vstack((x_hour,y_hour,z_hour)).T
    hourglass *= scaling_factor

    return hourglass

def _genereate_noises(sigmas, size, dimensions, seed=None):
    if seed is not None:
        np.random.seed(seed)
    if isinstance(sigmas, (collections.Sequence, np.ndarray)):
        assert len(sigmas) == dimensions, \
            'The size of sigmas should be the same as noises dimensions'
        return np.random.multivariate_normal(np.zeros(dimensions),
                                             np.diag(sigmas), size)
    else:
        return np.random.normal(0,sigmas,[size,dimensions])

def _add_noises_on_primary_dimensions(data,sigmas=0.1,seed=None):
    size,dim = data.shape
    noises = _genereate_noises(sigmas,size,dim)
    return data + noises

def _add_noises_on_additional_dimensions(data,addition_dims,sigmas=1,seed=None):
    if addition_dims == 0:
        return data
    else:
        noises = _genereate_noises(sigmas,data.shape[0],addition_dims,seed)
        return np.hstack((data,noises))

def generate_noisy_hourglass(size, sigma_primary=0.05, addition_dims=0,
                             sigma_additional=0.1, scaling_factor=1.75, seed=None):
    hourglass = generate_noisefree_hourglass(size, scaling_factor, seed)
    hourglass = _add_noises_on_primary_dimensions(hourglass, sigma_primary)
    hourglass = _add_noises_on_additional_dimensions(hourglass, addition_dims,
                                                     sigma_additional)
    return hourglass
