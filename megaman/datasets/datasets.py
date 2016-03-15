"""Some sample datasets"""
from __future__ import division

import os

import numpy as np
from scipy import ndimage
from sklearn.utils import check_random_state


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
