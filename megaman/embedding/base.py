""" base estimator class for megaman """

# Author: James McQueen  -- <jmcq@u.washington.edu>
# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE

import numpy as np
from scipy.sparse import isspmatrix

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array

from ..geometry.geometry import Geometry

# from sklearn.utils.validation import FLOAT_DTYPES
FLOAT_DTYPES = (np.float64, np.float32, np.float16)


class BaseEmbedding(BaseEstimator, TransformerMixin):
    """ Base Class for all megaman embeddings.

    Inherits BaseEstimator and TransformerMixin from sklearn.

    BaseEmbedding creates the common interface to the geometry
    class for all embeddings as well as providing a common
    .fit_transform().

    Parameters
    ----------
    n_components : integer
        number of coordinates for the manifold.
    radius : float (optional)
        radius for adjacency and affinity calculations. Will be overridden if
        either is set in `geom`
    geom : dict or megaman.geometry.Geometry object
        specification of geometry parameters: keys are
        ["adjacency_method", "adjacency_kwds", "affinity_method",
         "affinity_kwds", "laplacian_method", "laplacian_kwds"]

    Attributes
    ----------
    geom_ : a fitted megaman.geometry.Geometry object.
    """
    def __init__(self, n_components=2, radius=None, geom=None):
        self.n_components = n_components
        self.radius = radius
        self.geom = geom

    def _validate_input(self, X, input_type):
        if input_type == 'data':
            sparse_formats = None
        elif input_type in ['adjacency', 'affinity']:
            sparse_formats = ['csr', 'coo', 'lil', 'bsr', 'dok', 'dia']
        else:
            raise ValueError("unrecognized input_type: {0}".format(input_type))
        return check_array(X, dtype=FLOAT_DTYPES, accept_sparse=sparse_formats)

    # # The world is not ready for this...
    # def estimate_radius(self, X, input_type='data', intrinsic_dim=None):
    #     """Estimate a radius based on the data and intrinsic dimensionality
    #
    #     Parameters
    #     ----------
    #     X : array_like, [n_samples, n_features]
    #         dataset for which radius is estimated
    #     intrinsic_dim : int (optional)
    #         estimated intrinsic dimensionality of the manifold. If not
    #         specified, then intrinsic_dim = self.n_components
    #
    #     Returns
    #     -------
    #     radius : float
    #         The estimated radius for the fit
    #     """
    #     if input_type == 'affinity':
    #         return None
    #     elif input_type == 'adjacency':
    #         return X.max()
    #     elif input_type == 'data':
    #         if intrinsic_dim is None:
    #             intrinsic_dim = self.n_components
    #         mean_std = np.std(X, axis=0).mean()
    #         n_features = X.shape[1]
    #         return 0.5 * mean_std / n_features ** (1. / (intrinsic_dim + 6))
    #     else:
    #         raise ValueError("Unrecognized input_type: {0}".format(input_type))

    def fit_geometry(self, X=None, input_type='data'):
        """Inputs self.geom, and produces the fitted geometry self.geom_"""
        if self.geom is None:
            self.geom_ = Geometry()
        elif isinstance(self.geom, Geometry):
            self.geom_ = self.geom
        else:
            try:
                kwds = dict(**self.geom)
            except TypeError:
                raise ValueError("geom must be a Geometry instance or "
                                 "a mappable/dictionary")
            self.geom_ = Geometry(**kwds)

        if self.radius is not None:
            self.geom_.set_radius(self.radius, override=False)

        # if self.radius == 'auto':
        #     if X is not None and input_type != 'affinity':
        #         self.geom_.set_radius(self.estimate_radius(X, input_type),
        #                               override=False)
        # else:
        #     self.geom_.set_radius(self.radius,
        #                           override=False)

        if X is not None:
            self.geom_.set_matrix(X, input_type)

        return self

    def fit_transform(self, X, y=None, input_type='data'):
        """Fit the model from data in X and transform X.

        Parameters
        ----------
        input_type : string, one of: 'data', 'distance' or 'affinity'.
            The values of input data X. (default = 'data')
        X: array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

        If self.input_type is 'distance':

        X : array-like, shape (n_samples, n_samples),
            Interpret X as precomputed distance or adjacency graph
            computed from samples.

        Returns
        -------
        X_new: array-like, shape (n_samples, n_components)
        """
        self.fit(X, y=y, input_type=input_type)
        return self.embedding_

    def transform(self, X, y=None, input_type='data'):
        raise NotImplementedError("transform() not implemented. "
                                  "Try fit_transform()")
