""" base estimator class for megaman """

# Author: James McQueen  -- <jmcq@u.washington.edu>
# License: BSD 3 clause (C) 2016

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, FLOAT_DTYPES

from ..geometry.geometry import Geometry

class BaseEmbedding(BaseEstimator, TransformerMixin):
    """ Base Class for all megaman embeddings.

    Inherits BaseEstimator and TransformerMixin from sklearn.

    BaseEmbedding creates the common interface to the geometry
    class for all embeddings as well as providing a common
    .fit_transform().

    Parameters
    ----------
    geom :  either a Geometry object from megaman.geometry or a dictionary
            containing (some or all) geometry parameters: adjacency_method,
            adjacency_kwds, affinity_method, affinity_kwds, laplacian_method,
            laplacian_kwds as keys.

    Attributes
    ----------
    geom : a fitted megaman.geometry.Geometry object.

    """
    def __init__(self, geom):
        self.geom = geom

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

        if X is not None:
            if input_type == 'data':
                X = check_array(X, dtype=FLOAT_DTYPES)
                self.geom_.set_data_matrix(X)
            elif input_type == 'adjacency':
                X = check_array(X, dtype=FLOAT_DTYPES,
                                accept_sparse=['csr', 'csc', 'coo'])
                self.geom_.set_adjacency_matrix(X)
            elif input_type == 'affinity':
                X = check_array(X, dtype=FLOAT_DTYPES,
                                accept_sparse=['csr', 'csc', 'coo'])
                self.geom_.set_affinity_matrix(X)
            else:
                raise ValueError("Unrecognized input_type: "
                                 "{0}".format(input_type))
        return self

    def fit_transform(self, X, input_type='data'):
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
        self.fit(X, input_type=input_type)
        return self.embedding_
