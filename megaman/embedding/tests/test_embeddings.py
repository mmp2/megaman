"""General tests for embeddings"""
from itertools import product

import numpy as np
from numpy.testing import assert_raises, assert_allclose

from megaman.embedding import (Isomap, LocallyLinearEmbedding,
                               LTSA, SpectralEmbedding)
from megaman.geometry import Geometry

EMBEDDINGS = [Isomap, LocallyLinearEmbedding, LTSA, SpectralEmbedding]

def test_embeddings_fit_vs_transform():
    rand = np.random.RandomState(42)
    X = rand.rand(100, 5)
    geom = Geometry(X, neighborhood_radius=1.0)

    def check_embedding(Embedding, n_components):
        model = Embedding(n_components=n_components,
                          Geometry=geom, random_state=rand)
        embedding = model.fit_transform(X)
        assert model.embedding_.shape == (X.shape[0], n_components)
        assert_allclose(embedding, model.embedding_)

    for Embedding in EMBEDDINGS:
        for n_components in [1, 2, 3]:
            yield check_embedding, Embedding, n_components


def test_embeddings_bad_arguments():
    rand = np.random.RandomState(32)
    X = rand.rand(100, 3)

    def check_bad_args(Embedding):
        model = Embedding(n_components=2, Geometry='blah')
        assert_raises(ValueError, model.fit, X)

    for Embedding in EMBEDDINGS:
        yield check_bad_args, Embedding
