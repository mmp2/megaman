"""General tests for embeddings"""

# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE

from itertools import product

import numpy as np
from numpy.testing import assert_raises, assert_allclose

from megaman.embedding import (Isomap, LocallyLinearEmbedding,
                               LTSA, SpectralEmbedding)
from megaman.geometry.geometry import Geometry

EMBEDDINGS = [Isomap, LocallyLinearEmbedding, LTSA, SpectralEmbedding]

# # TODO: make estimator_checks pass!
# def test_estimator_checks():
#     from sklearn.utils.estimator_checks import check_estimator
#     for Embedding in EMBEDDINGS:
#         yield check_estimator, Embedding


def test_embeddings_fit_vs_transform():
    rand = np.random.RandomState(42)
    X = rand.rand(100, 5)
    geom = Geometry(adjacency_kwds = {'radius':1.0},
                    affinity_kwds = {'radius':1.0})

    def check_embedding(Embedding, n_components):
        model = Embedding(n_components=n_components,
                          geom=geom, random_state=rand)
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
        # no radius set
        embedding = Embedding()
        assert_raises(ValueError, embedding.fit, X)

        # unrecognized geometry
        embedding = Embedding(radius=2, geom='blah')
        assert_raises(ValueError, embedding.fit, X)

    for Embedding in EMBEDDINGS:
        yield check_bad_args, Embedding
