# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE

import numpy as np
from numpy.testing import assert_allclose

from megaman.utils.testing import assert_raise_message
from megaman.geometry.geometry import Geometry
from megaman.embedding.base import BaseEmbedding


def test_geometry_dict():
    """ Test passing a dictionary and confirm the output """
    geom_dict = dict(adjacency_method = 'auto',
                     adjacency_kwds = {'radius':4},
                     affinity_method = 'auto',
                     affinity_kwds = {'radius':4},
                     laplacian_method = 'geometric',
                     laplacian_kwds = {'scaling_eps':4})
    g1 = Geometry(**geom_dict)
    base_embedding = BaseEmbedding(geom=geom_dict).fit_geometry()
    assert(g1.__dict__ == base_embedding.geom_.__dict__)


def test_geometry_object():
    """ Test passing a geometry object and confirm the output """
    g1 = Geometry(adjacency_method = 'auto',
                  adjacency_kwds = {'radius':4},
                  affinity_method = 'auto',
                  affinity_kwds = {'radius':4},
                  laplacian_method = 'geometric',
                  laplacian_kwds = {'scaling_eps':4})
    base_embedding = BaseEmbedding(geom=g1).fit_geometry()
    assert(g1.__dict__ == base_embedding.geom_.__dict__)


def test_geometry_update():
    """ Test passing geometry object then independently update a parameter and confirm that the embedding
        geometry is also updated """
    g1 = Geometry(adjacency_method = 'auto',
                  adjacency_kwds = {'radius':4},
                  affinity_method = 'auto',
                  affinity_kwds = {'radius':4},
                  laplacian_method = 'geometric',
                  laplacian_kwds = {'scaling_eps':4})
    base_embedding = BaseEmbedding(geom=g1)
    X = np.random.rand(10, 2)
    # Now update g1 -- object that was passed
    g1.set_data_matrix(X)
    # confirm internal object is updated
    assert_allclose(g1.X, base_embedding.geom.X)
