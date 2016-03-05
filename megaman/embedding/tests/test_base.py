import numpy as np
from megaman.utils.testing import assert_raise_message
from megaman.geometry.geometry import Geometry
from megaman.embedding.base import BaseEmbedding

def test_geometry_dict():
	""" Test passing a dictionary and confirm the output """
	adjacency_method = 'auto'
	adjacency_kwds = {'radius':4}
	affinity_method = 'auto'
	affinity_kwds = {'radius':4}
	laplacian_method = 'geometric'
	laplacian_kwds = {'scaling_eps':4}
	geom_dict = {'adjacency_method':adjacency_method,
				 'adjacency_kwds':adjacency_kwds,
				 'affinity_method':affinity_method,
				 'affinity_kwds':affinity_kwds,
				 'laplacian_method':laplacian_method,
				 'laplacian_kwds':laplacian_kwds}
	g1 = Geometry(adjacency_method=adjacency_method, adjacency_kwds=adjacency_kwds,
				  affinity_method=affinity_method, affinity_kwds=affinity_kwds,
				  laplacian_method=laplacian_method, laplacian_kwds=laplacian_kwds)
	base_embedding = BaseEmbedding(geom_dict)
	assert(g1.__dict__ == base_embedding.geom.__dict__)
	
def test_geometry_object():
	""" Test passing a geometry object and confirm the output """
	adjacency_method = 'auto'
	adjacency_kwds = {'radius':4}
	affinity_method = 'auto'
	affinity_kwds = {'radius':4}
	laplacian_method = 'geometric'
	laplacian_kwds = {'scaling_eps':4}
	g1 = Geometry(adjacency_method=adjacency_method, adjacency_kwds=adjacency_kwds,
				  affinity_method=affinity_method, affinity_kwds=affinity_kwds,
				  laplacian_method=laplacian_method, laplacian_kwds=laplacian_kwds)
	base_embedding = BaseEmbedding(g1)
	assert(g1.__dict__ == base_embedding.geom.__dict__)

def test_geometry_update():
	""" Test passing geometry object then independently update a parameter and confirm that the embedding
		geometry is also updated """
	adjacency_method = 'auto'
	adjacency_kwds = {'radius':4}
	affinity_method = 'auto'
	affinity_kwds = {'radius':4}
	laplacian_method = 'geometric'
	laplacian_kwds = {'scaling_eps':4}
	g1 = Geometry(adjacency_method=adjacency_method, adjacency_kwds=adjacency_kwds,
				  affinity_method=affinity_method, affinity_kwds=affinity_kwds,
				  laplacian_method=laplacian_method, laplacian_kwds=laplacian_kwds)
	base_embedding = BaseEmbedding(g1)
	X = np.random.rand(10, 2)
	# Now update g1 -- object that was passed
	g1.set_data_matrix(X)
	# confirm internal object is updated 
	assert(np.all(g1.X == base_embedding.geom.X))
