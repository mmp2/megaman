""" base estimator class for megaman """

# Author: James McQueen  -- <jmcq@u.washington.edu>
# License: BSD 3 clause (C) 2016

from sklearn.base import BaseEstimator, TransformerMixin
from ..geometry.geometry_new import Geometry

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
		self.geom = self.fit_geometry(geom)
	
	def fit_geometry(self, geom):
		if isinstance(geom, Geometry):
			self.geom = geom
		elif isinstance(geom, dict):
			# Unpack geometry parameter:
			if 'adjacency_method' in geom.keys():
				adjacency_method = geom['adjacency_method']
			else:
				adjacency_method = 'auto'
			if 'adjacency_kwds' in geom.keys():
				adjacency_kwds = geom['adjacency_kwds']
			else:
				adjacency_kwds = {}
			if 'affinity_method' in geom.keys():
				affinity_method = geom['affinity_method']
			else:
				affinity_method = 'auto'
			if 'affinity_kwds' in geom.keys():
				affinity_kwds = geom['affinity_kwds']
			else:
				affinity_kwds = {}
			if 'laplacian_method' in geom.keys():
				laplacian_method = geom['laplacian_method']
			else:
				laplacian_method = 'auto'
			if 'laplacian_kwds' in geom.keys():
				laplacian_kwds = geom['laplacian_kwds']
			else:
				laplacian_kwds = {}
			# instantiate geometry class
			self.geom = Geometry(adjacency_method=adjacency_method, adjacency_kwds=adjacency_kwds,
								 affinity_method=affinity_method, affinity_kwds=affinity_kwds,
								 laplacian_method=laplacian_method, laplacian_kwds=laplacian_kwds)
		else:
			raise ValueError("must pass either Geometry object or dictionary of parameters")
		return self.geom
		
	def fit_transform(self, X):
		self.fit(X)
		return self.embedding_