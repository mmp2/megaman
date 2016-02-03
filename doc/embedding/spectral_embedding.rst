.. _spectral_embedding:

Spectral Embedding
==================

Spectral Embedding is on of the methods implemented in the Mmani package.
Spectral embedding (and diffusion maps) uses the spectrum (eigen vectors
and eigen values) of a graph Laplacian estimated from the data set. There
are a number of different graph Laplacians that can be used.

For more information see:

* A Tutorial on Spectral Clustering, 2007
  Ulrike von Luxburg
  http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.165.9323

:class:'~Mmani.embedding.SpectralEmbedding'
    This class is used to interface with spectral embedding function. 
    Like all embedding functions in Mmani it operates using a
    Geometry object. The Isomap class allows you to optionally 
    pass an exiting Geometry object, otherwise it creates one.

API of Spectral Embedding
-------------------------

The Spectral Embedding model, along with all the other models in Mmani, 
have an API designed to follow in the same vein of 
`scikit-learn <http://scikit-learn.org/>`_ API.

Consequentially, the LTSA class functions as follows

1. At class instantiation `.SpectralEmbedding()` parameters are passed. See API 
   documementation for more information. An existing Geometry object
   can be passed to `.SpectralEmbedding()`. Here is also where
   you have the option to use diffusion maps.
2. The `.fit()` method creates a Geometry object if one was not 
   already passed and then calculates th embedding. 
   The eigen solver can also be passed to the 
   `.fit()` function. WARNING: NOT COMPLETED
   Since Geometry caches important quantities 
   (like the graph Laplacian) which do not change by selecting 
   different eigen solvers and this can be passed
   and a new embedding computed without re-computing existing quantities.
   the `.fit()` function does not return anything but it does create
   the attribute `self.embedding_` only one `self.embedding_` exists
   at a given time. If a new embedding is computed the old one is overwritten.
3. The `.fit_transform()` function calls the `fit()` function and returns
   the embedding. It does not allow for changing parameters.     

See the API documentation for further information.

Example Usage
-------------

Here is an example using the function on a random data set::

   import numpy as np
   import Mmani.geometry.geometry as geom
   import Mmani.embedding.spectral_embedding as se
   
   X = np.random.randn(100, 10)
   Geometry = geom.Geometry(X, input_type = 'data', distance_method = 'cython',
                           neighborhood_radius = 4., affinity_radius = 4.,
                           laplacian_type = 'geometric')
   Spectral = se.SpectralEmbedding(n_components = 2, eigen_solver = 'pyamg', 
                                   Geometry = Geometry)
   embedding = Spectral.fit_transform(X)