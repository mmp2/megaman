.. _locally_linear:

Locally Linear Embedding
========================

Locally linear embedding is one of the methods implemented in the Mmani package.
Locally Linear Embedding uses reconstruction weights estiamted on the original
data set to produce an embedding that preserved the original reconstruction 
weights. 

For more information see:

* Roweis, S. & Saul, L. Nonlinear dimensionality reduction 
  by locally linear embedding.  Science 290:2323 (2000).

:class:'~Mmani.embedding.LocallyLinearEmbedding'
    This class is used to interface with locally linear embedding function. 
    Like all embedding functions in Mmani it operates using a
    Geometry object. The Locally Linear class allows you to optionally 
    pass an exiting Geometry object, otherwise it creates one.


API of Locally Linear Embedding
-------------------------------

The Locally Linear model, along with all the other models in Mmani, have an API
designed to follow in the same vein of 
`scikit-learn <http://scikit-learn.org/>`_ API.

Consequentially, the Locally Linear class functions as follows

1. At class instantiation `.LocallyLinear()` parameters are passed. See API 
   documementation for more information. An existing Geometry object
   can be passed to `.LocallyLinear()`.
2. The `.fit()` method creates a Geometry object if one was not 
   already passed and then calculates th embedding. 
   The number of components and eigen solver can also be passed to the 
   `.fit()` function. WARNING: NOT COMPLETED
   Since LocallyLinear caches important quantities 
   (like the barycenter weight matrix) which do not change by selecting 
   different eigen solvers and embeding dimension these can be passed
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
   import Mmani.embedding.locally_linear as lle
   
   X = np.random.randn(100, 10)
   Geometry = geom.Geometry(X, input_type = 'data', distance_method = 'cython',
                           neighborhood_radius = 4., affinity_radius = 4.,
                           laplacian_type = 'geometric')
   LLE = lle.LocallyLinear(n_components = 2, eigen_solver = 'arpack', Geometry = Geometry)
   embedding = LLE.fit_transform(X)