.. _isomap:

Isomap
======

Isomap is one of the embeddings implemented in the Mmani package. 
Isomap uses Multidimensional Scaling (MDS) to preserve pairwsise
graph shortest distance computed using a sparse neighborhood graph.

For more information see: 

* Tenenbaum, J.B.; De Silva, V.; & Langford, J.C. 
  A global geometric framework for nonlinear dimensionality reduction. 
  Science 290 (5500)

:class:'~Mmani.embedding.Isomap'
    This class is used to interface with isomap embedding function. 
    Like all embedding functions in Mmani it operates using a
    Geometry object. The Isomap class allows you to optionally 
    pass an exiting Geometry object, otherwise it creates one.

API of Isomap
-------------

The Isomap model, along with all the other models in Mmani, have an API
designed to follow in the same vein of 
`scikit-learn <http://scikit-learn.org/>`_ API.

Consequentially, the Isomap class functions as follows

1. At class instantiation `.Isomap()` parameters are passed. See API 
   documementation for more information. An existing Geometry object
   can be passed to `.Isomap()`.
2. The `.fit()` method creates a Geometry object if one was not 
   already passed and then calculates th embedding. 
   The number of components and eigen solver can also be passed to the 
   `.fit()` function. Since Isomap caches important quantities 
   (like the graph distance matrix) which do not change by selecting 
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
   import Mmani.embedding.isomap as iso
   
   X = np.random.randn(100, 10)
   Geometry = geom.Geometry(X, input_type = 'data', distance_method = 'cython',
                           neighborhood_radius = 4., affinity_radius = 4.,
                           laplacian_type = 'geometric')
   Iso = iso.Isomap(n_components = 2, eigen_solver = 'arpack', Geometry = Geometry)
   embedding = Iso.fit_transform(X)