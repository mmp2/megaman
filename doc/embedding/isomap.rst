.. _isomap:

Isomap
======

Isomap is one of the embeddings implemented in the megaman package.
Isomap uses Multidimensional Scaling (MDS) to preserve pairwsise
graph shortest distance computed using a sparse neighborhood graph.

For more information see:

* Tenenbaum, J.B.; De Silva, V.; & Langford, J.C.
  A global geometric framework for nonlinear dimensionality reduction.
  Science 290 (5500)

:class:'~megaman.embedding.Isomap'
    This class is used to interface with isomap embedding function.
    Like all embedding functions in megaman it operates using a
    Geometry object. The Isomap class allows you to optionally
    pass an exiting Geometry object, otherwise it creates one.

API of Isomap
-------------

The Isomap model, along with all the other models in megaman, have an API
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
   from megaman.geometry import Geometry
   from megaman.embedding import Isomap

   X = np.random.randn(100, 10)
   radius = 5
   adjacency_method = 'cyflann'
   adjacency_kwds = {'radius':radius} # ignore distances above this radius
   
   geom  = Geometry(adjacency_method=adjacency_method, adjacency_kwds=adjacency_kwds)
   
   isomap = Isomap(n_components=n_components, eigen_solver='arpack', geom=geom)
   embed_isomap = isomap.fit_transform(X)
