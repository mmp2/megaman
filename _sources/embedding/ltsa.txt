.. _ltsa:

Local Tangent Space Alignment
=============================

Local Tangent Space Alignment is one of the methods implemented in the megaman package.
Local Tangent Space Alighment uses independent estimates of the local tangent
space at each point and then uses a global alignment procedure with a
unit-scale condition to create a single embedding from each local tangent
space.

For more information see:

* Zhang, Z. & Zha, H. Principal manifolds and nonlinear
  dimensionality reduction via tangent space alignment.
  Journal of Shanghai Univ.  8:406 (2004)

:class:'~megaman.embedding.LTSA'
    This class is used to interface with local tangent space
    alignment embedding function.
    Like all embedding functions in megaman it operates using a
    Geometry object. The Locally Linear class allows you to optionally
    pass an exiting Geometry object, otherwise it creates one.


API of Local Tangent Space Alignment
------------------------------------

The Locally Tangent Space Alignment model, along with all the other models in megaman,
have an API designed to follow in the same vein of
`scikit-learn <http://scikit-learn.org/>`_ API.

Consequentially, the LTSA class functions as follows

1. At class instantiation `.LTSA()` parameters are passed. See API
   documementation for more information. An existing Geometry object
   can be passed to `.LTSA()`.
2. The `.fit()` method creates a Geometry object if one was not
   already passed and then calculates th embedding.
   The eigen solver can also be passed to the
   `.fit()` function. WARNING: NOT COMPLETED
   Since LTSA caches important quantities
   (like the local tangent spaces) which do not change by selecting
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
   from megaman.geometry import Geometry
   from megaman.embedding import (Isomap, LocallyLinearEmbedding, LTSA, SpectralEmbedding)

   X = np.random.randn(100, 10)
   radius = 5
   adjacency_method = 'cyflann'
   adjacency_kwds = {'radius':radius} # ignore distances above this radius
   
   geom  = Geometry(adjacency_method=adjacency_method, adjacency_kwds=adjacency_kwds)
   
   ltsa =LTSA(n_components=n_components, eigen_solver='arpack', geom=geom)
   embed_ltsa = ltsa.fit_transform(X)
