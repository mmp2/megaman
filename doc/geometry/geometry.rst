.. _geom:

Geometry
========

One of the fundamental objectives of manifold learning is to understand
the geometry of the data. As such the primary class of this package
is the geometry class:

:class:'~megaman.geometry.Geometry'
    This class is used as the interface to compute various quantities
    on the original data set including: pairwise distance graphs,
    affinity matrices, and laplacian matrices. It also caches these
    quantities and allows for fast re-computation with new parameters.

API of Geometry
---------------

The Geometry class is used to interface with functions that compute various
geometric quantities with respect to the original data set. This is the object
that is passed (or computed) within each embedding function. It is how
megaman caches important quantities allowing for fast re-computation with
various new parameters. Beyond instantiation, the Geometry class offers
three types of functions: compute, set & delete that work with the four
primary data matrices: (raw) data, adjacency matrix, affinity matrix,
and Laplacian Matrix. 

1. Class instantiation : during class instantiation you input the parameters
   concerning the original data matrix such as the distance calculation method,
   neighborhood and affinity radius, laplacian type. Each of the three
   computed matrices (adjacency, affinity, laplacian) have their
   own keyword dictionaries which permit these methods to easily be extended.
2. `set_[some]_matrix` : these functions allow you to assign a matrix of data
   to the geometry object. In particular these are used to fit the geometry
   to your input data (which may be of the form data_matrix, adjacency_matrix,
   or affinity_matrix). You can also set a Laplacian matrix. 
3. `compute_[some]_matrix` : these functions are designed to compute the 
   selected matrix (e.g. adjacency). Additional keyword arguments can be
   passed which override the ones passed at instantiation. NB: this method
   will always re-compute a matrix.
4. Geometry Attributes. Other than the parameters passed at instantiation each
   matrix that is computed is stored as an attribute e.g. geom.adjacency_matrix,
   geom.adjacency_matrix, geom.laplacian_matrix. Raw data is stored as geom.X.
   If you want to query for these matrices without recomputing you should use
   these attributes e.g. my_affinity = geom.affinity_matrix. 
5. `delete_[some]_matrix` : if you are working with large data sets and choose
    an algorithm (e.g. Isomap or Spectral Embedding) that do not require the
	original data_matrix, these methods can be used to clear memory. 

See the API documentation for further information.

Example Usage
-------------

Here is an example using the function on a random data set::

   import numpy as np
   from megaman.geometry import Geometry

   X = np.random.randn(100, 10)
   radius = 5
   adjacency_method = 'cyflann'
   adjacency_kwds = {'radius':radius} # ignore distances above this radius
   affinity_method = 'gaussian'
   affinity_kwds = {'radius':radius} # A = exp(-||x - y||/radius^2) 
   laplacian_method = 'geometric'
   laplacian_kwds = {'scaling_epps':radius} # scaling ensures convergence to Laplace-Beltrami operator
   
   geom  = Geometry(adjacency_method=adjacency_method, adjacency_kwds=adjacency_kwds,
                    affinity_method=affinity_method, affinity_kwds=affinity_kwds,
                    laplacian_method=laplacian_method, laplacian_kwds=laplacian_kwds)