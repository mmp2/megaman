.. _geom:

Geometry 
========

One of the fundamental objectives of manifold learning is to understand
the geometry of the data. As such the primary class of this package
is the geometry class:

:class:'~Mmani.geometry.Geometry'
    This class is used as the interface to compute various quantities
    on the original data set including: pairwise distance graphs,
    affinity matrices, and laplacian matrices. It also caches these
    quantities and allows for fast re-computation with new parameters.
    
API of Geometry 
---------------

The Geometry class is used to interface with functions that compute various
geometric quantities with respect to the original data set. This is the object
that is passed (or computed) within each embedding function. It is how
Mmani caches important quantities allowing for fast re-computation with
various new parameters. Geometry class has the following methods:

1. Class instantiation : during class instantiation you input the parameters
   concerning the original data matrix: the distance calculation method, 
   neighborhood and affinity radius, laplacian type and optional path
   to FLANN library if not installed to root. Data can be passed
   as raw data, pairwise distance matrix or pairwise affinity matrix.
2. `get_distance_matrix` : This function interfaces with the distance
   module to calculate the pairwise distance graph. A new (different)
   radius can be passed to this than the neighborhood_radius passed at 
   instantiation. If the original data was passed (instead of distance 
   or affinity) re-computation with a new radius is performed. 
3. `get_affinity_matrix` : This function takes the distance matrix
   and computes the Gaussian kernel using the affinity radius passed
   during instantiation. A new radius can be passed to this function and
   if the data was passed raw or as a distance the affinity can be
   recomputed. 
4. `get_laplacian_matrix` : This function takes the affinity matrix
   and computes the requested laplacian type. A different type can 
   be passed than was instantiated. Furthermore the scaling_epps and
   renormalization_exponent parameters can be adjusted here. If using
   the `geometric` graph laplacian it is suggested that the scaling_epps
   parameter be set to the affinity_radius that was used.
5. Assign functions `assign_data_matrix`, `assign_distance_matrix`,
   `assign_laplacian_matrix` : These functions allow you to manually
   assign matrices to be distance, affinity, or laplacian matrices. 
   The only checking that is performed is that the matrices are arrays of
   the appropriate dimension otherwise the validity is trusted to the user.
6. `assign_parameters` : You can use this function to assign parameters to
   the Geometry class if they weren't assigned at instantiation (for example
   if you assign your own laplacian). Note: self.neighborhood_radius, 
   self.affinity_radius, and self.laplacian_type refer to the CURRENT
   version of these matrices. If you want to re-calculate with a new parameter 
   DO NOT update these with assign_parameters, instead use get_distance_matrix(), 
   get_affinity_matrix(), or get_laplacian_matrix() and pass the desired new 
   parameter. This will automatically update the self.parameter version. 
   If you change these values with assign_parameters Geometry will assume
   that the existing matrix follows that parameter and so, for example,
   calling get_distance_matrix() with a passed radius will *not* 
   recalculate if the passed radius is equal to self.neighborhood_radius 
   and there already exists a distance matrix.

See the API documentation for further information. 

Example Usage 
-------------

Here is an example using the function on a random data set::

   import numpy as np
   import Mmani.geometry.geometry as geom
   
   X = np.random.randn(100, 10)
   Geometry = geom.Geometry(X, input_type = 'data', distance_method = 'cython',
                           neighborhood_radius = 4., affinity_radius = 4.,
                           laplacian_type = 'geometric')
   dist_mat = Geometry.get_distance_matrix() # default sparse csr
   affi_mat = Geometry.get_affinity_matrix() # default sparse csr
   lapl_mat = Geometry.get_laplacian_matrix(scaling_epps = 4.) # default sparse csr