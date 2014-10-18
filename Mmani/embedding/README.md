Core functionality
------------------
geometry.py:   nearest neighbors, distance matrix, affinity matrix, laplacian
	       and related functions
rmetric.py:    class RMetric the riemannian metric
graph.py:      from scikit-learn. not sure we need it. the "old" embedding
	       functions from scikit-learn do.


embed_with_rmetric.py: all steps from data to embedding with Riemannian metric
		        packaged in one functionn


__init__.py

Embedding algorithms:
--------------------
spectral_embedding_.py: written by MMP, uses Mmani functions

copied from scikit-learn
------------------------
isomap-copy.py
locally_linear.py
mds.py

isomap.py:	rewriting in progress by MMP


TODO:
-----
(see also the doc comments inside each function)
     * put calling the eigensolver in a function of its own (almost all functions here call it, with different sets of kwargs)
     * rewrite mds, isomap, ltsa to work with Mmani. mds, isomap -- only dense version, ltsa - sparse. we can also implement lle.