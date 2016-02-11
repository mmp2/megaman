Mmani: Scalable manifold learning
=================================

Mmani is a scalable manifold learning package implemented in 
python. It has a front-end API designed to be familiar
to `scikit-learn <http://scikit-learn.org/>`_ but harnesses
the C++ Fast Library for Approximate Nearest Neighbors (FLANN)
and the Sparse Symmetric Positive Definite (SSPD) solver
Locally Optimal Block Precodition Gradient (LOBPCG) method
to scale manifold learning algorithms to large data sets.
It is designed for researchers and as such caches intermediary
steps and indices to allow for fast re-computation with new parameters.

For issues & contributions, see the source repository
`on github <http://github.com/mmp2/Mmani/>`_.

Documentation
=============

.. toctree::
   :maxdepth: 2

   geometry/index
   embedding/index
   utils/index
   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

