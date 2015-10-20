benchmarks/bench_embed_with_rmetric.py in the works by MMP (not added)
benchmarks/bench_laplacian_dense_temp.py added, to be deleted later

JMCQ:
Created Development Branch for compiling into working library.

MMP: to do list October 2014

1. accelerate distance computation with flann
   may require revising the internal implementation of sparse laplacian
2. make proper package
3. evaluate e-value/vector computation
4. lazy rmetric

(the next tasks can be in any order. a * means that matlab implementation exists)

5* epsilon optimization
6  dimension estimation
7  representation with multiple patches
8* distance and area computations
9* gaussian processes
10* directed embedding and directed laplacian
11* visualisations

