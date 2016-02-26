import numpy as np

from sklearn.utils.graph import graph_shortest_path
from scipy.sparse.csgraph import shortest_path
from numpy.testing import assert_array_almost_equal
from scipy import sparse

def generate_graph(N=20):
    rng = np.random.RandomState(0)
    dist_matrix = rng.random_sample((N, N))
    dist_matrix = dist_matrix + dist_matrix.T
    i = (rng.randint(N, size=N * N // 2), rng.randint(N, size=N * N // 2))
    dist_matrix[i] = 0
    dist_matrix.flat[::N + 1] = 0
    return dist_matrix


dist_matrix = sparse.csr_matrix(generate_graph(20))

# auto
graph_sk = graph_shortest_path(dist_matrix, directed = False)
graph_sp = shortest_path(dist_matrix, directed = False)
assert_array_almost_equal(graph_sk, graph_sp)

# Floyd-Warshall
graph_sk = graph_shortest_path(dist_matrix, directed = False, method = 'FW')
graph_sp = shortest_path(dist_matrix, directed = False, method = 'FW')
assert_array_almost_equal(graph_sk, graph_sp)


# Dijkstra's
graph_sk = graph_shortest_path(dist_matrix, directed = False, method = 'D')
graph_sp = shortest_path(dist_matrix, directed = False, method = 'D')
assert_array_almost_equal(graph_sk, graph_sp)


from sklearn.utils.sparsetools import connected_components
from scipy.sparse.csgraph import connected_components as c_c

dist_matrix = sparse.csr_matrix(generate_graph(100))
(n_sk, labs_sk) = connected_components(dist_matrix)
(n_sp, labs_sp) = c_c(dist_matrix)
