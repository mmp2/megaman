# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE
from .adjacency import CyFLANNAdjacency, compute_adjacency_matrix
from scipy.sparse import vstack, hstack

def complete_adjacency_matrix(Dtrain, Xtrain, Xtest, adjacency_kwds):
    if 'cyflann_kwds' in adjacency_kwds.keys():
        cyflann_kwds = adjacency_kwds['cyflann_kwds']
    else:
        cyflann_kwds = {}
    radius = adjacency_kwds['radius']
    Cyflann = CyFLANNAdjacency(radius=radius, **cyflann_kwds)
    train_index = Cyflann.build_index(Xtrain)
    test_train_adjacency = train_index.radius_neighbors_graph(Xtest, radius)
    test_test_adjacency = compute_adjacency_matrix(Xtest, method='cyflann', **adjacency_kwds)    
    train_adjacency = hstack([Dtrain, test_train_adjacency.transpose()])
    test_adjacency = hstack([test_train_adjacency, test_test_adjacency])    
    return vstack([train_adjacency, test_adjacency])