from .adjacency import CyFLANNAdjacency, compute_adjacency_matrix
from scipy.sparse import vstack, hstack

def complete_adjacency_matrix(Dtrain, Xtrain, Xtest, train_index = None, radius=None, **kwargs):
    if train_index is None:
        Cyflann = CyFLANNAdjacency(radius=radius, **kwargs)
        train_index = Cyflann.build_index(Xtrain)
    else:
        Cyflann = CyFLANNAdjacency(radius=radius, flann_index = train_index, **kwargs)        
    test_train_adjacency = train_index.radius_neighbors_graph(Xtest, radius)
    test_test_adjacency = compute_adjacency_matrix(Xtest, method='cyflann', radius = radius, **kwargs)    
    train_adjacency = hstack([Dtrain, test_train_adjacency.transpose()])
    test_adjacency = hstack([test_train_adjacency, test_test_adjacency])    
    return vstack([train_adjacency, test_adjacency])