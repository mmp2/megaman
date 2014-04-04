import numpy as np
X = np.random.random((1000, 5))
#from sklearn.manifold import LocallyLinearEmbedding
from Mmani.embedding import LocallyLinearEmbedding
model = LocallyLinearEmbedding(15, 2)
model.fit(X)
Y = model.transform(X)
Y.shape
X.shape

