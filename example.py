import numpy as np
X = np.random.random((1000, 5))

isSP = True
#from sklearn.manifold import LocallyLinearEmbedding
if isSP:
    from Mmani.embedding import SpectralEmbedding
    model = SpectralEmbedding(neighbors_radius=1.)
else:
    from Mmani.embedding import LocallyLinearEmbedding
    model = LocallyLinearEmbedding(15, 2)

model.fit(X)
Y = model.transform(X)
Y.shape
X.shape

#se = manifold.SpectralEmbedding(n_components=n_components,
#                                n_neighbors=n_neighbors)
#Y = se.fit_transform(X)

