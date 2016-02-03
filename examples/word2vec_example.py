#!/usr/bin/env python
import sys
import numpy as np

sys.path.append('/homes/jmcq/Mmani')
from Mmani.geometry.geometry import Geometry
from Mmani.embedding.spectral_embedding import spectral_embedding
from scipy.io import mmwrite, mmread
from word2vec import *

# Download the data at: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
# Change this to the location of the word2vec data set:
fname = '/homes/jmcq/GoogleNews-vectors-negative300.bin.gz'

# This takes a few minutes:
'''
model = Word2Vec.load_word2vec_format(fname, binary=True) # this loads the data
'''

# example usage:
'''
model.vector_size # the dimension of the word2vec projection
model.vocab # a dictionary of Vocab() objects with the word as the key
model.index2word # a list such that index2word[Vocab().index] = word
model.syn0 # (nwords, ndim) data corresponding to the words in index2word

my_word = model.vocab['king']
print(my_word.count) # the occurances of that word in the corpus
print(my_word.index) # the index of the word in 
print(model.index2word[my_word.index]) # the word itself ('king')
print(model.syn0[my_word.index]) # and the word2vec projection of 'king'
'''

# If you need to re-run the distance calculation:
'''
# convert to float64 (aka 'd' = double)
X = np.array(model.syn0, dtype = 'd')
(nwords, ndim) = X.shape

# we want the radius to be big enough that each data point has at least 1 nbr
radius = 20 

# instantiate the Geometry class
Geom = Geometry(X, neighborhood_radius = radius, affinity_radius = radius, 
                distance_method = 'cython', input_type = 'data', 
                laplacian_type = 'geometric')
# Call the get distance -- no need to copy
dists = Geom.get_distance_matrix(copy=False)

# check to see if there are any isolated points:
n_no_nbrs = 0
for i in range(3000000):
    if dists[i,:].nnz == 1:
        n_no_nbrs += 1
# for radius = 20 this is 0

# Save it out
mmwrite( 'wor2vec_distance_radius_20.mtx', dists )
'''


# To read back:
print('loading distance data...')
dists = mmread('wor2vec_distance_radius_20.mtx') # note this loads as COO and takes a few minutes
dists = dists.tocsr() # convert to CSR
radius = 20

sub_n = 50000 # for testing
print('subsetting data...')
dists = dists[:sub_n, :sub_n]
print('instantiating Geometry...')
Geom = Geometry(dists, neighborhood_radius = radius, affinity_radius = radius, 
                distance_method = 'cython', input_type = 'distance', 
                laplacian_type = 'geometric')
print('computing Laplacian...')
Lapl = Geom.get_laplacian_matrix(scaling_epps = radius, copy = False, return_lapsym = True)

print('computing embedding...')
embed = spectral_embedding(Geom, n_components = 4, eigen_solver = 'amg')

print('saving embedding...')
mmwrite( 'wor2vec_embedding_radius_20_d_6_subset.mtx', embed )

import matplotlib
matplotlib.use('Agg')
import pylab as plt

print('making pairwise plot...')

fig, axes = plt.subplots(nrows=2, ncols = 3, figsize=(8,8))
fig.subplots_adjust(hspace=0.05,wspace =0.05)

axes[0, 0].scatter(embed[:, 0], embed[:, 1], s = 1, c = 'k')
axes[0, 0].set_title('1 vs 2')
axes[0, 0].xaxis.set_visible(False)
axes[0, 0].yaxis.set_visible(False)

axes[0, 1].scatter(embed[:, 0], embed[:, 2], s = 1, c = 'k')
axes[0, 1].set_title('1 vs 3')
axes[0, 1].xaxis.set_visible(False)
axes[0, 1].yaxis.set_visible(False)

axes[0, 2].scatter(embed[:, 0], embed[:, 3], s = 1, c = 'k')
axes[0, 2].set_title('1 vs 4')
axes[0, 2].xaxis.set_visible(False)
axes[0, 2].yaxis.set_visible(False)

axes[1, 0].scatter(embed[:, 1], embed[:, 2], s = 1, c = 'k')
axes[1, 0].set_title('2 vs 3')
axes[1, 0].xaxis.set_visible(False)
axes[1, 0].yaxis.set_visible(False)

axes[1, 1].scatter(embed[:, 1], embed[:, 3], s = 1, c = 'k')
axes[1, 1].set_title('2 vs 4')
axes[1, 1].xaxis.set_visible(False)
axes[1, 1].yaxis.set_visible(False)

axes[1, 2].scatter(embed[:, 2], embed[:, 3], s = 1, c = 'k')
axes[1, 2].set_title('3 vs 4')
axes[1, 2].xaxis.set_visible(False)
axes[1, 2].yaxis.set_visible(False)
plt.suptitle("pairwise components from spectral embedding into 4 dimensions")

print('saving figure...')
plt.savefig('word2vec_pairwise_embedding'+'.png', format='png')
print('done!')