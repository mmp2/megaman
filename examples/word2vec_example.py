from word2vec import *
# Download the data at: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
# Change this to the location of the word2vec data set:
fname = '/homes/jmcq/Mmani/GoogleNews-vectors-negative300.bin.gz'

# This takes a few minutes:
model = Word2Vec.load_word2vec_format(fname, binary=True) # this loads the data

model.vector_size # the dimension of the word2vec projection
model.vocab # a dictionary of Vocab() objects with the word as the key
model.index2word # a list such that index2word[Vocab().index] = word
model.syn0 # (nwords, ndim) data corresponding to the words in index2word

# example:
my_word = model.vocab['king']
print(my_word.count) # the occurances of that word in the corpus
print(my_word.index) # the index of the word in 

print(model.index2word[my_word.index]) # the word itself ('king')

print(model.syn0[my_word.index]) # and the word2vec projection of 'king'

'''
    it might be interesting to use a subset of the data to project 
    before using all 3M samples
'''