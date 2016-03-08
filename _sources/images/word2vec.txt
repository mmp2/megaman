.. _word2vec:

Word2Vec Plot
====================

.. figure:: word2vec_rmetric_plot_no_digits.png
   :scale: 50 %
   :alt: word2vec embedding with R. metric

   A three-dimensional embedding of the main sample of galaxy spectra
   from the Sloan Digital Sky Survey (approximately 675,000 spectra
   observed in 3750 dimensions). Colors in the above figure indicate
   the strength of Hydrogen alpha emission, a very nonlinear feature
   which requires dozens of dimensions to be captured in a linear embedding.
   
   3,000,000 words and phrases mapped by word2vec using Google News into 300
   dimensions. The data was then embedded into 2 dimensions using Spectral
   Embedding. The plot shows a sample of 10,000 points displaying the overall
   shape of the embedding as well as the estimated "stretch" 
   (i.e. dual push-forward Riemannian metric) at various locations in the embedding. 