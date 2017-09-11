'''
Script loading google word2vec pretrained vectors and fine-training them on training set (hardcoded questions set)
(allows also to embed new words such as technical words)

Run this script again when working on a different data set
'''


import numpy as np
import gensim


print('Loading fasttext pretrained embeddings...')
self.model = gensim.models.Word2Vec.load_word2vec_format('../../model/GoogleNews-vectors-negative300.bin', binary=True) 
#self.model = fasttext.load_model('/home/urjit/Capstone_2K16/fastText/model.bin')
self.num_features = 300 # 300 if using word2vec (word2vec embedding dimension)