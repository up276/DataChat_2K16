'''
Script to embed hardcoded question database (data side questions) using Google pretrained word2vec + CBOW

inspired from https://github.com/Poyuli/sentiment.analysis/blob/master/sentiment.py
'''

import question_embedding 
import numpy as np

textEmbedder =  question_embedding.TextEmbedder()

# word2vec embedding dimension
num_features = 300


def gen_text_vecs(questions, model, num_features):
    """
    Function which generates a m-by-n numpy array from all questions,
    where m is len(questions), and n is num_feature
    
    Input:
            questions: a list of lists. 
                     Inner lists are words from each question.
                     Outer lists consist of all questions
            model: trained word2vec model
            num_feature: dimension of word2vec vectors
    Output: m-by-n numpy array, where m is len(questions) and n is num_feature
    """

    curr_index = 0
    review_feature_vecs = np.zeros((len(questions), num_features), dtype="float32")

    for question in questions:

       if curr_index%1000 == 0.:
           print "Vectorizing questions %d of %d" % (curr_index, len(questions))
   
       review_feature_vecs[curr_index] = textEmbedder.text_to_vec(review, model, num_features)
       curr_index += 1
       
	return review_feature_vecs








