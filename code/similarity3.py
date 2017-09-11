'''
Find similarity using Skpi-thought vectors
'''

import numpy as np


class Similarity3():

	def sentence_cosine_sim(self, v1, v2):
		'''
		find pairwise cosine similarity between 2 vectors
		'''
		cosine_similarity = np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))
		return cosine_similarity


	def find_similarity(self, vector1, vector2):
		similarity_score = self.sentence_cosine_sim(vector1, vector1)
		return similarity_score






