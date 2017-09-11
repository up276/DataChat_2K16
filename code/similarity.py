import re, math, operator
from heapq import nlargest

class Similarity:

	def dot_product(self, vector1, vector2):
		return sum(map(operator.mul, vector1, vector2))

	def cosine_sim(self, vector1, vector2):
		prod = self.dot_product(vector1, vector2)
		len1 = math.sqrt(self.dot_product(vector1, vector1))
		len2 = math.sqrt(self.dot_product(vector2, vector2))
		return prod / (len1 * len2)

	def find_cosine_sim(self, x, y):
		'''
		returns list of cosine similarities between vector y (user question) 
		and each row of array x (hardcoded question set)
		'''
		results = list()
		for i in range(x.shape[0]):
			results.append(self.cosine_sim(x[i],y))
		return results

	def find_best_match(self, x, y, nmax=1):
		'''
		find the nmax most similar questions to question y in question set x
		'''
		results = self.find_cosine_sim(x,y)
		best_match_index = []
		print "largest scores:"
		print nlargest(nmax, results)
		for score in nlargest(nmax, results): 
			best_match_index.append(results.index(score))
		best_match = x[best_match_index]
		return best_match, best_match_index

