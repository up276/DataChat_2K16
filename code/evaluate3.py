import pandas as pd
import numpy as np
import similarity2
from scipy.spatial.distance import cdist
import sys; sys.path.append('/Users/vincentchabot/Desktop/capstone/skip_thoughts/skip-thoughts-master/')
import skipthoughts

class Evaluator3():

	def __init__(self, hardcoded_question_set, keep_pos=['NN', 'NNS', 'JJ', 'VB']):
		print 'loading skip-thoughts model...'
		self.model = skipthoughts.load_model()
		print 'embedding hardcoded questions...'
		X = hardcoded_question_set['hardcoded_question'].tolist()
		self.vectors_hardcoded = skipthoughts.encode(self.model, X, verbose=False)
		# POS to keep
		self.keep_pos = keep_pos


	def softmax(self, x):
		"""Compute softmax values for each sets of scores in x."""
		return np.exp(x) / np.sum(np.exp(x), axis=0)


	def pseudo_distance(self, x, y):
		return np.linalg.norm(x - np.dot(x, y)*y/np.linalg.norm(y))


	def find_best_mult_match(self, hardcoded_question_set, user_question, W):
		dim = hardcoded_question_set['hardcoded_question'].shape[0]
		scores = []
		#print user_question
		i = 0
		v1 = skipthoughts.encode(self.model, [user_question], verbose=False)
		for v2 in self.vectors_hardcoded:
			#print hardcoded_question_set['hardcoded_question'][i]
			cosine_similarity = np.dot(v1, np.dot(W,v2))/(np.linalg.norm(v1) * np.linalg.norm(v2))
			np.dot(v1, np.dot(W,v2))
			scores.append(cosine_similarity)
			i += 1 
		
		#print 's', scores
		s = self.softmax(scores)

		print "scores", s
		
		# Keep track of questions indexes when sorting s in descending order
		index =range(len(s))
		s_ind = zip(s, index)
		s_ind.sort(reverse=True)
		new_index = [ind for value, ind in s_ind]
		p = ['p%i' % i for i in xrange(1,dim+1)]
		df = pd.DataFrame(s.tolist())
		df = df.transpose()
		df.columns = p
		df = df.apply(lambda s: s.sort_values(ascending=False).values, axis=1)

		M = np.transpose([map(lambda x: 1./x if x >= i else 0, xrange(1, dim+1)) for i in xrange(1, dim+1)])

		df["pds"] = pd.DataFrame(cdist(df[p], M, self.pseudo_distance)).apply(pd.Series.argmin, axis=1)+1

		# number of matches to return
		n = df["pds"].iloc[0]
		#print 'n', n
		if n > 3:
			best_match = ['fallback']
			best_score = [0]
		else:
			best_match = hardcoded_question_set['hardcoded_question'].iloc[new_index[:n]].tolist()
			best_score = sorted(s, reverse=True)[:n]

		return best_match, best_score


	def find_best_match(self, hardcoded_question_set, user_question, W):
		scores = []
		v1 = skipthoughts.encode(self.model, user_question, verbose=False)
		for v2 in self.vectors_hardcoded:
			cosine_similarity = np.dot(v1, np.dot(W,v2))/(np.linalg.norm(v1) * np.linalg.norm(v2))
			scores.append(cosine_similarity)
		best_score = max(scores)
		best_match = hardcoded_question_set['hardcoded_question'][scores.index(best_score)]
		return [best_match], [best_score]


	def predict(self, test_set, hardcoded_question_set, projection_matrix='identity', mult_answ = True):
		if projection_matrix=='identity':
			embedding_length = skipthoughts.encode(model, ['radom'], verbose=False).shape[1]
			W = np.identity(embedding_length)
		else:
			W=projection_matrix
		predictions = []
		scores = []
		for user_question in test_set['user_question']:
			if mult_answ == False:
				# perform cosine similarity and find best match
				best_match, best_score = self.find_best_match(hardcoded_question_set, user_question, W)
				predictions.append(best_match)
				scores.append(best_score)
			elif mult_answ == True:
				# perform cosine similarity and find best match
				best_match, best_score = self.find_best_mult_match(hardcoded_question_set, user_question, W)				
				predictions.append(best_match)
				scores.append(best_score)
		return predictions, scores


	def merge_test_set_with_predictions(self, test_set, predictions, scores):
		'''
		returns a data frame with columns 'hardcoded_questions', 'user_question', 'prediction' and 'correct' (0 or 1)
		'''
		# prediction correct = 1, incorrect = 0						
		correct = []
		for i in range(len(predictions)) :
			#print 'predictions[i]', predictions[i]
			if test_set['hardcoded_question'].iloc()[i] in predictions[i]:
				correct.append(1)
			else :
				correct.append(0)

		pred = pd.DataFrame({'prediction': predictions})
		scor = pd.DataFrame({'scores': scores})
		corr = pd.DataFrame({'correct': correct})
		result = pd.concat([test_set, pred, scor, corr], axis=1, join_axes=[test_set.index])
		return result


	def compute_accuracy(self, results):
		correct_answers = results['correct'].sum() # 0 or 1 for correct answer is at position 3 in results
		accuracy = correct_answers/float(results.shape[0])
		return accuracy

