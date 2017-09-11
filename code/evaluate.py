import pandas as pd
import numpy as np
import similarity

class Evaluator():

	def predict(self, test_set, embedded_hardcoded_question_set, hardcoded_question_set, textEmbedder, idf_dictionnary):
		predictions = []
		for user_question in test_set['user_question']:
			user_question = user_question.split()
			embedded_user_question = textEmbedder.text_to_vec(user_question, idf_dictionnary)
			# perform cosine similarity and find best match
			cosineSim = similarity.Similarity()
			prediction, prediction_index = cosineSim.find_best_match(embedded_hardcoded_question_set,embedded_user_question)
			predictions.append(hardcoded_question_set['hardcoded_question'][prediction_index[0]])
		return predictions

	def merge_test_set_with_predictions(self, test_set, predictions):
		'''
		returns a data frame with columns 'hardcoded_questions', 'user_question', 'prediction' and 'correct' (0 or 1)
		'''
		# prediction correct = 1, incorrect = 0						
		correct = []
		for i in range(len(predictions)) :
			if predictions[i] == test_set['hardcoded_question'][i]:
				correct.append(1)
			else :
				correct.append(0)

		pred = pd.DataFrame({'prediction': predictions})
		corr = pd.DataFrame({'correct': correct})
		result = pd.concat([test_set, pred, corr], axis=1, join_axes=[test_set.index])
		return result

	def compute_accuracy(self, results):
		correct_answers = results['correct'].sum() # 0 or 1 for correct answer is at position 3 in results
		accuracy = correct_answers/float(results.shape[0])
		return accuracy

